import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import random
from tokenizers import Tokenizer
import math

os.environ['TORCHDYNAMO_VERBOSE'] = '1'

# 设置随机种子，确保结果可复现
SEED = 37
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# 确保 CUDA 相关的随机性也被控制 (如果使用 CUDA)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# 配置参数
class Config:
    """训练和模型配置"""
    batch_size = 128
    embedding_dim = 512  # Transformer 的嵌入维度 (d_model)
    d_feedforward = 2048   # Transformer 的 feedforward 维度 (d_ff)
    num_layers = 6      # Transformer 编码器和解码器的层数
    num_heads = 8       # Multi-Head Attention 头数 (n_head)
    learning_rate = 1e-4
    num_epochs = 20
    max_length = 80 # 最大序列长度 (包括 SOS 和 EOS token)
    dropout = 0.1
    save_path = "./model_checkpoints" # 模型检查点保存目录
    tensorboard_log_dir = "./logs" # TensorBoard 日志目录

    # 词汇表大小，会在数据加载后由 tokenizer 自动设置
    src_vocab_size = 0
    tgt_vocab_size = 0
    # 特殊标记 ID，会在数据加载后由 tokenizer 自动设置
    pad_id = 0
    sos_id = 0
    eos_id = 0
    unk_id = 0

    def __init__(self):
        # 设置设备
        # 支持 CUDA、Intel XPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.xpu.is_available():
            self.device = torch.device('xpu')
        else:
            self.device = torch.device('cpu')
        print(f"使用设备: {self.device}")

# 实例化配置
config = Config()

# 检查并创建保存目录
os.makedirs(config.save_path, exist_ok=True)
os.makedirs(config.tensorboard_log_dir, exist_ok=True)

# 1. 数据集加载和预处理
class TranslationDataset(Dataset):
    """自定义翻译数据集"""
    def __init__(self, file_path):
        self.data = []
        self.max_length = config.max_length

        # 加载预训练的BPE tokenizer (确保 tokenizer.json 文件存在)
        # 通常 tokenizer 应该先独立训练好并保存
        try:
            self.tokenizer = Tokenizer.from_file("tokenizer.json")
        except FileNotFoundError:
            print("Error: tokenizer.json not found. Please train or download the tokenizer first.")
            raise

        # 获取特殊标记的ID
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.sos_id = self.tokenizer.token_to_id("[CLS]")  # 使用[CLS]作为<sos>
        self.eos_id = self.tokenizer.token_to_id("[SEP]")  # 使用[SEP]作为<eos>
        self.unk_id = self.tokenizer.token_to_id("[UNK]")

        # 将特殊标记ID更新到 config 中
        config.pad_id = self.pad_id
        config.sos_id = self.sos_id
        config.eos_id = self.eos_id
        config.unk_id = self.unk_id

        # 设置词汇表大小（源语言和目标语言共享同一个 tokenizer）
        vocab_size = self.tokenizer.get_vocab_size()
        config.src_vocab_size = vocab_size
        config.tgt_vocab_size = vocab_size
        print(f"Vocabulary size: {vocab_size}")

        # 创建索引到token的映射 (重要，后续翻译时会用到)
        self.idx_to_token = {idx: token for token, idx in self.tokenizer.get_vocab().items()}

        # 读取数据
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc=f"Loading data from {os.path.basename(file_path)}"):
            try:
                item = json.loads(line.strip())
                # 不对英文进行小写处理，与tokenizer训练时保持一致
                # tokenizers 默认会处理 normalization，取决于 tokenizer 的配置
                src_text = item.get('english', '')
                tgt_text = item.get('chinese', '')

                if not src_text or not tgt_text:
                    continue # 跳过空行或不完整的行

                self.data.append((src_text, tgt_text))

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not parse line: {line.strip()}. Error: {e}")

        print(f"Loaded {len(self.data)} pairs from {os.path.basename(file_path)}")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]

        # 使用tokenizer编码文本
        # add_special_tokens=False 是为了手动控制 SOS/EOS 的添加
        src_encoding = self.tokenizer.encode(src_text, add_special_tokens=False) # 英文
        tgt_encoding = self.tokenizer.encode(tgt_text, add_special_tokens=False) # 中文

        src_ids = src_encoding.ids
        tgt_ids = tgt_encoding.ids

        # 截断以留出空间给 sos 和 eos
        # max_length 是最终 tensor 的长度，所以实际文本 token 长度是 max_length - 2
        src_ids = src_ids[:self.max_length - 2]
        tgt_ids = tgt_ids[:self.max_length - 2]

        # 添加开始和结束标记
        src_indices = [self.sos_id] + src_ids + [self.eos_id]
        tgt_indices = [self.sos_id] + tgt_ids + [self.eos_id]

        # 确保最终长度是 max_length
        src_indices = src_indices + [self.pad_id] * (self.max_length - len(src_indices))
        tgt_indices = tgt_indices + [self.pad_id] * (self.max_length - len(tgt_indices))

        # 再次截断，以防万一长度超出 max_length (理论上不会，除非初始 token 数量 > max_length - 2)
        src_indices = src_indices[:self.max_length]
        tgt_indices = tgt_indices[:self.max_length]


        return {
            'src': torch.tensor(src_indices, dtype=torch.long),
            'tgt': torch.tensor(tgt_indices, dtype=torch.long),
        }


# 2. 数据加载器 Collate 函数
class CollateFn:
    """批量处理函数，将样本列表转换为批次 tensor"""
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        """
        Args:
            batch: 一个列表，其中每个元素是 TranslationDataset 的 __getitem__ 返回的字典。
                   例如: [{'src': tensor([...]), 'tgt': tensor([...])}, ...]
        Returns:
            包含批次 tensor 的字典。
        """
        # 从批次中提取 src 和 tgt tensor 列表
        src_batch = [item['src'] for item in batch]
        tgt_batch = [item['tgt'] for item in batch]

        # 这些 tensor 已经由 Dataset padded 到 max_length 了，
        # 所以这里直接堆叠 (stack) 即可，无需 pad_sequence
        src_batch = torch.stack(src_batch, dim=0) # 形状: (batch_size, max_length)
        tgt_batch = torch.stack(tgt_batch, dim=0) # 形状: (batch_size, max_length)

        # 将批次数据移动到指定的设备
        return {
            'src': src_batch.to(config.device),
            'tgt': tgt_batch.to(config.device),
        }

# 3. 位置编码
class PositionEncoding(nn.Module):
    """
    位置编码模块

    为输入序列的 token embedding 添加位置信息。
    位置编码与 embedding vector 的维度相同，通过相加的方式与 embedding 结合。
    """
    def __init__(self, d_model: int, max_len=5000, drop_out=0.1):
        """
        初始化位置编码模块
        Args:
            d_model: embedding 维度，必须与输入 token embedding 的维度一致。
            max_len: 支持的最大序列长度，决定了 pe 矩阵的大小。建议设置一个较大的值。
                     这里的 max_len 应该 >= config.max_length
            drop_out: 应用于位置编码后的 dropout 比率。
        """
        super().__init__()
        # dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(drop_out)

        # 创建 positional encoding 矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # 生成位置索引 (0, 1, ..., max_len-1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算分母部分，使用对数空间避免数值问题
        # div_term 形状为 (d_model / 2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 应用 sin 到偶数位置
        pe[:, 0::2] = torch.sin(position * div_term)
        # 应用 cos 到奇数位置
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加一个 batch 维度，形状变为 (1, max_len, d_model)
        # 这样在 forward 中可以利用 PyTorch 的广播机制与输入 x (batch_size, seq_len, d_model) 相加
        pe = pe.unsqueeze(0)

        # 将 pe 注册为一个 buffer
        # buffer 是 state_dict 的一部分，但不会被优化器更新 (不是 learnable parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播函数，将位置编码添加到输入 tensor。
        Args:
            x: 输入 tensor，形状通常为 (batch_size, seq_len, d_model)。
        Returns:
            添加了位置编码并应用了 dropout 的 tensor，形状与输入 x 相同。
        """
        # 检查输入的序列长度是否超出最大支持长度
        if x.size(1) > self.pe.size(1):
             raise ValueError(f"Input sequence length ({x.size(1)}) exceeds maximum positional encoding length ({self.pe.size(1)}). Increase max_len in PositionEncoding.")

        # 将位置编码 (1, max_len, d_model) 切片到与输入 x 的 seq_len 匹配
        # 然后与输入 x (batch_size, seq_len, d_model) 相加。
        # PyTorch 会自动广播第一个维度。
        # 注意：根据 Transformer 论文，通常会在添加位置编码前对 embedding 进行缩放，这里在 TranslatModel 中处理。
        x = x + self.pe[:, :x.size(1), :]
        # 应用 dropout
        return self.dropout(x)

# 4. 定义 TranslatModel 模型
class TranslatModel(nn.Module):
    """
    基于 PyTorch 内置 nn.Transformer 模块构建的序列到序列 (Seq2Seq) Transformer 模型，
    用于机器翻译任务。
    """
    def __init__(self, vocabulary_size: int, d_model=512, n_head=8, n_layers=6, d_ff=2048, drop_out=0.1, max_len=5000, pad_idx=0, sos_id=0, eos_id=0):
        """
        初始化 TranslatModel 模型。
        Args:
            vocabulary_size: 源语言和目标语言的词汇表大小 (假设源和目标使用相同的词汇表)。
            d_model: embedding 维度和模型内部表示的维度。
            n_head: 多头注意力机制中的头数。
            n_layers: 编码器和解码器堆叠的层数。
            d_ff: 前馈网络 (Feedforward Network) 中隐藏层的维度。
            drop_out: dropout 比率。
            max_len: 位置编码支持的最大序列长度。
            pad_idx: 填充 token (padding) 在词汇表中的索引。用于生成 padding masks。
            sos_id: 起始 token (SOS) 的索引。
            eos_id: 结束 token (EOS) 的索引。
        """
        super().__init__()
        self.d_model = d_model
        self.vocabulary_size = vocabulary_size
        self.pad_idx = pad_idx # 保存 padding 索引
        self.sos_id = sos_id   # 保存 SOS 索引
        self.eos_id = eos_id   # 保存 EOS 索引

        # Token embedding 层 (源语言和目标语言共享 embedding)
        self.embedding = nn.Embedding(vocabulary_size, d_model, padding_idx=pad_idx) # 指定 padding_idx 可以优化性能和显存

        # 位置编码层
        # PositionEncoding 的 max_len 应大于等于实际使用的最大序列长度
        self.positional_encoding = PositionEncoding(d_model, max_len=max_len, drop_out=drop_out)

        # PyTorch 内置的 Transformer 模型
        # 它包含了 num_encoder_layers 个 TransformerEncoderLayer
        # 和 num_decoder_layers 个 TransformerDecoderLayer
        # batch_first=True 参数非常重要，它指定输入 tensor 的形状为 (batch_size, seq_len, feature_dim)
        self.transformer = nn.Transformer(
            d_model=d_model,            # 模型维度
            nhead=n_head,               # 注意力头数
            num_encoder_layers=n_layers,  # 编码器层数
            num_decoder_layers=n_layers,  # 解码器层数
            dim_feedforward=d_ff,       # 前馈网络隐藏层维度
            dropout=drop_out,           # dropout 比率
            batch_first=True            # 设置为 True 以支持 (batch, seq, feature) 形状
        )

        # 最后的线性层，将 Transformer 解码器的输出维度 (d_model) 映射回词汇表大小
        # 用于预测下一个 token 的概率分布
        self.generator = nn.Linear(d_model, vocabulary_size)

        # 初始化参数 (通常采用 Xavier 或 Glorot 初始化)
        self._reset_parameters()

    def _reset_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                # 初始化权重
                nn.init.xavier_uniform_(p)
        # 特殊处理 embedding 层，padding 索引的 embedding 通常初始化为零
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)

    # 帮助函数：生成 PyTorch Transformer 所需的各种掩码
    def generate_mask(self, src, tgt):
        """
        生成 PyTorch Transformer 训练阶段需要的掩码。

        Args:
            src: 源语言输入 tensor，形状 (batch_size, src_seq_len)。
            tgt: 目标语言输入 tensor，形状 (batch_size, tgt_seq_len)。

        Returns:
            src_key_padding_mask: 编码器自注意力及解码器编码-解码注意力的 padding 掩码。
                                   形状 (batch_size, src_seq_len)。True 表示对应位置是 padding，应被忽略。
            tgt_key_padding_mask: 解码器自注意力的 padding 掩码。
                                   形状 (batch_size, tgt_seq_len)。True 表示对应位置是 padding，应被忽略。
            tgt_mask: 解码器自注意力的因果掩码 (causal mask / look-ahead mask)。
                     形状 (tgt_seq_len, tgt_seq_len)。用于防止解码器在预测当前 token 时“看到”未来 token。
                     通常是一个上三角矩阵，上三角部分为 True (表示掩盖)。
            memory_key_padding_mask: 解码器编码-解码注意力的 memory padding 掩码，等同于 src_key_padding_mask。
        """
        # src_key_padding_mask: 根据源语言输入 src 找出 padding 位置 (值为 self.pad_idx)
        # (src == self.pad_idx) 会生成一个 boolean tensor，其中 src 中等于 pad_idx 的位置为 True
        # PyTorch Transformer 的 key_padding_mask 参数中，True 表示忽略该位置
        src_key_padding_mask = (src == self.pad_idx) # 形状: (batch_size, src_seq_len)

        # tgt_key_padding_mask: 根据目标语言输入 tgt 找出 padding 位置
        # 用于解码器自注意力，忽略目标序列中的 padding
        tgt_key_padding_mask = (tgt == self.pad_idx) # 形状: (batch_size, tgt_seq_len)

        # tgt_mask: 生成目标语言序列的因果掩码 (attn_mask)
        # 这个掩码是一个上三角矩阵，确保解码器在时间步 i 只能attend到时间步 j <= i 的位置。
        # generate_square_subsequent_mask(L) 生成一个 (L, L) 的 tensor，其中上三角部分为 True。
        # PyTorch Transformer 的 attn_mask 参数中，True 表示需要掩盖的位置。
        tgt_seq_len = tgt.size(1)
        # Note: generate_square_subsequent_mask 默认生成 float 类型，需要确保和 input/pe 在同一 device
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device) # 形状: (tgt_seq_len, tgt_seq_len)

        # memory_key_padding_mask: 用于解码器 Encoder-Decoder Attention。
        # Decoder Q, Encoder KV。需要掩盖 Encoder 输出 (memory) 中对应源语言 padding 的位置。
        # 因此，它就是 src_key_padding_mask。
        memory_key_padding_mask = src_key_padding_mask

        # 标准的机器翻译模型中，src_mask (encoder self-attention attn_mask) 和 memory_mask (encoder-decoder attention attn_mask) 通常为 None。
        return src_key_padding_mask, tgt_key_padding_mask, tgt_mask, memory_key_padding_mask

    def forward(self, src, trg):
        """
        模型在训练阶段的前向传播函数 (使用 Teacher Forcing)。

        Args:
            src: 源语言输入序列 tensor，形状 (batch_size, src_seq_len)，包含 token 索引。
                 应该包含 <sos> 和 <eos>，并 padded。
            trg: 目标语言输入序列 tensor，形状 (batch_size, trg_seq_len)，包含 token 索引。
                 在训练时，通常是目标序列加上一个起始符 (<sos>)，但不包含结束符 (<eos>)，并 padded。
                 注意：PyTorch 的 Transformer 输入 trg 对应的是 decoder 的输入序列 (带有 <sos>)，
                 其输出的每个时间步是预测 *下一个* token。

        Returns:
            Transformer 解码器的输出 logits，形状 (batch_size, trg_seq_len, vocabulary_size)。
            这些 logits 可以直接用于计算交叉熵损失。
            输出的第 i 个位置预测的是 trg 序列的第 i+1 个 token。
        """
        # 1. 生成 PyTorch Transformer 所需的各种掩码
        # generate_mask 需要完整的 src 和 trg tensors
        src_key_padding_mask, tgt_key_padding_mask, tgt_mask, memory_key_padding_mask = self.generate_mask(src, trg)

        # 2. 将 token 索引转换为 embedding 向量
        # src_emb = self.embedding(src) 形状: (batch_size, src_seq_len, d_model)
        # trg_emb = self.embedding(trg) 形状: (batch_size, trg_seq_len, d_model)

        # 3. 对 embedding 进行缩放并添加位置编码
        # 根据 "Attention Is All You Need" 论文，对 embedding 进行缩放 (乘以 sqrt(d_model))
        # 再与位置编码相加，有助于平衡两者对模型的影响。
        src_emb = self.positional_encoding(self.embedding(src) * math.sqrt(self.d_model)) # 形状: (batch, src_seq_len, d_model)
        trg_emb = self.positional_encoding(self.embedding(trg) * math.sqrt(self.d_model)) # 形状: (batch, trg_seq_len, d_model)

        # 4. 将处理后的 embedding 和掩码输入到 PyTorch Transformer 模型中
        # transformer 的输入参数 (注意 key_padding_mask 中 True 表示忽略)：
        # - src: 源序列 embedding (batch, src_seq_len, d_model)
        # - tgt: 目标序列 embedding (batch, trg_seq_len, d_model)
        # - src_mask: 编码器自注意力掩码 (本例为 None)
        # - tgt_mask: 解码器自注意力因果掩码 (tgt_seq_len, tgt_seq_len)
        # - memory_mask: 解码器编码-解码注意力掩码 (本例为 None)
        # - src_key_padding_mask: 编码器输入 padding 掩码 (batch, src_seq_len)
        # - tgt_key_padding_mask: 解码器输入 padding 掩码 (batch, trg_seq_len)
        # - memory_key_padding_mask: 编码器输出 padding 掩码 (即 src_key_padding_mask) (batch, src_seq_len)
        transformer_output = self.transformer(
            src=src_emb,
            tgt=trg_emb,
            tgt_mask=tgt_mask,                   # 解码器自注意力 attn_mask (因果掩码)
            src_key_padding_mask=src_key_padding_mask, # 编码器自注意力 key_padding_mask
            tgt_key_padding_mask=tgt_key_padding_mask, # 解码器自注意力 key_padding_mask
            memory_key_padding_mask=memory_key_padding_mask # 编码器-解码器注意力 key_padding_mask
        ) # transformer_output 形状: (batch_size, trg_seq_len, d_model)

        # 5. 通过最终的线性层将 Transformer 输出映射到词汇表大小，得到 logits
        # 形状: (batch_size, trg_seq_len, vocabulary_size)
        logits = self.generator(transformer_output)

        # 返回 logits。在训练时，可以使用 nn.CrossEntropyLoss，它内部会自动进行 Softmax/LogSoftmax。
        return logits

    @torch.no_grad() # 确保在推理时不会计算梯度
    def translate(self, src, max_length=None):
        """
        使用模型进行翻译推断 (贪婪搜索)。

        Args:
            src: 源语言输入序列 tensor，形状 (batch_size, src_seq_len)，包含 token 索引。
                 应该包含 <sos> 和 <eos>，并 padded。
            max_length: 生成目标序列的最大长度。如果为 None，则使用 config.max_length。

        Returns:
            一个列表，其中每个元素是对应批次输入样本的预测 token ID 列表。
        """
        self.eval() # 设置模型为评估模式
        max_length = max_length if max_length is not None else config.max_length
        batch_size = src.size(0)

        # 1. 计算源序列的 padding mask
        src_key_padding_mask = (src == self.pad_idx) # 形状: (batch_size, src_seq_len)
        # memory_key_padding_mask 是 src_key_padding_mask
        memory_key_padding_mask = src_key_padding_mask

        # 2. 编码源序列
        # src_emb = self.embedding(src) 形状: (batch_size, src_seq_len, d_model)
        # 添加位置编码和缩放
        src_emb = self.positional_encoding(self.embedding(src) * math.sqrt(self.d_model))
        # 通过编码器获取 memory (encoder output)
        # src_mask 在 Encoder 自注意力中，通常为 None (如果不需要额外限制)
        encoder_output = self.transformer.encoder(
            src_emb,
            src_key_padding_mask=src_key_padding_mask # 传递 src 的 padding mask 给 encoder
        ) # 形状: (batch_size, src_seq_len, d_model)

        # 3. 解码目标序列 (逐步生成)
        # 初始化目标序列为只包含 <sos> token
        # trg_tokens 形状: (batch_size, current_trg_seq_len)
        trg_tokens = torch.full((batch_size, 1), self.sos_id, dtype=torch.long, device=src.device)

        # 存储每个样本的完成状态 (是否已生成 <eos>)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)

        # 存储每个样本的翻译结果 (token IDs)
        translated_tokens = [[] for _ in range(batch_size)]

        # 逐步生成直到达到最大长度或所有样本都生成了 <eos>
        for step in range(max_length - 1): # 减去 SOS 的一个位置

            # 检查是否所有样本都已完成
            if finished.all():
                break

            # 获取当前目标序列的 embedding 和位置编码
            # trg_emb 形状: (batch_size, current_trg_seq_len, d_model)
            # 添加位置编码和缩放
            trg_emb = self.positional_encoding(self.embedding(trg_tokens) * math.sqrt(self.d_model))

            # 生成当前目标序列的因果掩码
            # current_trg_seq_len = step + 1 (SOS + step 个生成的 token)
            current_trg_seq_len = trg_tokens.size(1)
            # tgt_mask 形状: (current_trg_seq_len, current_trg_seq_len)
            tgt_mask = self.transformer.generate_square_subsequent_mask(current_trg_seq_len).to(src.device)

            # 解码器前向传播一步
            # tgt_key_padding_mask 不需要，因为我们是逐步构建，不会有 padding (除了初始的 SOS)
            # 但为了严谨，如果需要，可以构建一个全 False 的掩码，或者忽略这个参数 (默认 None)
            # memory_key_padding_mask 需要，用于掩盖 encoder output 中的 padding
            decoder_output = self.transformer.decoder(
                trg_emb,
                encoder_output, # 编码器输出
                tgt_mask=tgt_mask, # 解码器自注意力因果掩码
                memory_key_padding_mask=memory_key_padding_mask # 编码-解码注意力 padding 掩码
            ) # 形状: (batch_size, current_trg_seq_len, d_model)

            # 获取最后一个时间步的输出，用于预测下一个 token
            # last_step_output 形状: (batch_size, d_model)
            last_step_output = decoder_output[:, -1, :]

            # 通过 generator 层获取 logits
            # prediction_logits 形状: (batch_size, vocabulary_size)
            prediction_logits = self.generator(last_step_output)

            # 贪婪地选择概率最高的 token 作为下一个 token
            # predicted_tokens 形状: (batch_size,)
            predicted_tokens = torch.argmax(prediction_logits, dim=-1)

            # 将预测的 token 添加到当前目标序列 (trg_tokens)
            # 在维度 1 上拼接 predicted_tokens (需要将其 unsqueeze(1) 使其形状变为 (batch_size, 1))
            trg_tokens = torch.cat([trg_tokens, predicted_tokens.unsqueeze(1)], dim=1) # 形状: (batch_size, current_trg_seq_len + 1)

            # 更新翻译结果列表和完成状态
            for i in range(batch_size):
                # 如果该样本还未完成
                if not finished[i]:
                    # 获取当前样本预测的 token ID
                    token_id = predicted_tokens[i].item()
                    # 将 token ID 添加到该样本的翻译结果列表
                    translated_tokens[i].append(token_id)
                    # 如果预测的 token 是 EOS，则标记该样本为已完成
                    if token_id == self.eos_id:
                        finished[i] = True

        # 返回翻译结果列表 (token ID 列表的列表)
        return translated_tokens


# 训练函数
def train(model, data_loader, optimizer, criterion, epoch, writer):
    """
    训练模型一个 epoch。
    Args:
        model: 待训练的模型。
        data_loader: 训练数据的数据加载器。
        optimizer: 优化器。
        criterion: 损失函数。
        epoch: 当前 epoch 编号。
        writer: TensorBoard SummaryWriter。
    Returns:
        当前 epoch 的平均训练损失。
    """
    model.train() # 设置模型为训练模式
    epoch_loss = 0
    all_correct = 0 # 用于计算训练集准确率 (忽略 padding)
    all_total = 0   # 用于计算训练集准确率 (忽略 padding)

    # tqdm 进度条
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1} [Train]")

    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad() # 清除之前的梯度

        src = batch['src'] # 源序列，形状 (batch_size, src_seq_len)
        tgt = batch['tgt'] # 目标序列 (带有 <sos> 和 <eos>)，形状 (batch_size, tgt_seq_len)

        with torch.autocast(device_type=config.device.type, dtype=torch.float32, enabled=True):
            # 前向传播
            # model 的 forward 函数接收的是 decoder 的输入序列 (带有 <sos>)
            # output 形状: (batch_size, tgt_seq_len, vocabulary_size)
            output = model(src, tgt)

            # 计算损失 (忽略<pad>标记)
            # 模型的输出 logits 是预测下一个 token
            # output[i, j, :] 是预测 trg[i, j+1] 的概率分布
            # 所以，我们需要将输出 tensor 和目标 tensor 向后错开一位
            # 输出 logits 需要去掉最后一个时间步的预测 (它没有对应的真实标签)
            # 目标标签需要去掉第一个时间步的 <sos> token
            # 形状调整为 (batch_size * (tgt_seq_len - 1), vocabulary_size) 和 (batch_size * (tgt_seq_len - 1))

            output_dim = output.shape[-1] # 词汇表大小

            # 移除输出 logits 的最后一个时间步，并展平
            # 形状从 (batch_size, tgt_seq_len, output_dim) 变为 (batch_size * (tgt_seq_len - 1), output_dim)
            output = output[:, :-1, :].reshape(-1, output_dim)

            # 移除目标序列的第一个时间步 (<sos>)，并展平
            # 形状从 (batch_size, tgt_seq_len) 变为 (batch_size * (tgt_seq_len - 1))
            # 这个展平后的 tgt 包含了真实的 token 标签以及后面的 <eos> 和 <pad>
            tgt = tgt[:, 1:].reshape(-1)

            # 计算交叉熵损失
            # criterion (CrossEntropyLoss) 会自动处理忽略 ignore_index 的位置
            loss = criterion(output, tgt)

        # 反向传播
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 更新参数
        optimizer.step()

        # 累加损失和更新进度条
        epoch_loss += loss.item()
        # 计算准确率 (忽略 padding)
        with torch.no_grad(): # 确保准确率计算不产生梯度
            pred = output.argmax(1) # 获取预测的 token ID
            non_pad_mask = (tgt != config.pad_id) # 创建非 padding 位置的掩码
            correct = (pred == tgt) & non_pad_mask # 找到预测正确且不是 padding 的位置
            all_correct += correct.sum().item() # 累加正确预测数
            all_total += non_pad_mask.sum().item() # 累加非 padding 位置总数

        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct.sum().item() / (non_pad_mask.sum().item() + 1e-6):.4f}") # 计算并显示当前批次的准确率 (避免除以0)


        # 每100个batch记录到tensorboard
        if batch_idx % 100 == 0:
            global_step = epoch * len(data_loader) + batch_idx
            writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)
            # 计算并记录批量准确率
            batch_accuracy = correct.sum().item() / (non_pad_mask.sum().item() + 1e-6) # 避免除以零
            writer.add_scalar('Train/Batch_Accuracy', batch_accuracy, global_step)


    # 计算并返回整个 epoch 的平均损失
    average_epoch_loss = epoch_loss / len(data_loader)
    # 计算整个 epoch 的平均准确率
    average_epoch_accuracy = all_correct / (all_total + 1e-6) # 避免除以零

    writer.add_scalar('Train/Epoch_Loss', average_epoch_loss, epoch)
    writer.add_scalar('Train/Epoch_Accuracy', average_epoch_accuracy, epoch)

    return average_epoch_loss

# 评估函数
def evaluate(model, data_loader, criterion, epoch, writer):
    """
    评估模型一个 epoch。
    Args:
        model: 待评估的模型。
        data_loader: 验证数据的数据加载器。
        criterion: 损失函数。
        epoch: 当前 epoch 编号。
        writer: TensorBoard SummaryWriter。
    Returns:
        当前 epoch 的平均验证损失和准确率。
    """
    model.eval() # 设置模型为评估模式
    epoch_loss = 0
    all_correct = 0 # 用于计算准确率 (忽略 padding)
    all_total = 0   # 用于计算准确率 (忽略 padding)

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1} [Eval]")

    with torch.no_grad(): # 在评估时不计算梯度
        for batch_idx, batch in enumerate(progress_bar):
            src = batch['src']
            tgt = batch['tgt']

            # 前向传播
            output = model(src, tgt) # 形状: (batch_size, tgt_seq_len, vocabulary_size)

            # 计算损失
            output_dim = output.shape[-1]

            # 移除输出 logits 的最后一个时间步，并展平
            output = output[:, :-1, :].reshape(-1, output_dim) # 形状: (batch_size * (tgt_seq_len - 1), output_dim)

            # 移除目标序列的第一个时间步 (<sos>)，并展平
            tgt = tgt[:, 1:].reshape(-1) # 形状: (batch_size * (tgt_seq_len - 1))

            loss = criterion(output, tgt)
            epoch_loss += loss.item()

            # 计算准确率
            pred = output.argmax(1) # 获取预测的 token ID
            non_pad_mask = (tgt != config.pad_id) # 创建非 padding 位置的掩码
            correct = (pred == tgt) & non_pad_mask # 找到预测正确且不是 padding 的位置
            all_correct += correct.sum().item() # 累加正确预测数
            all_total += non_pad_mask.sum().item() # 累加非 padding 位置总数

            # 更新进度条
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct.sum().item() / (non_pad_mask.sum().item() + 1e-6):.4f}") # 计算并显示当前批次的准确率


    # 计算整个 epoch 的平均损失和准确率
    average_epoch_loss = epoch_loss / len(data_loader)
    average_epoch_accuracy = all_correct / (all_total + 1e-6) # 避免除以零

    # 记录到tensorboard
    writer.add_scalar('Eval/Epoch_Loss', average_epoch_loss, epoch)
    writer.add_scalar('Eval/Epoch_Accuracy', average_epoch_accuracy, epoch)

    return average_epoch_loss, average_epoch_accuracy

# 翻译示例函数 (使用模型的 translate 方法)
def translate_examples(model, dataset, examples):
    """
    对示例句子进行翻译。
    Args:
        model: 待翻译的模型实例。
        dataset: 用于获取 tokenizer 和 ID 到 token 映射的数据集实例 (或包含这些信息的对象)。
        examples: 需要翻译的英文句子列表。
    """
    # 将模型切换到评估模式 (在 translate 方法内部已经做了)
    # model.eval()

    for example in examples:
        # 1. 预处理输入句子
        # 使用 tokenizer 编码英文文本
        src_encoding = dataset.tokenizer.encode(example, add_special_tokens=False)
        src_ids = src_encoding.ids

        # 截断以留出空间给 sos 和 eos
        src_ids = src_ids[:config.max_length - 2]

        # 添加开始和结束标记
        src_indices = [dataset.sos_id] + src_ids + [dataset.eos_id]

        # Pad sequences to max_length (如果需要)
        # 对于单个句子，通常不需要 pad，但为了使用批处理模型，这里还是 pad 到 max_length
        # 或者更常见的是，为推理单独写一个函数处理批量的不同长度句子
        # 这里简化处理，pad 到 config.max_length
        src_indices += [dataset.pad_id] * (config.max_length - len(src_indices))
        src_indices = src_indices[:config.max_length] # 确保长度不超过 max_length

        # 转换为 tensor 并移到设备
        src_tensor = torch.tensor([src_indices], dtype=torch.long, device=config.device) # 加一个 batch 维度

        # 2. 调用模型的 translate 方法进行翻译
        # translate 方法返回一个列表，每个元素是一个 token ID 列表
        predicted_ids_list = model.translate(src_tensor) # translate 方法内部处理 greedy search

        # 由于输入只有一个样本 (batch_size=1)，我们取第一个结果
        predicted_ids = predicted_ids_list[0]

        # 3. 后处理：将 token ID 转换回文本
        # 移除 <sos> 和 <eos> 以及 padding (如果存在)
        # 从第一个 token 开始 (跳过可能的 <sos>)
        # 停止于第一个 <eos> 或序列结束
        translation_tokens = []
        for token_id in predicted_ids:
            if token_id == dataset.eos_id or token_id == dataset.pad_id:
                break # 遇到 EOS 或 PAD 停止
            if token_id != dataset.sos_id: # 跳过可能的 SOS (虽然 translate 方法返回的应该不包含 SOS)
                translation_tokens.append(dataset.idx_to_token[token_id])

        # 将 token 列表拼接成字符串
        # 使用 ''.join() 对于中文通常比 ' '.join() 更合适
        translation_text = ''.join(translation_tokens)

        print(f"Input: {example}")
        print(f"Translation: {translation_text}")
        print("-" * 70)

# 保存检查点
def save_checkpoint(model, optimizer, epoch, loss, accuracy, save_path):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'src_vocab_size': config.src_vocab_size, # 保存词汇表大小和特殊标记ID
        'tgt_vocab_size': config.tgt_vocab_size,
        'pad_id': config.pad_id,
        'sos_id': config.sos_id,
        'eos_id': config.eos_id,
        'unk_id': config.unk_id,
        'config': {k: getattr(config, k) for k in dir(config) if not k.startswith('_') and not callable(getattr(config, k))}, # 保存 config 参数
    }
    # 构建检查点文件名
    checkpoint_filename = f'checkpoint_epoch_{epoch+1}.pt'
    checkpoint_path = os.path.join(save_path, checkpoint_filename)
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_filename}")

# 加载检查点
def load_latest_checkpoint(model, optimizer, save_path):
    """加载最新的模型检查点"""
    checkpoints = [f for f in os.listdir(save_path) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    if not checkpoints:
        print("No checkpoint found.")
        return None, 0

    # 找到最新的检查点 (按 epoch 编号排序)
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_checkpoint_filename = checkpoints[-1]
    checkpoint_path = os.path.join(save_path, latest_checkpoint_filename)

    print(f"Loading checkpoint: {checkpoint_path}")
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=config.device)

    # 加载模型状态字典
    model_state_dict = checkpoint['model_state_dict']

    # 判断是否为torch.compile后保存的模型，进行兼容处理
    new_state_dict = {}
    for k, v in model_state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[len('_orig_mod.'):]] = v
        else:
            new_state_dict[k] = v

    model.to(config.device)
    model.load_state_dict(new_state_dict)

    # 加载优化器状态字典 (如果存在且需要恢复优化器状态)
    if 'optimizer_state_dict' in checkpoint and optimizer is not None:
         try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
         except ValueError as e:
            print(f"Warning: Could not load optimizer state_dict. It might not match the current optimizer. Error: {e}")

    # 恢复 epoch
    starting_epoch = checkpoint.get('epoch', 0) + 1 # 从下一个 epoch 开始训练

    # 恢复其他信息 (可选)
    # config 参数也可以从 checkpoint 中加载，以确保模型结构和训练参数一致
    # 例如： if 'config' in checkpoint: load_config_from_dict(checkpoint['config'])

    print(f"Checkpoint loaded. Resuming from epoch {starting_epoch}")
    return checkpoint, starting_epoch

# 主训练流程
def main():
    print(f"Using device: {config.device}")

    # 加载数据集
    try:
        # 请确保这些文件路径正确，并且 tokenizer.json 存在
        train_dataset = TranslationDataset("/home/dataset/one_third_translation2019zh_train.json")
        valid_dataset = TranslationDataset("/home/dataset/translation2019zh_valid.json")
    except FileNotFoundError as e:
        print(f"Error loading data or tokenizer: {e}")
        print("Please check the file paths for data files and ensure 'tokenizer.json' is in the same directory.")
        return

    # 创建 Collate 函数实例
    collate_fn = CollateFn(config.pad_id) # 使用 config 中的 pad_id

    # 创建数据加载器
    # num_workers > 0 会使用多进程加载数据，可以加快速度，但在 Windows 上可能需要特殊处理或设置为 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0, # 简化调试，生产环境可调整
        pin_memory=False if config.device.type == 'cuda' else False, # CUDA 可以开启 pin_memory 加快数据传输
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False, # 评估时通常不 shuffle
        collate_fn=collate_fn,
        num_workers=0, # 简化调试
        pin_memory=False if config.device.type == 'cuda' else False,
    )

    # 定义模型
    model = TranslatModel(
        vocabulary_size=config.src_vocab_size, # 使用 config 中由 dataset 设置的词汇表大小
        d_model=config.embedding_dim,
        n_head=config.num_heads,
        n_layers=config.num_layers,
        d_ff=config.d_feedforward,
        drop_out=config.dropout,
        max_len=config.max_length, # PositionEncoding 的 max_len 应该 >= 实际使用的 max_length
        pad_idx=config.pad_id,     # 将特殊标记 ID 传递给模型
        sos_id=config.sos_id,
        eos_id=config.eos_id,
    )

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # CrossEntropyLoss 期望输入是 (N, C) 和 (N) 或 (N, C, d1, d2...) 和 (N, d1, d2...)
    # 我们将 logits 和 targets 展平后是 (batch_size * seq_len, vocab_size) 和 (batch_size * seq_len)
    # ignore_index=config.pad_id 确保计算损失时忽略 padding 位置
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_id)

    # 创建TensorBoard写入器
    writer = SummaryWriter(config.tensorboard_log_dir)

    # 检查是否有保存的检查点并加载
    starting_epoch = 0
    # 只有当保存目录存在且不为空时尝试加载
    if os.path.exists(config.save_path) and any(f.startswith('checkpoint_epoch_') and f.endswith('.pt') for f in os.listdir(config.save_path)):
        checkpoint, starting_epoch = load_latest_checkpoint(model, optimizer, config.save_path)
    else:
         print("No existing checkpoints found. Starting training from epoch 1.")
         model = model.to(config.device)


    # 示例英文句子用于展示翻译效果
    examples = [
        "Hello, how are you.",
        "This is a test sentence.",
        "I love learning new languages.",
        "The weather is nice today.",
        "Artificial intelligence is transforming the world.",
        "Machine learning algorithms are becoming more sophisticated.",
    ]

    # 训练循环
    best_valid_loss = float('inf') # 记录最佳验证损失以保存最佳模型
    model = torch.compile(model)

    for epoch in range(starting_epoch, config.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{config.num_epochs} ---")

        # 训练
        train_loss = train(model, train_loader, optimizer, criterion, epoch, writer)
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}")

        # 评估
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, epoch, writer) # 评估函数不再需要 dataset 对象，只需要 config 中的 pad_id
        print(f"Epoch {epoch+1} Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")

        # 展示翻译示例 (需要传递 dataset 对象给 translate_examples 以获取 tokenizer 和 ID 到 token 映射)
        print("\nTranslation examples:")
        # 注意：这里使用了训练集的 dataset 对象，因为它包含了 tokenizer 和 idx_to_token 映射
        translate_examples(model, train_dataset, examples) # 确保 dataset 对象被传递

        # 保存检查点
        save_checkpoint(model, optimizer, epoch, valid_loss, valid_acc, config.save_path)

        # 更新最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # 保存最佳模型的 state_dict
            torch.save(model.state_dict(), os.path.join(config.save_path, 'best_model.pt'))
            print("Validation loss improved. Best model state_dict saved!")

    writer.close() # 关闭 TensorBoard 写入器
    print("\nTraining complete!")

# 运行主函数
if __name__ == "__main__":
    main()