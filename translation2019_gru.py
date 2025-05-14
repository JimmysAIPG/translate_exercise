import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import nltk
import numpy as np
from tqdm import tqdm
import random
from tokenizers import Tokenizer
import time

# 确保下载nltk的tokenizer
nltk.download('punkt')

# 设置随机种子，确保结果可复现
SEED = 37
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 配置参数
class Config:
    batch_size = 64
    hidden_size = 256
    embedding_dim = 384
    num_layers = 3
    learning_rate = 0.001
    num_epochs = 20
    max_length = 70
    teacher_forcing_ratio = 0.5
    save_path = "./model_checkpoints"
    tensorboard_log_dir = "./logs"
    # 词汇表大小，会在数据加载后更新
    src_vocab_size = 0
    tgt_vocab_size = 0

    def __post_init__(self):  # 使用 __post_init__ 确保在所有参数初始化后设置 device
        # 设置设备
        # 支持 CUDA、Intel XPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:  # 需要 Intel PyTorch 扩展
            try:
                import intel_extension_for_pytorch as ipex
                self.device = torch.device('xpu')
                self.ipex = ipex  # Store ipex module if available
            except ImportError:
                self.device = torch.device('cpu')  # Fallback to CPU if ipex is not available
        print(f"使用设备: {self.device}")


config = Config()
config.__post_init__()

# 检查并创建保存目录
os.makedirs(config.save_path, exist_ok=True)
os.makedirs(config.tensorboard_log_dir, exist_ok=True)

class TranslationDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        self.max_length = config.max_length
        # 读取数据
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in tqdm(lines, desc="Loading data"):
            try:
                item = json.loads(line.strip())
                # 不对英文进行小写处理，与tokenizer训练时保持一致
                self.data.append((item['english'], item['chinese']))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not parse line: {line}. Error: {e}")
        # 加载预训练的BPE tokenizer
        self.tokenizer = Tokenizer.from_file("tokenizer.json")
        # 获取特殊标记的ID
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.sos_id = self.tokenizer.token_to_id("[CLS]")  # 使用[CLS]作为<sos>
        self.eos_id = self.tokenizer.token_to_id("[SEP]")  # 使用[SEP]作为<eos>
        self.unk_id = self.tokenizer.token_to_id("[UNK]")

        # 设置词汇表大小（源语言和目标语言共享）
        vocab_size = self.tokenizer.get_vocab_size()
        config.src_vocab_size = vocab_size
        config.tgt_vocab_size = vocab_size

        # 创建索引到token的映射 (重要，后续翻译时会用到)
        self.idx_to_token = {idx: token for token, idx in self.tokenizer.get_vocab().items()}

        print(f"Vocabulary size: {vocab_size}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        # 使用tokenizer编码文本
        src_encoding = self.tokenizer.encode(src)  # 英文
        tgt_encoding = self.tokenizer.encode(tgt)  # 中文
        src_ids = src_encoding.ids
        tgt_ids = tgt_encoding.ids
        # 截断以留出空间给sos和eos
        src_ids = src_ids[:self.max_length - 2]
        tgt_ids = tgt_ids[:self.max_length - 2]
        # 添加开始和结束标记
        src_indices = [self.sos_id] + src_ids + [self.eos_id]
        tgt_indices = [self.sos_id] + tgt_ids + [self.eos_id]
        return {
            'src': torch.tensor(src_indices, dtype=torch.long),
            'tgt': torch.tensor(tgt_indices, dtype=torch.long),
            'src_len': len(src_indices),
            'tgt_len': len(tgt_indices)
        }

class CollateFn:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        max_src_len = max(item['src_len'] for item in batch)
        max_tgt_len = max(item['tgt_len'] for item in batch)
        src_batch = torch.full((len(batch), max_src_len), self.pad_id, dtype=torch.long)
        tgt_batch = torch.full((len(batch), max_tgt_len), self.pad_id, dtype=torch.long)
        src_mask = torch.zeros(len(batch), max_src_len, dtype=torch.bool)
        tgt_mask = torch.zeros(len(batch), max_tgt_len, dtype=torch.bool)
        for i, item in enumerate(batch):
            src = item['src']
            tgt = item['tgt']
            src_len = item['src_len']
            tgt_len = item['tgt_len']
            src_batch[i, :src_len] = src
            tgt_batch[i, :tgt_len] = tgt
            src_mask[i, :src_len] = 1
            tgt_mask[i, :tgt_len] = 1
        return {
            'src': src_batch.to(config.device),
            'tgt': tgt_batch.to(config.device),
            'src_mask': src_mask.to(config.device),
            'tgt_mask': tgt_mask.to(config.device),
            'src_lengths': torch.tensor([item['src_len'] for item in batch], dtype=torch.long).to(config.device),
            'tgt_lengths': torch.tensor([item['tgt_len'] for item in batch], dtype=torch.long).to(config.device)
        }

# 编码器定义
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))
        # embedded: [batch_size, src_len, embedding_dim]
        # 对序列进行打包，以便GRU能够忽略填充部分
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        # 通过GRU
        outputs, hidden = self.gru(packed)
        # 解包序列
        outputs, _= nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # outputs: [batch_size, src_len, hidden_size*2]
        # 将前向和后向的隐藏状态连接起来
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        # hidden: [batch_size, hidden_size*2]
        # 将隐藏状态映射回原来的维度
        hidden = torch.tanh(self.fc(hidden))
        # hidden: [batch_size, hidden_size]
        # 调整hidden的形状以适应解码器
        hidden = hidden.unsqueeze(0).repeat(config.num_layers, 1, 1)
        # hidden: [num_layers, batch_size, hidden_size]
        return outputs, hidden

# 解码器定义
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        # input: [batch_size, 1]
        # hidden: [num_layers, batch_size, hidden_size]
        embedded = self.dropout(self.embedding(input))
        # embedded: [batch_size, 1, embedding_dim]
        output, hidden = self.gru(embedded, hidden)
        # output: [batch_size, 1, hidden_size]
        # hidden: [num_layers, batch_size, hidden_size]
        prediction = self.fc_out(output.squeeze(1))
        # prediction: [batch_size, vocab_size]
        return prediction, hidden

# Seq2Seq 模型
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, src_lengths, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.fc_out.out_features
        # 用于存储每个时间步的预测
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(src.device)
        # 编码
        _, hidden = self.encoder(src, src_lengths)
        # 第一个输入是<sos>标记
        input = tgt[:, 0:1]
        for t in range(1, tgt_len):
            # 解码
            output, hidden = self.decoder(input, hidden)
            # output: [batch_size, vocab_size]
            # 存储当前时间步的预测
            outputs[:, t, :] = output
            # 决定是否使用teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            # 获取最大概率的词
            top1 = output.argmax(1)
            # 准备下一个时间步的输入
            input = tgt[:, t:t+1] if teacher_force else top1.unsqueeze(1)
        return outputs

    def translate(self, src, src_lengths, dataset, max_length=70):  # 修改了参数列表
        # 设置为评估模式
        self.eval()
        with torch.no_grad():
            # 编码
            _, hidden = self.encoder(src, src_lengths)
            # 第一个输入是<sos>标记
            input = torch.tensor([[dataset.sos_id]]).to(src.device)  # 使用dataset中的sos_id
            translations = []
            for _ in range(max_length):
                # 解码
                output, hidden = self.decoder(input, hidden)
                # 获取最大概率的词
                top1 = output.argmax(1)
                # 如果生成了<eos>，则停止
                if top1.item() == dataset.eos_id:  # 使用dataset中的eos_id
                    break
                # 添加到翻译结果
                translations.append(dataset.idx_to_token[top1.item()])  # 使用dataset中的映射
                # 准备下一个时间步的输入
                input = top1.unsqueeze(1)
            return translations

# 训练函数
def train(model, data_loader, optimizer, criterion, epoch, writer):
    model.train()
    epoch_loss = 0
    batch_count = 0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1} [Train]")
    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        src = batch['src']
        tgt = batch['tgt']
        src_lengths = batch['src_lengths']
        # 前向传播
        output = model(src, tgt, src_lengths, config.teacher_forcing_ratio)
        # 计算损失 (忽略<pad>标记)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)  # 不计算<sos>的损失
        tgt = tgt[:, 1:].reshape(-1)  # 不计算<sos>的目标
        loss = criterion(output, tgt)
        # 反向传播
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # 更新参数
        optimizer.step()
        epoch_loss += loss.item()
        batch_count += 1
        # 更新进度条
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        # 每100个batch记录到tensorboard
        if batch_idx % 100 == 0:
            global_step = epoch * len(data_loader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            # 计算准确率
            pred = output.argmax(1)
            correct = (pred == tgt).float()
            acc = correct.sum() / len(correct)
            writer.add_scalar('Train/Accuracy', acc.item(), global_step)
    return epoch_loss / batch_count

# 评估函数
def evaluate(model, data_loader, criterion, dataset, epoch, writer):
    model.eval()
    epoch_loss = 0
    batch_count = 0
    all_correct = 0
    all_total = 0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1} [Eval]")
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            src = batch['src']
            tgt = batch['tgt']
            src_lengths = batch['src_lengths']
            # 前向传播
            output = model(src, tgt, src_lengths, 0)  # 不使用teacher forcing
            # 计算损失
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
            batch_count += 1
            # 计算准确率
            pred = output.argmax(1)
            non_pad = tgt != dataset.pad_id
            correct = (pred == tgt) & non_pad
            all_correct += correct.sum().item()
            all_total += non_pad.sum().item()
            # 更新进度条
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    # 记录到tensorboard
    writer.add_scalar('Eval/Loss', epoch_loss / batch_count, epoch)
    accuracy = all_correct / all_total if all_total > 0 else 0
    writer.add_scalar('Eval/Accuracy', accuracy, epoch)
    return epoch_loss / batch_count, accuracy

# 翻译示例函数
def translate_examples(model, dataset, examples):  # 修改了参数列表
    model.eval()
    for example in examples:
        # 使用tokenizer编码文本
        src_encoding = dataset.tokenizer.encode(example)  # 英文
        src_ids = src_encoding.ids

        # 截断以留出空间给sos和eos
        src_ids = src_ids[:dataset.max_length - 2]

        # 添加开始和结束标记
        src_indices = [dataset.sos_id] + src_ids + [dataset.eos_id]

        # 转换为张量
        src = torch.tensor([src_indices]).to(config.device)
        src_lengths = torch.tensor([len(src_indices)]).to(config.device)
        # 翻译
        translations = model.translate(
            src,
            src_lengths,
            dataset,  # 传递dataset对象
            max_length=70
        )
        print(f"Input: {example}")
        print(f"Translation: {''.join(translations)}")
        print("-" * 70)

# 保存检查点
def save_checkpoint(model, optimizer, epoch, loss, accuracy, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'src_vocab_size': config.src_vocab_size,
        'tgt_vocab_size': config.tgt_vocab_size,
    }
    torch.save(checkpoint, os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pt'))
    print(f"Checkpoint saved: epoch_{epoch+1}")

# 加载检查点
def load_latest_checkpoint(model, optimizer, save_path):
    checkpoints = [f for f in os.listdir(save_path) if f.startswith('checkpoint_epoch_')]
    if not checkpoints:
        return None, 0
    # 找到最新的检查点
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    checkpoint_path = os.path.join(save_path, latest_checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint, checkpoint['epoch']

def main():
    print(f"Using device: {config.device}")
    # 加载数据集
    try:
        train_dataset = TranslationDataset("/home/dataset/one_third_translation2019zh_train.json")
        valid_dataset = TranslationDataset("/home/dataset/translation2019zh_valid.json")
    except FileNotFoundError:
        print("Error: Training or validation data file not found.")
        return
    collate_fn = CollateFn(train_dataset.pad_id)
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    # 定义模型
    encoder = Encoder(
        config.src_vocab_size,
        config.embedding_dim,
        config.hidden_size,
        config.num_layers
    )
    decoder = Decoder(
        config.tgt_vocab_size,
        config.embedding_dim,
        config.hidden_size,
        config.num_layers
    )
    model = Seq2Seq(encoder, decoder).to(config.device)
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.pad_id)
    # 创建TensorBoard写入器
    writer = SummaryWriter(config.tensorboard_log_dir)
    # 检查是否有保存的检查点
    starting_epoch = 0
    if os.path.exists(config.save_path) and len(os.listdir(config.save_path)) > 0:
        checkpoint, starting_epoch = load_latest_checkpoint(model, optimizer, config.save_path)
    print(f"Resuming training from epoch {starting_epoch+1}")
    # 示例英文句子用于展示翻译效果
    examples = [
        "Hello, how are you.",
        "This is a test sentence.",
        "I love learning new languages.",
        "The weather is nice today."
    ]
    # 训练循环
    best_valid_loss = float('inf')

    if hasattr(config, 'ipex') and (config.device.type == 'xpu'): # 确保ipex存在
        model, optimizer = config.ipex.optimize(model, optimizer=optimizer)

    for epoch in range(starting_epoch, config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        # 训练
        train_loss = train(model, train_loader, optimizer, criterion, epoch, writer)
        print(f"Train Loss: {train_loss:.4f}")
        # 评估
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, train_dataset, epoch, writer)
        print(f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")
        # 展示翻译示例
        print("\nTranslation examples:")
        translate_examples(model, train_dataset, examples)
        # 保存检查点
        save_checkpoint(model, optimizer, epoch, valid_loss, valid_acc, config.save_path)
        # 更新最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(config.save_path, 'best_model.pt'))
        print("Best model saved!")
    writer.close()
    print("Training complete!")

if __name__ == "__main__":
    main()