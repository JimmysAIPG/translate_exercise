# Neural Machine Translation System

A neural machine translation system implemented with GRU and Transformer architectures.

## Project Description 项目简介

This project implements two neural machine translation models:
- GRU-based encoder-decoder (in `translation2019_gru.py`)
- Transformer-based model (in `translation2019_transformer.py`)

Key features include:
- Bidirectional GRU encoder
- Packed sequences for efficient batch processing
- Teacher forcing during training
- Beam search for decoding

本项目实现了两种神经机器翻译模型：
- 基于GRU的编码器-解码器架构（在`translation2019_gru.py`中）
- 基于Transformer的模型（在`translation2019_transformer.py`中）

主要功能包括：
- 双向GRU编码器
- 使用打包序列提高批量处理效率
- 训练时使用教师强制
- 解码时采用束搜索

