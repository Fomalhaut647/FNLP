#!/usr/bin/env python3
"""
WordPiece分词器训练脚本
在PubMed生物医学语料库上训练WordPiece分词器
"""

import json
import os
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import BertNormalizer
from tokenizers.processors import BertProcessing
from tqdm import tqdm


def load_corpus_texts(corpus_file):
    """
    从文本文件中加载文本数据
    
    Args:
        corpus_file (str): 语料库文件路径
        
    Yields:
        str: 文本内容
    """
    print(f"正在加载语料库: {corpus_file}")
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="读取语料库")):
            try:
                line = line.strip()
                if not line:
                    continue
                
                # 尝试解析为JSON，如果失败则当作纯文本处理
                try:
                    data = json.loads(line)
                    # JSON格式处理
                    text = ""
                    if isinstance(data, dict):
                        # 常见的文本字段名
                        text_fields = ['text', 'content', 'body', 'abstract', 'title', 'document']
                        for field in text_fields:
                            if field in data and data[field]:
                                if isinstance(data[field], str):
                                    text += data[field] + " "
                                elif isinstance(data[field], list):
                                    text += " ".join(str(item) for item in data[field]) + " "
                        
                        # 如果没有找到标准字段，尝试组合所有字符串值
                        if not text.strip():
                            for value in data.values():
                                if isinstance(value, str) and value.strip():
                                    text += value + " "
                    
                    elif isinstance(data, str):
                        text = data
                    
                    if text.strip():
                        yield text.strip()
                        
                except json.JSONDecodeError:
                    # 不是JSON格式，当作纯文本处理
                    text = line.strip()
                    if text:
                        yield text
                    
            except Exception as e:
                print(f"警告: 第{line_num+1}行处理失败: {e}")
                continue


def train_wordpiece_tokenizer(corpus_file, vocab_size=30000, output_dir="./tokenizer_output"):
    """
    训练WordPiece分词器
    
    Args:
        corpus_file (str): 语料库文件路径
        vocab_size (int): 词汇表大小
        output_dir (str): 输出目录
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在初始化WordPiece分词器...")
    
    # 初始化分词器
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    
    # 设置标准化器（类似BERT的标准化）
    tokenizer.normalizer = BertNormalizer(clean_text=True, handle_chinese_chars=True, strip_accents=False, lowercase=False)
    
    # 设置预分词器（按空格分词）
    tokenizer.pre_tokenizer = Whitespace()
    
    # 设置训练器
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        limit_alphabet=1000,
        initial_alphabet=[],
        show_progress=True
    )
    
    print("开始训练分词器...")
    
    # 训练分词器
    def text_iterator():
        for text in load_corpus_texts(corpus_file):
            yield text
    
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    
    # 设置后处理器（添加特殊标记）
    tokenizer.post_processor = BertProcessing(
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
    )
    
    # 保存分词器
    tokenizer_path = os.path.join(output_dir, "wordpiece_tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"分词器已保存到: {tokenizer_path}")
    
    # 保存词汇表
    vocab_path = os.path.join(output_dir, "vocab.txt")
    vocab = tokenizer.get_vocab()
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for token, token_id in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{token}\n")
    print(f"词汇表已保存到: {vocab_path}")
    
    return tokenizer


def test_tokenizer(tokenizer, test_texts=None):
    """
    测试分词器
    
    Args:
        tokenizer: 训练好的分词器
        test_texts (list): 测试文本列表
    """
    if test_texts is None:
        test_texts = [
            "The patient was diagnosed with acute myocardial infarction.",
            "COVID-19 is caused by the SARS-CoV-2 virus.",
            "Treatment with antibiotics showed significant improvement.",
            "The molecular structure of DNA consists of nucleotides.",
            "Immunotherapy has shown promising results in cancer treatment."
        ]
    
    print("\n=== 分词器测试 ===")
    for text in test_texts:
        encoded = tokenizer.encode(text)
        tokens = encoded.tokens
        token_ids = encoded.ids
        
        print(f"\n原文: {text}")
        print(f"分词: {tokens}")
        print(f"Token IDs: {token_ids}")
        print(f"重构文本: {tokenizer.decode(token_ids)}")


def main():
    """主函数"""
    corpus_file = "pubmed_sampled_corpus.jsonline"
    vocab_size = 30000
    output_dir = "./output"
    
    print("=== WordPiece分词器训练程序 ===")
    print(f"语料库文件: {corpus_file}")
    print(f"词汇表大小: {vocab_size}")
    print(f"输出目录: {output_dir}")
    
    # 检查语料库文件是否存在
    if not os.path.exists(corpus_file):
        print(f"错误: 找不到语料库文件 {corpus_file}")
        return
    
    try:
        # 训练分词器
        tokenizer = train_wordpiece_tokenizer(corpus_file, vocab_size, output_dir)
        
        # 测试分词器
        test_tokenizer(tokenizer)
        
        print(f"\n=== 训练完成! ===")
        print(f"分词器文件: {output_dir}/wordpiece_tokenizer.json")
        print(f"词汇表文件: {output_dir}/vocab.txt")
        
        # 显示词汇表统计信息
        vocab = tokenizer.get_vocab()
        print(f"词汇表大小: {len(vocab)}")
        print(f"特殊标记: [UNK], [CLS], [SEP], [PAD], [MASK]")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 