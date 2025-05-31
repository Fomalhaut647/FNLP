#!/usr/bin/env python3
"""
分析HoC数据集在原始BERT和扩展BERT分词器下的分词效果对比
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import os
import random

def load_hoc_data():
    """加载HoC数据集"""
    print("正在加载HoC数据集...")
    
    train_path = "data/HoC/train.parquet"
    test_path = "data/HoC/test.parquet"
    
    if not os.path.exists(train_path):
        print(f"错误: 找不到训练数据文件 {train_path}")
        return None, None
    
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path) if os.path.exists(test_path) else None
    
    print(f"训练集大小: {len(train_df)}")
    if test_df is not None:
        print(f"测试集大小: {len(test_df)}")
    
    print(f"数据列: {train_df.columns.tolist()}")
    print(f"前几行数据:")
    print(train_df.head())
    
    return train_df, test_df

def analyze_tokenization_comparison(texts, original_tokenizer, extended_tokenizer, sample_size=1000):
    """
    比较原始BERT和扩展BERT分词器的效果
    
    Args:
        texts: 文本列表
        original_tokenizer: 原始BERT分词器
        extended_tokenizer: 扩展BERT分词器
        sample_size: 采样大小
    """
    print(f"\n正在分析 {min(len(texts), sample_size)} 个文本的分词效果...")
    
    # 随机采样
    if len(texts) > sample_size:
        sample_texts = random.sample(texts, sample_size)
    else:
        sample_texts = texts
    
    original_lengths = []
    extended_lengths = []
    improvements = []
    
    for text in sample_texts:
        # 原始BERT分词
        original_tokens = original_tokenizer.tokenize(str(text))
        original_length = len(original_tokens)
        
        # 扩展BERT分词
        extended_tokens = extended_tokenizer.tokenize(str(text))
        extended_length = len(extended_tokens)
        
        original_lengths.append(original_length)
        extended_lengths.append(extended_length)
        
        # 计算改进
        improvement = original_length - extended_length
        improvements.append(improvement)
    
    # 统计分析
    original_avg = np.mean(original_lengths)
    extended_avg = np.mean(extended_lengths)
    improvement_avg = np.mean(improvements)
    improvement_rate = (improvement_avg / original_avg) * 100 if original_avg > 0 else 0
    
    print(f"\n=== 分词长度统计 ===")
    print(f"原始BERT平均token数: {original_avg:.2f}")
    print(f"扩展BERT平均token数: {extended_avg:.2f}")
    print(f"平均减少token数: {improvement_avg:.2f}")
    print(f"Token减少率: {improvement_rate:.2f}%")
    
    # 改进分布
    positive_improvements = [imp for imp in improvements if imp > 0]
    negative_improvements = [imp for imp in improvements if imp < 0]
    no_change = [imp for imp in improvements if imp == 0]
    
    print(f"\n=== 改进分布 ===")
    print(f"有改进的文本: {len(positive_improvements)} ({len(positive_improvements)/len(improvements)*100:.1f}%)")
    print(f"无变化的文本: {len(no_change)} ({len(no_change)/len(improvements)*100:.1f}%)")
    print(f"变差的文本: {len(negative_improvements)} ({len(negative_improvements)/len(improvements)*100:.1f}%)")
    
    if positive_improvements:
        print(f"平均改进幅度: {np.mean(positive_improvements):.2f} tokens")
        print(f"最大改进幅度: {max(positive_improvements)} tokens")
    
    return {
        'original_avg': original_avg,
        'extended_avg': extended_avg,
        'improvement_avg': improvement_avg,
        'improvement_rate': improvement_rate,
        'positive_improvements': len(positive_improvements),
        'no_change': len(no_change),
        'negative_improvements': len(negative_improvements)
    }

def show_sample_comparisons(texts, original_tokenizer, extended_tokenizer, num_samples=3):
    """显示具体的分词对比示例"""
    print(f"\n=== 分词对比示例 ===")
    
    # 随机选择几个样本
    sample_texts = random.sample(texts, min(num_samples, len(texts)))
    
    for i, text in enumerate(sample_texts, 1):
        text_str = str(text)
        if len(text_str) > 200:
            text_str = text_str[:200] + "..."
        
        print(f"\n【示例 {i}】")
        print(f"原文: {text_str}")
        
        # 原始BERT分词
        original_tokens = original_tokenizer.tokenize(text)
        print(f"原始BERT ({len(original_tokens)} tokens): {original_tokens}")
        
        # 扩展BERT分词
        extended_tokens = extended_tokenizer.tokenize(text)
        print(f"扩展BERT ({len(extended_tokens)} tokens): {extended_tokens}")
        
        # 改进情况
        improvement = len(original_tokens) - len(extended_tokens)
        if improvement > 0:
            print(f"✅ 改进: 减少了 {improvement} 个token")
        elif improvement == 0:
            print(f"➖ 无变化")
        else:
            print(f"❌ 增加了 {-improvement} 个token")

def main():
    """主函数"""
    print("="*80)
    print("HoC数据集分词效果分析")
    print("="*80)
    
    # 设置随机种子
    random.seed(42)
    
    # 加载数据
    train_df, test_df = load_hoc_data()
    if train_df is None:
        return
    
    # 加载分词器
    print("\n正在加载分词器...")
    try:
        original_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        extended_tokenizer = AutoTokenizer.from_pretrained('extended_bert_output')
        print("✅ 分词器加载成功")
    except Exception as e:
        print(f"❌ 分词器加载失败: {e}")
        return
    
    # 获取文本数据
    # 假设文本在'text'列中，如果不是需要调整
    text_column = None
    for col in ['text', 'sentence', 'content', 'abstract', 'title']:
        if col in train_df.columns:
            text_column = col
            break
    
    if text_column is None:
        print("错误: 找不到文本列")
        print(f"可用列: {train_df.columns.tolist()}")
        return
    
    texts = train_df[text_column].dropna().tolist()
    print(f"找到 {len(texts)} 个有效文本")
    
    # 分析分词效果
    stats = analyze_tokenization_comparison(texts, original_tokenizer, extended_tokenizer)
    
    # 显示示例
    show_sample_comparisons(texts, original_tokenizer, extended_tokenizer)
    
    print(f"\n=== 总结 ===")
    print(f"✅ 扩展BERT分词器在HoC数据集上的表现:")
    print(f"   • 平均token数从 {stats['original_avg']:.2f} 减少到 {stats['extended_avg']:.2f}")
    print(f"   • Token减少率: {stats['improvement_rate']:.2f}%")
    print(f"   • {stats['positive_improvements']} 个文本有改进")
    print(f"   • {stats['no_change']} 个文本无变化")
    print(f"   • {stats['negative_improvements']} 个文本变差")

if __name__ == "__main__":
    main() 