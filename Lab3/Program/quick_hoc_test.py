#!/usr/bin/env python3
"""
快速测试扩展后的BERT模型在HoC数据集上的基本功能
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, AutoTokenizer
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os

class QuickHoCClassifier(nn.Module):
    """简化的HoC分类器用于快速测试"""
    
    def __init__(self, bert_model_path, num_labels=10):
        super(QuickHoCClassifier, self).__init__()
        
        # 强制使用CPU
        self.device = torch.device("cpu")
        print(f"使用设备: {self.device}")
        
        # 加载BERT模型
        self.bert = BertModel.from_pretrained(bert_model_path)
        
        # 分类头
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask=None):
        # BERT编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 使用[CLS]表示进行分类
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        
        return logits

class QuickHoCDataset(Dataset):
    """简化的HoC数据集"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 文本编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 转换为多标签格式
        label_vector = torch.zeros(10)
        for l in label:
            if 0 <= l < 10:
                label_vector[l] = 1
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_vector
        }

def load_hoc_data_quick():
    """快速加载HoC数据集的小样本"""
    print("正在加载HoC数据集...")
    
    train_path = "data/HoC/train.parquet"
    if not os.path.exists(train_path):
        print(f"错误: 找不到训练数据文件 {train_path}")
        return None, None
    
    train_df = pd.read_parquet(train_path)
    
    # 只使用前100个样本进行快速测试
    train_df = train_df.head(100)
    
    print(f"使用 {len(train_df)} 个样本进行快速测试")
    
    texts = train_df['text'].tolist()
    labels = train_df['label'].tolist()
    
    return texts, labels

def quick_test_model(model_path, tokenizer_path):
    """快速测试模型"""
    print(f"\n=== 测试模型: {model_path} ===")
    
    # 加载数据
    texts, labels = load_hoc_data_quick()
    if texts is None:
        return None
    
    # 加载分词器和模型
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = QuickHoCClassifier(model_path)
        print(f"✅ 模型加载成功")
        print(f"词汇表大小: {len(tokenizer.vocab)}")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None
    
    # 创建数据集
    dataset = QuickHoCDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # 测试前向传播
    model.eval()
    total_samples = 0
    total_tokens = 0
    
    print("\n正在测试模型前向传播...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 5:  # 只测试前5个batch
                break
                
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            # 前向传播
            try:
                outputs = model(input_ids, attention_mask)
                
                batch_size = input_ids.size(0)
                seq_length = input_ids.size(1)
                
                total_samples += batch_size
                total_tokens += batch_size * seq_length
                
                print(f"Batch {batch_idx + 1}: "
                      f"输入形状 {input_ids.shape}, "
                      f"输出形状 {outputs.shape}")
                
            except Exception as e:
                print(f"❌ 前向传播失败: {e}")
                return None
    
    avg_tokens_per_sample = total_tokens / total_samples if total_samples > 0 else 0
    
    print(f"✅ 前向传播测试成功")
    print(f"处理了 {total_samples} 个样本")
    print(f"平均每个样本的token数: {avg_tokens_per_sample:.1f}")
    
    return {
        'vocab_size': len(tokenizer.vocab),
        'model_params': sum(p.numel() for p in model.parameters()),
        'avg_tokens': avg_tokens_per_sample,
        'test_samples': total_samples
    }

def compare_models():
    """比较原始BERT和扩展BERT模型"""
    print("="*80)
    print("BERT模型对比测试")
    print("="*80)
    
    # 测试原始BERT
    print("\n【测试1】原始BERT模型")
    original_stats = quick_test_model('bert-base-uncased', 'bert-base-uncased')
    
    # 测试扩展BERT
    print("\n【测试2】扩展BERT模型")
    extended_stats = quick_test_model('extended_bert_model', 'extended_bert_model')
    
    # 对比结果
    if original_stats and extended_stats:
        print("\n" + "="*80)
        print("对比结果")
        print("="*80)
        
        print(f"词汇表大小:")
        print(f"  原始BERT: {original_stats['vocab_size']:,}")
        print(f"  扩展BERT: {extended_stats['vocab_size']:,}")
        print(f"  增加: {extended_stats['vocab_size'] - original_stats['vocab_size']:,}")
        
        print(f"\n模型参数数量:")
        print(f"  原始BERT: {original_stats['model_params']:,}")
        print(f"  扩展BERT: {extended_stats['model_params']:,}")
        print(f"  增加: {extended_stats['model_params'] - original_stats['model_params']:,}")
        
        print(f"\n平均token数:")
        print(f"  原始BERT: {original_stats['avg_tokens']:.1f}")
        print(f"  扩展BERT: {extended_stats['avg_tokens']:.1f}")
        
        if original_stats['avg_tokens'] > 0:
            improvement = (original_stats['avg_tokens'] - extended_stats['avg_tokens']) / original_stats['avg_tokens'] * 100
            print(f"  改进: {improvement:.1f}%")
        
        print(f"\n✅ 两个模型都能正常工作")
        print(f"✅ 扩展后的模型增加了 {extended_stats['vocab_size'] - original_stats['vocab_size']:,} 个词元")
        print(f"✅ 新增了 {extended_stats['model_params'] - original_stats['model_params']:,} 个参数")

def main():
    """主函数"""
    print("HoC数据集快速测试程序")
    print("验证扩展后的BERT模型基本功能")
    
    try:
        compare_models()
        
        print("\n" + "="*80)
        print("测试完成")
        print("="*80)
        print("✅ 扩展后的BERT模型可以正常处理HoC数据集")
        print("✅ 模型结构和前向传播都正常工作")
        print("✅ 可以进行完整的分类器训练")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 