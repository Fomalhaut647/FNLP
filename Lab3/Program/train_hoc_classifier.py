#!/usr/bin/env python3
"""
使用扩展后的BERT模型在HoC (Hallmarks of Cancer) 数据集上训练分类器
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertModel, BertConfig, AutoTokenizer, 
    get_linear_schedule_with_warmup,
    TrainingArguments, Trainer
)
from datasets import load_dataset
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score
import os
import json
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HoCClassifier(nn.Module):
    """
    基于扩展BERT的HoC分类器
    """
    
    def __init__(self, bert_model_path, num_labels=10, dropout_rate=0.1):
        """
        初始化分类器
        
        Args:
            bert_model_path (str): BERT模型路径
            num_labels (int): 分类标签数量
            dropout_rate (float): Dropout率
        """
        super(HoCClassifier, self).__init__()
        
        # 加载BERT模型并移动到设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用的设备: {device}")
        self.bert = BertModel.from_pretrained(bert_model_path).to(device)
        
        # 分类头
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # 多标签分类使用sigmoid
        self.sigmoid = nn.Sigmoid()
        
        self.num_labels = num_labels
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            token_type_ids: token类型IDs
            labels: 标签（可选）
        
        Returns:
            dict: 包含损失和logits的字典
        """
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 使用[CLS]表示进行分类
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # 分类
        logits = self.classifier(pooled_output)
        
        # 计算损失（如果提供了标签）
        loss = None
        if labels is not None:
            # 多标签分类使用BCEWithLogitsLoss
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        
        return {
            'loss': loss,
            'logits': logits,
            'probabilities': self.sigmoid(logits)
        }

class HoCDataset(Dataset):
    """
    HoC数据集类
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        初始化数据集
        
        Args:
            texts (list): 文本列表
            labels (list): 标签列表
            tokenizer: 分词器
            max_length (int): 最大序列长度
        """
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
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

def load_hoc_dataset():
    """
    加载HoC数据集
    
    Returns:
        tuple: (训练集, 验证集, 测试集)
    """
    print("正在加载HoC数据集...")
    
    try:
        # 从Hugging Face加载数据集
        dataset = load_dataset("qanastek/HoC")
        
        print(f"数据集大小:")
        print(f"  训练集: {len(dataset['train'])}")
        print(f"  验证集: {len(dataset['validation'])}")
        print(f"  测试集: {len(dataset['test'])}")
        
        return dataset['train'], dataset['validation'], dataset['test']
        
    except Exception as e:
        print(f"加载HoC数据集失败: {e}")
        print("请确保已安装datasets库: pip install datasets")
        return None, None, None

def convert_labels_to_multilabel(labels, num_classes=10):
    """
    将标签列表转换为多标签二进制向量
    
    Args:
        labels (list): 标签列表，每个元素是一个标签ID列表
        num_classes (int): 总类别数
    
    Returns:
        numpy.ndarray: 多标签二进制矩阵
    """
    multilabel_matrix = np.zeros((len(labels), num_classes))
    
    for i, label_list in enumerate(labels):
        for label_id in label_list:
            if 0 <= label_id < num_classes:
                multilabel_matrix[i, label_id] = 1
    
    return multilabel_matrix

def prepare_datasets(train_data, val_data, test_data, tokenizer, max_length=512):
    """
    准备训练、验证和测试数据集
    
    Args:
        train_data: 训练数据
        val_data: 验证数据
        test_data: 测试数据
        tokenizer: 分词器
        max_length (int): 最大序列长度
    
    Returns:
        tuple: (训练数据集, 验证数据集, 测试数据集)
    """
    print("正在准备数据集...")
    
    # 提取文本和标签
    train_texts = [item['text'] for item in train_data]
    train_labels = [item['label'] for item in train_data]
    
    val_texts = [item['text'] for item in val_data]
    val_labels = [item['label'] for item in val_data]
    
    test_texts = [item['text'] for item in test_data]
    test_labels = [item['label'] for item in test_data]
    
    # 转换为多标签格式
    train_labels_binary = convert_labels_to_multilabel(train_labels)
    val_labels_binary = convert_labels_to_multilabel(val_labels)
    test_labels_binary = convert_labels_to_multilabel(test_labels)
    
    print(f"数据统计:")
    print(f"  训练集大小: {len(train_texts)}")
    print(f"  验证集大小: {len(val_texts)}")
    print(f"  测试集大小: {len(test_texts)}")
    print(f"  标签维度: {train_labels_binary.shape[1]}")
    
    # 创建数据集
    train_dataset = HoCDataset(train_texts, train_labels_binary, tokenizer, max_length)
    val_dataset = HoCDataset(val_texts, val_labels_binary, tokenizer, max_length)
    test_dataset = HoCDataset(test_texts, test_labels_binary, tokenizer, max_length)
    
    return train_dataset, val_dataset, test_dataset

def train_model(model, train_dataset, val_dataset, output_dir='hoc_classifier_output', 
                num_epochs=3, batch_size=16, learning_rate=2e-5):
    """
    训练模型
    
    Args:
        model: 模型
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        output_dir (str): 输出目录
        num_epochs (int): 训练轮数
        batch_size (int): 批次大小
        learning_rate (float): 学习率
    
    Returns:
        训练后的模型
    """
    print(f"开始训练模型...")
    print(f"  训练轮数: {num_epochs}")
    print(f"  批次大小: {batch_size}")
    print(f"  学习率: {learning_rate}")
    
    # 定义优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 检查是否有可用的GPU，并设置device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # 训练循环
    best_val_f1 = 0
    
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        
        # 训练阶段
        model.train()
        total_train_loss = 0
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1} - Training")
        
        for batch in train_progress:
            # 移动到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            loss = outputs['loss']
            total_train_loss += loss.item()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # 更新进度条
            train_progress.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # 验证阶段
        val_f1, val_loss = evaluate_model(model, val_loader, device)
        
        print(f"训练损失: {avg_train_loss:.4f}")
        print(f"验证损失: {val_loss:.4f}")
        print(f"验证F1: {val_f1:.4f}")
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f"新的最佳F1分数: {best_val_f1:.4f}")
            
            # 保存模型
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
            
            # 保存训练信息
            train_info = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'val_f1': val_f1,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            }
            
            with open(os.path.join(output_dir, 'training_info.json'), 'w') as f:
                json.dump(train_info, f, indent=2)
    
    print(f"\n训练完成！最佳验证F1: {best_val_f1:.4f}")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pt')))
    
    return model

def evaluate_model(model, data_loader, device, threshold=0.5):
    """
    评估模型
    
    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备
        threshold (float): 二分类阈值
    
    Returns:
        tuple: (F1分数, 平均损失)
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            loss = outputs['loss']
            total_loss += loss.item()
            
            # 获取预测
            probabilities = outputs['probabilities']
            predictions = (probabilities > threshold).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # 计算每个类别的F1分数
    f1_scores = []
    for i in range(all_labels.shape[1]):
        f1 = f1_score(all_labels[:, i], all_predictions[:, i], average='binary', zero_division=0)
        f1_scores.append(f1)
    
    # 宏平均F1
    macro_f1 = np.mean(f1_scores)
    avg_loss = total_loss / len(data_loader)
    
    return macro_f1, avg_loss

def test_model(model, test_dataset, tokenizer, output_dir='hoc_classifier_output'):
    """
    测试模型并生成详细报告
    
    Args:
        model: 训练好的模型
        test_dataset: 测试数据集
        tokenizer: 分词器
        output_dir (str): 输出目录
    """
    print("\n" + "="*80)
    print("模型测试与评估")
    print("="*80)
    
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HoCClassifier(model_path, num_labels=num_labels)
    model.to(device)
    model.eval()
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 评估
    test_f1, test_loss = evaluate_model(model, test_loader, device)
    
    print(f"测试结果:")
    print(f"  测试损失: {test_loss:.4f}")
    print(f"  测试F1分数: {test_f1:.4f}")
    
    # 生成详细预测
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="生成预测"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            probabilities = outputs['probabilities']
            predictions = (probabilities > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # 保存结果
    results = {
        'test_loss': test_loss,
        'test_f1': test_f1,
        'predictions': np.array(all_predictions).tolist(),
        'labels': np.array(all_labels).tolist(),
        'probabilities': np.array(all_probabilities).tolist()
    }
    
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"测试结果已保存到: {output_dir}/test_results.json")
    
    # 打印每个类别的性能
    hallmarks = [
        "Sustaining Proliferative Signaling",
        "Evading Growth Suppressors", 
        "Resisting Cell Death",
        "Enabling Replicative Immortality",
        "Inducing Angiogenesis",
        "Activating Invasion & Metastasis",
        "Genome Instability & Mutation",
        "Tumor-Promoting Inflammation",
        "Deregulating Cellular Energetics",
        "Avoiding Immune Destruction"
    ]
    
    print(f"\n各类别F1分数:")
    all_labels_np = np.array(all_labels)
    all_predictions_np = np.array(all_predictions)
    
    for i, hallmark in enumerate(hallmarks):
        if i < all_labels_np.shape[1]:
            f1 = f1_score(all_labels_np[:, i], all_predictions_np[:, i], average='binary', zero_division=0)
            print(f"  {i}: {hallmark}: {f1:.4f}")

def main():
    """主函数"""
    print("="*80)
    print("HoC (Hallmarks of Cancer) 分类器训练")
    print("使用扩展后的BERT模型进行多标签分类")
    print("="*80)
    
    # 检查扩展后的BERT模型是否存在
    extended_model_path = 'extended_bert_model'
    if not os.path.exists(extended_model_path):
        print(f"❌ 找不到扩展后的BERT模型: {extended_model_path}")
        print("请先运行 resize_bert_embeddings.py 来创建扩展后的模型")
        return
    
    try:
        # 1. 加载数据集
        print("\n【步骤1】加载HoC数据集")
        train_data, val_data, test_data = load_hoc_dataset()
        
        if train_data is None:
            print("❌ 数据集加载失败")
            return
        
        # 2. 加载扩展后的分词器
        print("\n【步骤2】加载扩展后的分词器")
        tokenizer = AutoTokenizer.from_pretrained(extended_model_path)
        print(f"分词器词汇表大小: {len(tokenizer.vocab)}")
        
        # 3. 准备数据集
        print("\n【步骤3】准备数据集")
        train_dataset, val_dataset, test_dataset = prepare_datasets(
            train_data, val_data, test_data, tokenizer, max_length=512
        )
        
        # 4. 创建模型
        print("\n【步骤4】创建分类模型")
        model = HoCClassifier(
            bert_model_path=extended_model_path,
            num_labels=10,  # HoC数据集有10个类别
            dropout_rate=0.1
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型参数总数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        
        # 5. 训练模型
        print("\n【步骤5】开始训练")
        trained_model = train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir='hoc_classifier_output',
            num_epochs=3,
            batch_size=16,
            learning_rate=2e-5
        )
        
        # 6. 测试模型
        print("\n【步骤6】测试模型")
        test_model(trained_model, test_dataset, tokenizer)
        
        print("\n" + "="*80)
        print("训练完成！")
        print("="*80)
        print("✅ 模型已成功训练并保存")
        print("✅ 测试结果已生成")
        print("✅ 输出文件保存在 'hoc_classifier_output' 目录")
        
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 