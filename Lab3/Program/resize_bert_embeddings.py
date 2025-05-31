#!/usr/bin/env python3
"""
调整BERT模型的输入嵌入层以适应扩展后的词汇表
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, AutoTokenizer
import numpy as np
import os
import json

def resize_bert_embeddings(model_name='bert-base-uncased', 
                          extended_tokenizer_path='extended_bert_output',
                          output_path='extended_bert_model'):
    """
    调整BERT模型的嵌入层大小以匹配扩展后的词汇表
    
    Args:
        model_name (str): 原始BERT模型名称
        extended_tokenizer_path (str): 扩展后分词器路径
        output_path (str): 输出模型路径
    
    Returns:
        tuple: (调整后的模型, 扩展后的分词器)
    """
    print("="*80)
    print("调整BERT模型嵌入层以适应扩展词汇表")
    print("="*80)
    
    # 1. 加载原始BERT模型和扩展后的分词器
    print("\n【步骤1】加载模型和分词器")
    
    print(f"加载原始BERT模型: {model_name}")
    original_model = BertModel.from_pretrained(model_name)
    original_config = original_model.config
    
    print(f"加载扩展后的分词器: {extended_tokenizer_path}")
    extended_tokenizer = AutoTokenizer.from_pretrained(extended_tokenizer_path)
    
    # 获取词汇表大小
    original_vocab_size = original_config.vocab_size
    extended_vocab_size = len(extended_tokenizer.vocab)
    new_tokens_count = extended_vocab_size - original_vocab_size
    
    print(f"原始词汇表大小: {original_vocab_size}")
    print(f"扩展后词汇表大小: {extended_vocab_size}")
    print(f"新增词元数量: {new_tokens_count}")
    
    # 2. 创建新的配置和模型
    print("\n【步骤2】创建调整后的模型配置")
    
    # 更新配置
    new_config = BertConfig.from_pretrained(model_name)
    new_config.vocab_size = extended_vocab_size
    
    print(f"更新模型配置 - 新词汇表大小: {new_config.vocab_size}")
    
    # 创建新模型
    print("创建新的BERT模型...")
    new_model = BertModel(new_config)
    
    # 3. 复制原始权重
    print("\n【步骤3】复制原始模型权重")
    
    # 复制除嵌入层外的所有权重
    original_state_dict = original_model.state_dict()
    new_state_dict = new_model.state_dict()
    
    copied_layers = 0
    skipped_layers = 0
    
    for name, param in original_state_dict.items():
        if name in new_state_dict:
            if 'word_embeddings' not in name:
                # 复制非词嵌入层的权重
                new_state_dict[name].copy_(param)
                copied_layers += 1
            else:
                skipped_layers += 1
        else:
            print(f"警告: 层 {name} 在新模型中不存在")
    
    print(f"复制了 {copied_layers} 个层的权重")
    print(f"跳过了 {skipped_layers} 个嵌入层")
    
    # 4. 调整词嵌入层
    print("\n【步骤4】调整词嵌入层")
    
    # 获取原始词嵌入权重
    original_embeddings = original_model.embeddings.word_embeddings.weight.data
    embedding_dim = original_embeddings.size(1)
    
    print(f"嵌入维度: {embedding_dim}")
    print(f"原始嵌入形状: {original_embeddings.shape}")
    
    # 创建新的嵌入矩阵
    new_embeddings = torch.zeros(extended_vocab_size, embedding_dim)
    
    # 复制原始嵌入
    new_embeddings[:original_vocab_size] = original_embeddings
    
    # 5. 初始化新词元的嵌入
    print("\n【步骤5】初始化新词元的嵌入向量")
    
    if new_tokens_count > 0:
        # 方法1: 使用原始嵌入的均值和标准差进行初始化
        original_mean = original_embeddings.mean(dim=0)
        original_std = original_embeddings.std(dim=0)
        
        # 为新词元生成随机嵌入
        new_token_embeddings = torch.normal(
            mean=original_mean.unsqueeze(0).expand(new_tokens_count, -1),
            std=original_std.unsqueeze(0).expand(new_tokens_count, -1)
        )
        
        # 方法2: 也可以使用最相似词元的嵌入作为初始化（更复杂但可能更好）
        # 这里我们使用简单的统计初始化
        
        new_embeddings[original_vocab_size:] = new_token_embeddings
        
        print(f"使用统计初始化为 {new_tokens_count} 个新词元生成嵌入向量")
        print(f"初始化均值: {original_mean.mean():.6f}")
        print(f"初始化标准差: {original_std.mean():.6f}")
    
    # 将新嵌入矩阵设置到模型中
    new_model.embeddings.word_embeddings.weight.data = new_embeddings
    
    print(f"新嵌入矩阵形状: {new_embeddings.shape}")
    
    # 6. 调整位置嵌入（如果需要）
    print("\n【步骤6】检查其他嵌入层")
    
    # 复制位置嵌入和类型嵌入
    new_model.embeddings.position_embeddings.weight.data = \
        original_model.embeddings.position_embeddings.weight.data.clone()
    new_model.embeddings.token_type_embeddings.weight.data = \
        original_model.embeddings.token_type_embeddings.weight.data.clone()
    
    print("复制位置嵌入和类型嵌入")
    
    # 7. 保存调整后的模型
    print("\n【步骤7】保存调整后的模型")
    
    os.makedirs(output_path, exist_ok=True)
    
    # 保存模型
    new_model.save_pretrained(output_path)
    extended_tokenizer.save_pretrained(output_path)
    
    # 保存调整信息
    resize_info = {
        'original_vocab_size': original_vocab_size,
        'extended_vocab_size': extended_vocab_size,
        'new_tokens_count': new_tokens_count,
        'embedding_dim': embedding_dim,
        'original_model': model_name,
        'extended_tokenizer_path': extended_tokenizer_path
    }
    
    with open(os.path.join(output_path, 'resize_info.json'), 'w') as f:
        json.dump(resize_info, f, indent=2)
    
    print(f"模型保存到: {output_path}")
    print(f"调整信息保存到: {output_path}/resize_info.json")
    
    # 8. 验证模型
    print("\n【步骤8】验证调整后的模型")
    
    # 测试模型是否可以正常运行
    test_texts = [
        "The patient was diagnosed with acute myocardial infarction.",
        "Immunotherapy shows promising results in cancer treatment.",
        "Gene editing with CRISPR-Cas9 enables precise DNA modification."
    ]
    
    print("测试调整后的模型...")
    new_model.eval()
    
    with torch.no_grad():
        for i, text in enumerate(test_texts, 1):
            inputs = extended_tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            outputs = new_model(**inputs)
            
            print(f"测试 {i}: 输入token数: {inputs['input_ids'].size(1)}, "
                  f"输出形状: {outputs.last_hidden_state.shape}")
    
    print("\n✅ 模型调整完成！")
    print(f"✅ 原始模型词汇表: {original_vocab_size} → 扩展后: {extended_vocab_size}")
    print(f"✅ 新增 {new_tokens_count} 个生物医学词元")
    print(f"✅ 模型保存路径: {output_path}")
    
    return new_model, extended_tokenizer

def test_extended_model(model_path='extended_bert_model'):
    """
    测试扩展后的模型
    
    Args:
        model_path (str): 模型路径
    """
    print("\n" + "="*80)
    print("测试扩展后的BERT模型")
    print("="*80)
    
    # 加载模型和分词器
    model = BertModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"加载模型: {model_path}")
    print(f"词汇表大小: {len(tokenizer.vocab)}")
    print(f"嵌入层大小: {model.embeddings.word_embeddings.weight.shape}")
    
    # 测试生物医学文本
    biomedical_texts = [
        "The immunotherapy treatment showed significant efficacy in cancer patients.",
        "Myocardial infarction was diagnosed through electrocardiography.",
        "CRISPR-Cas9 gene editing enables precise genomic modifications.",
        "Pharmacokinetics studies revealed optimal drug dosing strategies.",
        "Mitochondrial dysfunction leads to impaired cellular respiration."
    ]
    
    print("\n测试生物医学文本处理:")
    model.eval()
    
    with torch.no_grad():
        for i, text in enumerate(biomedical_texts, 1):
            # 分词
            tokens = tokenizer.tokenize(text)
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            
            # 前向传播
            outputs = model(**inputs)
            
            print(f"\n【测试 {i}】")
            print(f"原文: {text}")
            print(f"分词: {tokens}")
            print(f"Token数: {len(tokens)}")
            print(f"输出形状: {outputs.last_hidden_state.shape}")
            print(f"池化表示: {outputs.pooler_output.shape}")

def compare_models():
    """
    比较原始BERT和扩展后BERT的差异
    """
    print("\n" + "="*80)
    print("比较原始BERT与扩展BERT")
    print("="*80)
    
    # 加载模型
    original_model = BertModel.from_pretrained('bert-base-uncased')
    original_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    extended_model = BertModel.from_pretrained('extended_bert_model')
    extended_tokenizer = AutoTokenizer.from_pretrained('extended_bert_model')
    
    print(f"原始BERT词汇表大小: {len(original_tokenizer.vocab)}")
    print(f"扩展BERT词汇表大小: {len(extended_tokenizer.vocab)}")
    
    # 测试文本
    test_text = "Immunotherapy with checkpoint inhibitors shows remarkable pharmacokinetics."
    
    print(f"\n测试文本: {test_text}")
    
    # 原始BERT分词
    original_tokens = original_tokenizer.tokenize(test_text)
    print(f"原始BERT分词: {original_tokens}")
    print(f"Token数量: {len(original_tokens)}")
    
    # 扩展BERT分词
    extended_tokens = extended_tokenizer.tokenize(test_text)
    print(f"扩展BERT分词: {extended_tokens}")
    print(f"Token数量: {len(extended_tokens)}")
    
    # 分析改进
    improvement = len(original_tokens) - len(extended_tokens)
    if improvement > 0:
        print(f"✅ 改进: 减少了 {improvement} 个token ({improvement/len(original_tokens)*100:.1f}%)")
    elif improvement == 0:
        print("➖ 无变化")
    else:
        print(f"❌ 增加了 {-improvement} 个token")

def main():
    """主函数"""
    print("BERT模型嵌入层调整程序")
    print("将扩展词汇表的BERT模型调整为适合训练的版本")
    
    try:
        # 1. 调整BERT模型嵌入层
        print("\n" + "="*80)
        print("开始调整BERT模型")
        print("="*80)
        
        model, tokenizer = resize_bert_embeddings(
            model_name='bert-base-uncased',
            extended_tokenizer_path='extended_bert_output',
            output_path='extended_bert_model'
        )
        
        # 2. 测试调整后的模型
        test_extended_model('extended_bert_model')
        
        # 3. 比较原始和扩展模型
        compare_models()
        
        print("\n" + "="*80)
        print("程序执行完成！")
        print("="*80)
        print("✅ BERT模型嵌入层已成功调整")
        print("✅ 模型已保存到 'extended_bert_model' 目录")
        print("✅ 可以使用该模型进行下游任务训练")
        
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 