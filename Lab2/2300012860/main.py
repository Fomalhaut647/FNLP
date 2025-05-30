#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py  —— 统一脚本：Log‑Linear  与  BERT 文本分类
Author: (your name)
Course: Foundations of NLP, PKU 2025 Spring

用法说明
-----
# 单个模型在单个数据集上运行
python main.py --model loglinear --dataset 20news
python main.py --model loglinear --dataset hoc
python main.py --model bert       --dataset 20news
python main.py --model bert       --dataset hoc

# 一键运行所有实验组合
python main.py --all
"""
import argparse, json, os, random, pathlib, sys
import numpy as np, pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
import datasets

# ---------------- 全局随机种子设定 ----------------
# 设定随机种子以保证实验结果的可复现性
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ================= 工具函数 ===================
def compute_metrics(true_labels, predicted_labels):
    """
    计算并返回评估指标字典，包含准确率、宏平均F1和微平均F1。

    Args:
        true_labels (list or np.array): 真实的标签列表。
        predicted_labels (list or np.array): 模型预测的标签列表。

    Returns:
        dict: 包含 'accuracy', 'macro_f1', 'micro_f1' 的字典，值已四舍五入到小数点后4位。
    """
    from sklearn.metrics import accuracy_score, f1_score
    # 计算准确率
    accuracy = accuracy_score(true_labels, predicted_labels)
    # 计算宏平均 F1 分数
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro')
    # 计算微平均 F1 分数
    micro_f1 = f1_score(true_labels, predicted_labels, average='micro')
    # 返回包含所有指标的字典
    return {"accuracy": round(accuracy, 4), "macro_f1": round(macro_f1, 4), "micro_f1": round(micro_f1, 4)}

def save_results(json_filepath, model_name, dataset_name, train_metrics, test_metrics):
    """
    将训练和测试指标保存到指定的 JSON 文件中。

    Args:
        json_filepath (str): 保存结果的 JSON 文件路径。
        model_name (str): 当前使用的模型名称（例如 'loglinear', 'bert'）。
        dataset_name (str): 当前使用的数据集名称（例如 '20news', 'hoc'）。
        train_metrics (dict): 训练集上的评估指标。
        test_metrics (dict): 测试集上的评估指标。
    """
    # 检查 JSON 文件是否存在，如果存在则加载现有数据
    if os.path.exists(json_filepath):
        with open(json_filepath, 'r', encoding='utf-8') as f:
            all_results_data = json.load(f)
    else:
        # 如果文件不存在，则初始化一个空字典
        all_results_data = {}

    # 使用 setdefault 确保模型键存在于字典中
    all_results_data.setdefault(model_name, {})
    # 将当前实验的结果（训练和测试指标）存储到对应模型和数据集的条目下
    all_results_data[model_name][dataset_name] = {"train": train_metrics, "test": test_metrics}

    # 将更新后的结果数据写回 JSON 文件，使用 indent=2 格式化输出
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(all_results_data, f, indent=2, ensure_ascii=False) # ensure_ascii=False 保证中文正常显示

# ================= 数据加载函数 ===================
def load_20news_data():
    """
    加载 20 Newsgroups 数据集。

    Returns:
        tuple: 包含训练文本、训练标签、测试文本、测试标签和类别数量的元组。
    """
    from datasets import load_dataset
    # 从 Hugging Face Hub 加载 SetFit/20_newsgroups 数据集
    dataset = load_dataset('SetFit/20_newsgroups')
    # 提取训练集和测试集的文本和标签
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']
    # 数据集类别数量
    num_classes = 20
    return train_texts, train_labels, test_texts, test_labels, num_classes

def load_hoc_data():
    """
    加载 HoC (Hallmarks of Cancer) 数据集。

    Returns:
        tuple: 包含训练文本、训练标签、测试文本、测试标签和类别数量的元组。
    """
    # 内部函数，用于从 Parquet 文件读取数据
    def read_split(split_name):
        # 从指定路径读取 Parquet 文件
        df = pd.read_parquet(f'./data/HoC/{split_name}.parquet')
        # 返回文本列表和标签列表
        return df['text'].tolist(), df['label'].tolist()

    # 读取训练集和测试集数据
    train_texts, train_labels = read_split('train')
    test_texts, test_labels = read_split('test')
    # 数据集类别数量
    num_classes = 11
    return train_texts, train_labels, test_texts, test_labels, num_classes

# ================= Log‑Linear 模型训练与预测 =================
def run_loglinear_model(train_texts, train_labels, test_texts, test_labels):
    """
    使用 Log-Linear 模型（逻辑回归 + TF-IDF）进行训练和预测。

    Args:
        train_texts (list): 训练文本列表。
        train_labels (list): 训练标签列表。
        test_texts (list): 测试文本列表。
        test_labels (list): 测试标签列表（此函数中未使用，但保持接口一致性）。

    Returns:
        tuple: 包含训练集预测标签和测试集预测标签的元组。
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    # 构建一个处理流程：先进行 TF-IDF 向量化，然后进行逻辑回归分类
    classifier_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),         # 使用 unigrams 和 bigrams
            max_features=50000,        # 最大特征数
            sublinear_tf=True,         # 应用 sublinear TF 缩放 (1 + log(tf))
            stop_words='english')),    # 使用英文停用词表
        ('lr', LogisticRegression(
            max_iter=1000,             # 最大迭代次数
            C=4.0,                     # 正则化强度的倒数
            class_weight='balanced',   # 自动调整类别权重以处理不平衡数据
            n_jobs=-1,                 # 使用所有可用的 CPU 核心
            random_state=RANDOM_SEED)) # 设置随机种子以保证结果可复现
    ])

    # 使用训练数据拟合（训练）整个流程
    classifier_pipeline.fit(train_texts, train_labels)

    # 对训练集和测试集进行预测
    train_predictions = classifier_pipeline.predict(train_texts)
    test_predictions = classifier_pipeline.predict(test_texts)

    return train_predictions, test_predictions

# =================== BERT 模型训练与预测 ===================
def run_bert_model(train_texts, train_labels, test_texts, test_labels, num_classes):
    """
    使用 BERT 模型进行训练和预测。

    Args:
        train_texts (list): 训练文本列表。
        train_labels (list): 训练标签列表。
        test_texts (list): 测试文本列表。
        test_labels (list): 测试标签列表。
        num_classes (int): 数据集的类别数量。

    Returns:
        tuple: 包含训练集预测标签和测试集预测标签的元组。
    """
    from transformers import (BertTokenizerFast, BertForSequenceClassification,
                              TrainingArguments, Trainer)

    # --- 1. 构造 Hugging Face Dataset 对象 ---
    # 辅助函数：将文本和标签列表转换为 Hugging Face Dataset 格式
    def convert_to_hf_dataset(texts, labels):
        return Dataset.from_dict({'text': texts, 'labels': labels})

    # 创建包含训练集和测试集的 DatasetDict
    dataset_dict = DatasetDict({
        'train': convert_to_hf_dataset(train_texts, train_labels),
        'test' : convert_to_hf_dataset(test_texts,  test_labels)
    })

    # --- 2. 数据预处理：Tokenization ---
    # 加载 'bert-base-uncased' 模型的快速分词器
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # 定义 Tokenization 函数：对文本进行编码，处理截断和填充
    def tokenize_function(examples):
        return tokenizer(examples['text'],
                         truncation=True,       # 截断超过最大长度的序列
                         padding="max_length",  # 填充到最大长度
                         max_length=128)        # 设定最大序列长度

    # 对整个数据集应用 Tokenization 函数（使用 batched=True 加速）
    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

    # 设置数据集格式为 PyTorch 张量，并指定模型所需的列
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'token_type_ids',
                                                        'attention_mask', 'labels'])

    # --- 3. 加载预训练 BERT 模型 ---
    # 加载用于序列分类的 BERT 模型，指定类别数量
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=num_classes)

    # --- 4. 设置训练参数 ---
    # 定义训练参数
    training_args = TrainingArguments(
        output_dir='outputs/tmp',           # 模型输出和检查点保存目录
        per_device_train_batch_size=8,     # 每个设备上的训练批次大小
        per_device_eval_batch_size=8,      # 每个设备上的评估批次大小
        num_train_epochs=3,                # 训练的总轮数
        learning_rate=2e-5,                # 学习率
        logging_steps=200,                 # 每隔多少步记录一次日志
        seed=RANDOM_SEED                   # 设置训练过程的随机种子
    )

    # --- 5. 初始化 Trainer ---
    # 创建 Trainer 实例，传入模型、训练参数和训练数据集
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=tokenized_datasets['train'])

    # --- 6. 模型训练 ---
    trainer.train()

    # --- 7. 模型预测 ---
    # 定义预测函数
    def predict_on_split(split_name):
        # 使用 Trainer 的 predict 方法获取指定数据集划分上的预测结果
        # .predictions 属性包含模型的原始输出（logits）
        logits = trainer.predict(tokenized_datasets[split_name]).predictions
        # 对 logits 在类别维度上取 argmax，得到预测的类别索引
        return np.argmax(logits, axis=1)

    # 对训练集和测试集进行预测
    train_predictions = predict_on_split('train')
    test_predictions = predict_on_split('test')

    return train_predictions, test_predictions

# ================ 主实验流程 ======================
def run_single_experiment(model_choice, dataset_choice):
    """
    执行单个模型的单个数据集实验。

    Args:
        model_choice (str): 要使用的模型名称 ('loglinear' 或 'bert')。
        dataset_choice (str): 要使用的数据集名称 ('20news' 或 'hoc')。
    """
    print(f"\n--- 开始实验: 模型={model_choice.upper()}, 数据集={dataset_choice} ---")

    # --- 1. 加载数据 ---
    print("正在加载数据...")
    if dataset_choice == '20news':
        # 加载 20 Newsgroups 数据
        train_texts, train_labels, test_texts, test_labels, num_classes = load_20news_data()
    elif dataset_choice == 'hoc':
        # 加载 HoC 数据
        train_texts, train_labels, test_texts, test_labels, num_classes = load_hoc_data()
    else:
        # 处理无效的数据集名称
        raise ValueError(f"未知的数据集名称: {dataset_choice}")
    print(f"数据加载完成: 训练集 {len(train_texts)} 条, 测试集 {len(test_texts)} 条, 类别数 {num_classes}")

    # --- 2. 运行模型训练和预测 ---
    print(f"正在使用 {model_choice.upper()} 模型进行训练和预测...")
    if model_choice == 'loglinear':
        # 运行 Log-Linear 模型
        predicted_train_labels, predicted_test_labels = run_loglinear_model(train_texts, train_labels, test_texts, test_labels)
    elif model_choice == 'bert':
        # 运行 BERT 模型
        predicted_train_labels, predicted_test_labels = run_bert_model(train_texts, train_labels, test_texts, test_labels, num_classes)
    else:
        # 处理无效的模型名称
        raise ValueError(f"未知的模型名称: {model_choice}")
    print("模型训练和预测完成。")

    # --- 3. 计算和保存评估指标 ---
    print("正在计算评估指标...")
    # 计算训练集指标
    train_metrics_results = compute_metrics(train_labels, predicted_train_labels)
    # 计算测试集指标
    test_metrics_results = compute_metrics(test_labels, predicted_test_labels)

    print("正在保存结果...")
    # 将结果保存到 JSON 文件
    results_filepath = 'outputs/results.json'
    save_results(results_filepath, model_choice, dataset_choice,
                 train_metrics_results, test_metrics_results)
    print(f"结果已保存至 {results_filepath}")

    # --- 4. 打印结果 ---
    print(f"✔ 实验完成: {model_choice.upper()} on {dataset_choice}:")
    print(f"  训练集指标 =", train_metrics_results)
    print(f"  测试集指标 =", test_metrics_results)
    print("-" * (len(f"--- 开始实验: 模型={model_choice.upper()}, 数据集={dataset_choice} ---")))


def main():
    """主函数：解析命令行参数并执行实验。"""
    # --- 1. 设置命令行参数解析器 ---
    parser = argparse.ArgumentParser(description="运行 Log-Linear 或 BERT 文本分类实验")
    parser.add_argument('--model', choices=['loglinear', 'bert'],
                        help="选择要使用的模型: 'loglinear' 或 'bert'")
    parser.add_argument('--dataset', choices=['20news', 'hoc'],
                        help="选择要使用的数据集: '20news' 或 'hoc'")
    parser.add_argument('--all', action='store_true',
                        help="如果设置此标志，则运行所有四个模型和数据集的组合")
    # 解析命令行参数
    args = parser.parse_args()

    # --- 2. 准备输出目录 ---
    # 创建 'outputs' 目录（如果不存在）
    pathlib.Path('outputs').mkdir(exist_ok=True)

    # --- 3. 确定要运行的实验组合 ---
    # 定义所有可能的实验组合
    all_combinations = [('loglinear', '20news'),
                        ('loglinear', 'hoc'),
                        ('bert', '20news'),
                        ('bert', 'hoc')]

    # 根据命令行参数确定要运行的组合
    if args.all:
        # 如果指定了 --all，则运行所有组合
        experiments_to_run = all_combinations
        print("将运行所有实验组合...")
    elif args.model and args.dataset:
        # 如果指定了模型和数据集，则只运行该组合
        experiments_to_run = [(args.model, args.dataset)]
        print(f"将运行单个实验: 模型={args.model}, 数据集={args.dataset}")
    else:
        # 如果参数不完整且未指定 --all，则提示错误并退出
        print("错误：请提供 '--model' 和 '--dataset' 参数，或者使用 '--all' 选项运行所有组合。")
        # 打印帮助信息
        parser.print_help()
        sys.exit(1) # 退出程序，返回错误码 1

    # --- 4. 循环执行选定的实验 ---
    for model_name, dataset_name in experiments_to_run:
        run_single_experiment(model_name, dataset_name)

    print("\n所有指定实验均已完成。")

# 当脚本作为主程序执行时，调用 main 函数
if __name__ == '__main__':
    main()