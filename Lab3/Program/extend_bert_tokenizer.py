#!/usr/bin/env python3
"""
扩展BERT分词器：添加生物医学领域特定词元
从训练好的WordPiece分词器中选择5000个领域特定词元，并添加到bert-base-uncased分词器中
"""

import os
import json
from transformers import AutoTokenizer
from tokenizers import Tokenizer
from collections import Counter
import re

def load_trained_tokenizer(tokenizer_path):
    """
    加载训练好的WordPiece分词器
    
    Args:
        tokenizer_path (str): 分词器文件路径
    
    Returns:
        Tokenizer: 加载的分词器对象
    """
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"找不到分词器文件: {tokenizer_path}")
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"成功加载训练好的分词器: {tokenizer_path}")
    
    vocab = tokenizer.get_vocab()
    print(f"训练好的分词器词汇表大小: {len(vocab)}")
    
    return tokenizer

def load_bert_tokenizer():
    """
    加载原始的BERT分词器
    
    Returns:
        AutoTokenizer: BERT分词器对象
    """
    try:
        bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        print(f"成功加载BERT分词器")
        print(f"BERT原始词汇表大小: {len(bert_tokenizer.vocab)}")
        return bert_tokenizer
    except Exception as e:
        print(f"加载BERT分词器失败: {e}")
        print("请确保已安装transformers库并有网络连接")
        return None

def is_biomedical_term(token):
    """
    判断词元是否可能是生物医学术语
    
    Args:
        token (str): 词元
    
    Returns:
        bool: 是否是生物医学术语
    """
    # 排除特殊标记和标点符号
    special_tokens = ['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]']
    if token in special_tokens:
        return False
    
    # 排除单个字符和纯数字
    if len(token) <= 1 or token.isdigit():
        return False
    
    # 排除纯标点符号
    if re.match(r'^[^\w]+$', token):
        return False
    
    # 生物医学相关词汇模式
    biomedical_patterns = [
        # 生物学术语后缀
        r'.*(?:osis|itis|emia|uria|pathy|ology|ectomy|oma|genesis|lysis|trophy|plasia|gram|scopy)$',
        # 化学/药物术语
        r'.*(?:ine|ase|ide|ium|acid|sterol|peptide|protein|enzyme)$',
        # 解剖学术语
        r'.*(?:cardio|neuro|gastro|hepato|nephro|pulmo|dermo|osteo|myo|arthro).*',
        # 医学前缀
        r'^(?:anti|hyper|hypo|pre|post|inter|intra|extra|sub|trans|micro|macro|multi).*',
        # 基因/分子生物学
        r'.*(?:DNA|RNA|gene|genom|transcript|chromo|allele).*',
        # 细胞生物学
        r'.*(?:cell|cellular|mitoch|nuclei|cyto|membrane|receptor).*',
        # 免疫学
        r'.*(?:immun|antibod|antigen|lymph|leuko|cytokine).*',
        # 病理学
        r'.*(?:tumor|cancer|malign|benign|metasta|neoplas).*',
        # 药理学
        r'.*(?:therapeutic|pharmac|clinical|dosage|treatment).*'
    ]
    
    # 检查是否匹配任何生物医学模式
    token_lower = token.lower()
    for pattern in biomedical_patterns:
        if re.match(pattern, token_lower):
            return True
    
    # 检查是否包含生物医学关键词
    biomedical_keywords = [
        'protein', 'gene', 'cell', 'dna', 'rna', 'enzyme', 'hormone', 'virus', 'bacteria',
        'infection', 'disease', 'treatment', 'therapy', 'clinical', 'medical', 'patient',
        'diagnosis', 'symptom', 'syndrome', 'disorder', 'cancer', 'tumor', 'immune',
        'blood', 'tissue', 'organ', 'molecular', 'genetic', 'biological', 'biochemical',
        'physiological', 'pathological', 'pharmaceutical', 'metabolism', 'neuron',
        'cardiac', 'pulmonary', 'renal', 'hepatic', 'gastric', 'dermal', 'skeletal'
    ]
    
    for keyword in biomedical_keywords:
        if keyword in token_lower:
            return True
    
    return False

def select_domain_specific_tokens(trained_tokenizer, bert_tokenizer, num_tokens=5000):
    """
    从训练好的分词器中选择领域特定词元
    
    Args:
        trained_tokenizer: 训练好的分词器
        bert_tokenizer: BERT分词器
        num_tokens (int): 要选择的词元数量
    
    Returns:
        list: 选择的领域特定词元列表
    """
    print(f"正在选择 {num_tokens} 个领域特定词元...")
    
    # 获取训练好的分词器的词汇表
    trained_vocab = trained_tokenizer.get_vocab()
    
    # 获取BERT的词汇表
    bert_vocab = set(bert_tokenizer.vocab.keys())
    
    # 找出不在BERT词汇表中的词元
    new_tokens = []
    biomedical_tokens = []
    
    for token in trained_vocab.keys():
        if token not in bert_vocab:
            new_tokens.append(token)
            if is_biomedical_term(token):
                biomedical_tokens.append(token)
    
    print(f"训练好的分词器中新词元总数: {len(new_tokens)}")
    print(f"其中生物医学相关词元: {len(biomedical_tokens)}")
    
    # 优先选择生物医学词元，然后选择其他新词元
    selected_tokens = []
    
    # 首先添加生物医学词元
    if len(biomedical_tokens) >= num_tokens:
        # 按词元长度和复杂度排序，优先选择更有意义的词元
        biomedical_tokens.sort(key=lambda x: (-len(x), x))
        selected_tokens = biomedical_tokens[:num_tokens]
    else:
        selected_tokens.extend(biomedical_tokens)
        
        # 如果生物医学词元不够，从其他新词元中补充
        remaining_tokens = [t for t in new_tokens if t not in biomedical_tokens]
        # 过滤掉过短或看起来不重要的词元
        remaining_tokens = [t for t in remaining_tokens if len(t) >= 3 and not t.startswith('##')]
        
        # 按长度和字母顺序排序
        remaining_tokens.sort(key=lambda x: (-len(x), x))
        
        needed = num_tokens - len(selected_tokens)
        selected_tokens.extend(remaining_tokens[:needed])
    
    print(f"最终选择了 {len(selected_tokens)} 个词元")
    
    return selected_tokens

def extend_bert_tokenizer(bert_tokenizer, new_tokens, output_dir="./extended_bert_output"):
    """
    扩展BERT分词器，添加新词元
    
    Args:
        bert_tokenizer: BERT分词器
        new_tokens (list): 要添加的新词元列表
        output_dir (str): 输出目录
    
    Returns:
        AutoTokenizer: 扩展后的分词器
    """
    print(f"正在扩展BERT分词器，添加 {len(new_tokens)} 个新词元...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 添加新词元到分词器
    num_added = bert_tokenizer.add_tokens(new_tokens)
    print(f"成功添加了 {num_added} 个新词元")
    
    # 保存扩展后的分词器
    bert_tokenizer.save_pretrained(output_dir)
    print(f"扩展后的分词器已保存到: {output_dir}")
    
    # 保存添加的词元列表
    tokens_file = os.path.join(output_dir, "added_tokens.txt")
    with open(tokens_file, 'w', encoding='utf-8') as f:
        for token in new_tokens:
            f.write(f"{token}\n")
    print(f"添加的词元列表已保存到: {tokens_file}")
    
    # 创建映射文件
    mapping_file = os.path.join(output_dir, "token_mapping.json")
    token_mapping = {}
    for token in new_tokens:
        if token in bert_tokenizer.vocab:
            token_mapping[token] = bert_tokenizer.vocab[token]
    
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(token_mapping, f, indent=2, ensure_ascii=False)
    print(f"词元映射已保存到: {mapping_file}")
    
    print(f"扩展后的分词器词汇表大小: {len(bert_tokenizer.vocab)}")
    
    return bert_tokenizer

def test_extended_tokenizer(tokenizer, test_texts=None):
    """
    测试扩展后的分词器
    
    Args:
        tokenizer: 扩展后的分词器
        test_texts (list): 测试文本列表
    """
    if test_texts is None:
        test_texts = [
            "The patient was diagnosed with acute myocardial infarction.",
            "SARS-CoV-2 viral RNA was detected using RT-PCR.",
            "Immunotherapy with checkpoint inhibitors shows promising results.",
            "The protein expression levels were analyzed using Western blot.",
            "Gene editing with CRISPR-Cas9 enables precise DNA modification.",
            "Mitochondrial dysfunction leads to impaired cellular respiration.",
            "Antibiotics resistance is a major public health concern.",
            "Stem cell differentiation is regulated by transcription factors."
        ]
    
    print("\n=== 扩展后分词器测试 ===")
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n【测试 {i}】")
        print(f"原文: {text}")
        
        # 分词
        tokens = tokenizer.tokenize(text)
        print(f"分词结果: {tokens}")
        print(f"Token数量: {len(tokens)}")
        
        # 编码和解码
        encoded = tokenizer.encode(text, add_special_tokens=True)
        decoded = tokenizer.decode(encoded)
        print(f"编码: {encoded[:10]}..." if len(encoded) > 10 else f"编码: {encoded}")
        print(f"解码: {decoded}")

def analyze_improvement(original_tokenizer, extended_tokenizer, test_texts):
    """
    分析扩展后分词器的改进效果
    
    Args:
        original_tokenizer: 原始BERT分词器
        extended_tokenizer: 扩展后的分词器
        test_texts (list): 测试文本列表
    """
    print("\n=== 分词效果对比分析 ===")
    
    total_original_tokens = 0
    total_extended_tokens = 0
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n【对比 {i}】")
        print(f"原文: {text}")
        
        # 原始分词器
        original_tokens = original_tokenizer.tokenize(text)
        extended_tokens = extended_tokenizer.tokenize(text)
        
        print(f"原始BERT: {original_tokens} ({len(original_tokens)} tokens)")
        print(f"扩展BERT: {extended_tokens} ({len(extended_tokens)} tokens)")
        
        total_original_tokens += len(original_tokens)
        total_extended_tokens += len(extended_tokens)
        
        if len(extended_tokens) < len(original_tokens):
            print(f"✅ 改进: 减少了 {len(original_tokens) - len(extended_tokens)} 个token")
        elif len(extended_tokens) == len(original_tokens):
            print("➖ 无变化")
        else:
            print(f"❌ 增加了 {len(extended_tokens) - len(original_tokens)} 个token")
    
    print(f"\n=== 总体统计 ===")
    print(f"原始BERT总token数: {total_original_tokens}")
    print(f"扩展BERT总token数: {total_extended_tokens}")
    print(f"Token减少率: {(total_original_tokens - total_extended_tokens) / total_original_tokens * 100:.2f}%")

def main():
    """主函数"""
    print("=== BERT分词器扩展程序 ===")
    print("从训练好的WordPiece分词器中选择5000个领域特定词元")
    print("并添加到bert-base-uncased分词器中")
    
    # 文件路径
    trained_tokenizer_path = "output/wordpiece_tokenizer.json"
    output_dir = "extended_bert_output"
    
    try:
        # 1. 加载训练好的分词器
        print("\n【步骤1】加载训练好的WordPiece分词器")
        trained_tokenizer = load_trained_tokenizer(trained_tokenizer_path)
        
        # 2. 加载原始BERT分词器
        print("\n【步骤2】加载原始BERT分词器")
        bert_tokenizer = load_bert_tokenizer()
        if bert_tokenizer is None:
            return
        
        # 保存原始分词器用于对比
        original_bert = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # 3. 选择领域特定词元
        print("\n【步骤3】选择领域特定词元")
        domain_tokens = select_domain_specific_tokens(
            trained_tokenizer, bert_tokenizer, num_tokens=5000
        )
        
        # 显示一些选择的词元示例
        print(f"\n选择的词元示例（前20个）:")
        for i, token in enumerate(domain_tokens[:20]):
            print(f"  {i+1:2d}. {token}")
        
        # 4. 扩展BERT分词器
        print("\n【步骤4】扩展BERT分词器")
        extended_tokenizer = extend_bert_tokenizer(
            bert_tokenizer, domain_tokens, output_dir
        )
        
        # 5. 测试扩展后的分词器
        print("\n【步骤5】测试扩展后的分词器")
        test_extended_tokenizer(extended_tokenizer)
        
        # 6. 分析改进效果
        test_texts = [
            "The patient was diagnosed with acute myocardial infarction.",
            "SARS-CoV-2 viral RNA was detected using RT-PCR.",
            "Immunotherapy with checkpoint inhibitors shows promising results.",
            "Gene editing with CRISPR-Cas9 enables precise DNA modification."
        ]
        
        print("\n【步骤6】分析改进效果")
        analyze_improvement(original_bert, extended_tokenizer, test_texts)
        
        print(f"\n=== 完成! ===")
        print(f"扩展后的BERT分词器已保存到: {output_dir}")
        print(f"原始词汇表大小: {len(original_bert.vocab)}")
        print(f"扩展后词汇表大小: {len(extended_tokenizer.vocab)}")
        print(f"新增词元数量: {len(extended_tokenizer.vocab) - len(original_bert.vocab)}")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 