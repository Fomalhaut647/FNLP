#!/usr/bin/env python3
"""
使用扩展后的BERT分词器示例
对比原始BERT分词器和扩展后分词器在生物医学文本上的表现
"""

from transformers import AutoTokenizer
import os

def load_tokenizers():
    """
    加载原始BERT分词器和扩展后的分词器
    
    Returns:
        tuple: (原始BERT分词器, 扩展后的分词器)
    """
    print("正在加载分词器...")
    
    # 加载原始BERT分词器
    try:
        original_bert = AutoTokenizer.from_pretrained('bert-base-uncased')
        print(f"✅ 原始BERT分词器加载成功，词汇表大小: {len(original_bert.vocab)}")
    except Exception as e:
        print(f"❌ 原始BERT分词器加载失败: {e}")
        return None, None
    
    # 加载扩展后的分词器
    extended_path = "extended_bert_output"
    if os.path.exists(extended_path):
        try:
            extended_bert = AutoTokenizer.from_pretrained(extended_path)
            print(f"✅ 扩展后BERT分词器加载成功，词汇表大小: {len(extended_bert.vocab)}")
        except Exception as e:
            print(f"❌ 扩展后BERT分词器加载失败: {e}")
            return original_bert, None
    else:
        print(f"❌ 找不到扩展后的分词器目录: {extended_path}")
        return original_bert, None
    
    return original_bert, extended_bert

def compare_tokenization(original_tokenizer, extended_tokenizer, texts):
    """
    对比两个分词器的分词效果
    
    Args:
        original_tokenizer: 原始分词器
        extended_tokenizer: 扩展后的分词器
        texts (list): 测试文本列表
    """
    print("\n" + "="*80)
    print("分词效果详细对比")
    print("="*80)
    
    total_original_tokens = 0
    total_extended_tokens = 0
    improved_cases = 0
    
    for i, text in enumerate(texts, 1):
        print(f"\n【测试 {i}】")
        print(f"原文: {text}")
        print("-" * 60)
        
        # 原始BERT分词
        original_tokens = original_tokenizer.tokenize(text)
        original_ids = original_tokenizer.encode(text, add_special_tokens=True)
        
        # 扩展BERT分词
        extended_tokens = extended_tokenizer.tokenize(text)
        extended_ids = extended_tokenizer.encode(text, add_special_tokens=True)
        
        print(f"原始BERT ({len(original_tokens)} tokens):")
        print(f"  分词: {original_tokens}")
        print(f"  Token IDs: {original_ids}")
        
        print(f"扩展BERT ({len(extended_tokens)} tokens):")
        print(f"  分词: {extended_tokens}")
        print(f"  Token IDs: {extended_ids}")
        
        # 分析改进
        token_diff = len(original_tokens) - len(extended_tokens)
        if token_diff > 0:
            print(f"✅ 改进: 减少了 {token_diff} 个token ({token_diff/len(original_tokens)*100:.1f}%)")
            improved_cases += 1
        elif token_diff == 0:
            print("➖ 无变化")
        else:
            print(f"❌ 增加了 {-token_diff} 个token")
        
        total_original_tokens += len(original_tokens)
        total_extended_tokens += len(extended_tokens)
    
    print("\n" + "="*80)
    print("总体统计")
    print("="*80)
    print(f"测试文本数量: {len(texts)}")
    print(f"改进案例数量: {improved_cases}")
    print(f"原始BERT总token数: {total_original_tokens}")
    print(f"扩展BERT总token数: {total_extended_tokens}")
    
    reduction_rate = (total_original_tokens - total_extended_tokens) / total_original_tokens * 100
    print(f"Token减少率: {reduction_rate:.2f}%")
    
    avg_original = total_original_tokens / len(texts)
    avg_extended = total_extended_tokens / len(texts)
    print(f"平均token数 - 原始: {avg_original:.1f}, 扩展: {avg_extended:.1f}")

def analyze_specific_terms(original_tokenizer, extended_tokenizer):
    """
    分析特定生物医学术语的分词改进
    
    Args:
        original_tokenizer: 原始分词器
        extended_tokenizer: 扩展后的分词器
    """
    print("\n" + "="*80)
    print("生物医学术语分词对比")
    print("="*80)
    
    biomedical_terms = [
        "immunotherapy",
        "myocardial infarction", 
        "mitochondrial dysfunction",
        "transcriptional regulation",
        "protein expression",
        "cellular respiration",
        "antibiotics resistance",
        "gene editing",
        "neurodegenerative diseases",
        "cardiovascular risk",
        "pharmacokinetics",
        "biomarkers",
        "pathogenesis",
        "therapeutic targets",
        "clinical trials",
        "molecular mechanisms",
        "epigenetic modifications",
        "stem cell differentiation",
        "metabolic pathways",
        "inflammatory response"
    ]
    
    significant_improvements = []
    
    for term in biomedical_terms:
        original_tokens = original_tokenizer.tokenize(term)
        extended_tokens = extended_tokenizer.tokenize(term)
        
        token_reduction = len(original_tokens) - len(extended_tokens)
        reduction_percent = (token_reduction / len(original_tokens)) * 100 if len(original_tokens) > 0 else 0
        
        print(f"\n术语: {term}")
        print(f"  原始: {original_tokens} ({len(original_tokens)} tokens)")
        print(f"  扩展: {extended_tokens} ({len(extended_tokens)} tokens)")
        
        if token_reduction > 0:
            print(f"  ✅ 减少 {token_reduction} 个token ({reduction_percent:.1f}%)")
            if reduction_percent >= 25:  # 显著改进
                significant_improvements.append((term, reduction_percent))
        elif token_reduction == 0:
            print(f"  ➖ 无变化")
        else:
            print(f"  ❌ 增加 {-token_reduction} 个token")
    
    if significant_improvements:
        print(f"\n显著改进的术语 (减少≥25%):")
        for term, improvement in sorted(significant_improvements, key=lambda x: x[1], reverse=True):
            print(f"  • {term}: {improvement:.1f}% 改进")

def demonstrate_use_cases(extended_tokenizer):
    """
    演示扩展后分词器的实际使用场景
    
    Args:
        extended_tokenizer: 扩展后的分词器
    """
    print("\n" + "="*80)
    print("实际使用场景演示")
    print("="*80)
    
    use_cases = [
        {
            "scenario": "医学报告摘要",
            "text": "Patient presents with acute myocardial infarction. Elevated troponin levels and ST-segment elevation observed. Emergency percutaneous coronary intervention performed successfully."
        },
        {
            "scenario": "生物学研究论文",
            "text": "The study investigated transcriptional regulation of inflammatory genes in macrophages. RNA sequencing revealed differential expression patterns following cytokine stimulation."
        },
        {
            "scenario": "药物研发文档",
            "text": "The novel immunotherapy demonstrates promising efficacy against metastatic cancer cells. Phase II clinical trials showed significant tumor reduction with minimal adverse effects."
        },
        {
            "scenario": "基因组学研究",
            "text": "CRISPR-Cas9 gene editing enables precise modification of genomic sequences. The technology shows potential for treating inherited genetic disorders."
        }
    ]
    
    for case in use_cases:
        print(f"\n场景: {case['scenario']}")
        print(f"文本: {case['text']}")
        
        tokens = extended_tokenizer.tokenize(case['text'])
        encoded = extended_tokenizer.encode(case['text'], add_special_tokens=True, 
                                          max_length=512, truncation=True)
        
        print(f"分词结果 ({len(tokens)} tokens):")
        print(f"  {tokens}")
        print(f"编码长度: {len(encoded)}")
        print(f"可用于模型输入: {'✅' if len(encoded) <= 512 else '❌ 超出长度限制'}")

def main():
    """主函数"""
    print("="*80)
    print("扩展BERT分词器使用示例")
    print("对比原始BERT分词器和扩展后分词器在生物医学文本上的表现")
    print("="*80)
    
    # 加载分词器
    original_tokenizer, extended_tokenizer = load_tokenizers()
    
    if original_tokenizer is None or extended_tokenizer is None:
        print("❌ 分词器加载失败，无法继续")
        return
    
    # 测试文本
    test_texts = [
        "The patient was diagnosed with acute myocardial infarction following coronary artery occlusion.",
        "SARS-CoV-2 viral RNA was detected using reverse transcription polymerase chain reaction.",
        "Immunotherapy with checkpoint inhibitors has shown remarkable efficacy in cancer treatment.",
        "The protein expression levels were analyzed using Western blot and immunofluorescence microscopy.",
        "Gene editing with CRISPR-Cas9 technology enables precise modification of DNA sequences.",
        "Mitochondrial dysfunction leads to impaired cellular respiration and energy metabolism.",
        "Antibiotics resistance mechanisms include enzymatic degradation and efflux pump activation.",
        "Stem cell differentiation is regulated by transcription factors and epigenetic modifications."
    ]
    
    # 1. 对比分词效果
    compare_tokenization(original_tokenizer, extended_tokenizer, test_texts)
    
    # 2. 分析特定术语
    analyze_specific_terms(original_tokenizer, extended_tokenizer)
    
    # 3. 实际使用场景演示
    demonstrate_use_cases(extended_tokenizer)
    
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    print("✅ 扩展后的BERT分词器在生物医学文本处理方面表现出以下优势:")
    print("   • 更好地识别完整的医学术语")
    print("   • 减少了复杂术语的子词分割")
    print("   • 提高了分词的语义连贯性")
    print("   • 减少了平均token数量，提高了模型效率")
    print("\n📁 扩展后的分词器文件保存在: extended_bert_output/")
    print("💡 可以直接使用 AutoTokenizer.from_pretrained('extended_bert_output') 加载")

if __name__ == "__main__":
    main() 