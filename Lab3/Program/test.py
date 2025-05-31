#!/usr/bin/env python3
"""
使用已训练的WordPiece分词器示例
"""

from tokenizers import Tokenizer
import os

def load_and_test_tokenizer(tokenizer_path):
    """
    加载并测试分词器
    
    Args:
        tokenizer_path (str): 分词器文件路径
    """
    if not os.path.exists(tokenizer_path):
        print(f"错误: 找不到分词器文件 {tokenizer_path}")
        return None
    
    # 加载分词器
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"成功加载分词器: {tokenizer_path}")
    
    # 获取词汇表信息
    vocab = tokenizer.get_vocab()
    print(f"词汇表大小: {len(vocab)}")
    
    return tokenizer

def test_biomedical_texts(tokenizer):
    """
    测试生物医学文本的分词效果
    
    Args:
        tokenizer: 分词器对象
    """
    biomedical_texts = [
        "The patient presented with acute myocardial infarction following coronary artery occlusion.",
        "SARS-CoV-2 viral RNA was detected using reverse transcription polymerase chain reaction (RT-PCR).",
        "The protein expression levels were analyzed using Western blot and immunofluorescence microscopy.",
        "Gene editing with CRISPR-Cas9 technology enables precise modification of DNA sequences.",
        "Immunotherapy with checkpoint inhibitors has shown remarkable efficacy in cancer treatment.",
        "The mitochondrial dysfunction leads to impaired cellular respiration and energy metabolism.",
        "Antibiotics resistance mechanisms include enzymatic degradation and efflux pump activation.",
        "Stem cell differentiation is regulated by transcription factors and epigenetic modifications."
    ]
    
    print("\n=== 生物医学文本分词测试 ===")
    
    for i, text in enumerate(biomedical_texts, 1):
        print(f"\n【测试 {i}】")
        print(f"原文: {text}")
        
        # 编码
        encoded = tokenizer.encode(text)
        tokens = encoded.tokens
        token_ids = encoded.ids
        
        print(f"分词结果: {tokens}")
        print(f"Token数量: {len(tokens)}")
        print(f"Token IDs: {token_ids}")
        
        # 解码验证
        decoded = tokenizer.decode(token_ids)
        print(f"重构文本: {decoded}")
        
        # 检查分词效果
        original_words = text.split()
        print(f"原始单词数: {len(original_words)}, Token数: {len(tokens)}")

def analyze_subword_splitting(tokenizer):
    """
    分析子词分割效果
    
    Args:
        tokenizer: 分词器对象
    """
    print("\n=== 子词分割分析 ===")
    
    test_words = [
        "immunotherapy",
        "cardiomyopathy", 
        "pneumonia",
        "antibiotics",
        "pathogenesis",
        "mitochondrial",
        "pharmacokinetics",
        "electrocardiogram",
        "bronchoscopy",
        "endocrinology"
    ]
    
    for word in test_words:
        encoded = tokenizer.encode(word)
        tokens = encoded.tokens
        # 移除特殊标记
        word_tokens = [t for t in tokens if t not in ['[CLS]', '[SEP]']]
        
        print(f"{word:20} -> {word_tokens}")

def main():
    """主函数"""
    print("=== WordPiece分词器使用示例 ===")
    
    tokenizer_path = "output/wordpiece_tokenizer.json"
    
    tokenizer = None
    
    # 尝试加载演示版本
    if os.path.exists(tokenizer_path):
        print("使用完整版本分词器...")
        tokenizer = load_and_test_tokenizer(tokenizer_path)
    else:
        print("错误: 找不到任何已训练的分词器文件")
        print("请先运行 train.py")
        return
    
    if tokenizer:
        # 测试生物医学文本
        test_biomedical_texts(tokenizer)
        
        # 分析子词分割
        analyze_subword_splitting(tokenizer)

if __name__ == "__main__":
    main() 