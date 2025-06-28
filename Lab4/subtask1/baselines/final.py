import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import requests
from tap import Tap
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# 全局变量，用于缓存模型和向量，避免重复加载，提升效率
EMBEDDING_MODEL = None
EMBEDDING_RERANK_MODEL = None
GRAMMAR_BOOK_VECTORS = {} # 使用字典缓存不同模型的向量
ALL_EXAMPLES_VECTORS = {}   # 使用字典缓存不同模型的向量

# ----------------------------------------------------------------------------
# 核心功能区
# ----------------------------------------------------------------------------

def load_embedding_model(model_name: str) -> 'SentenceTransformer':
    """加载并缓存指定的语义向量模型。"""
    global EMBEDDING_MODEL, EMBEDDING_RERANK_MODEL
    
    # 决定使用哪个全局变量来缓存
    if 'rerank' in model_name or 'e5' in model_name or 'mpnet' in model_name:
        model_cache = 'EMBEDDING_RERANK_MODEL'
    else:
        model_cache = 'EMBEDDING_MODEL'

    if globals()[model_cache] is None:
        try:
            from sentence_transformers import SentenceTransformer
            print(f">>> 正在加载语义模型: {model_name} (首次运行需要下载，请耐心等待)...")
            globals()[model_cache] = SentenceTransformer(model_name)
            print(f">>> {model_name} 加载完毕。")
        except ImportError:
            print("\n错误: 未找到所需库。请执行 pip install sentence-transformers torch scikit-learn")
            sys.exit(1)
    return globals()[model_cache]

def get_qwen_max_translation(prompt: str) -> str:
  
    api_key = os.getenv("API_KEY") # <--- 请务必替换为您的真实、有效的API Key
    if "sk-" not in api_key:
        raise ValueError("API Key格式不正确或未设置，请在脚本 get_qwen_max_translation 函数中写入。")

    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": "qwen-max", # 这是一个示例，请根据需要替换，比如 "deepseek-moe-16b-chat"
        "input": {
            "messages": [
                {
                    "role": "user", # 对于直接的任务，一个 "user" 角色通常就足够了
                    "content": prompt
                }
            ]
        },
        "parameters": {}
    }
    # --- 修正结束 ---

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        # 首先，尝试解析新格式的返回结果
        if result.get("output") and result["output"].get("choices"):
            return result["output"]["choices"][0]["message"]["content"].strip()
        # 如果新格式解析失败，尝试兼容旧格式（如qwen-max）
        elif result.get("output") and result["output"].get("text"):
            return result["output"]["text"].strip()
        # 如果两种格式都失败，则报告错误
        else:
            error_code = result.get("code", "N/A")
            error_message = result.get("message", "未知API错误")
            tqdm.write(f"API返回数据格式异常或错误: Code: {error_code}, Message: {error_message}")
            return f"[API返回格式错误: {error_message}]"

    except requests.exceptions.RequestException as e:
        tqdm.write(f"调用API时发生网络错误: {e}")
        return "[API 调用失败]"
def calculate_keyword_scores(sentence: str, test_item_words: Dict, grammar_book: List[Dict]) -> np.ndarray:
    """为所有语法规则计算关键词分数。"""
    input_words = set(sentence.lower().replace('.', '').replace(',', '').split())
    test_zhuang_keys = set(test_item_words.keys())
    scores = []
    for rule in grammar_book:
        score = 0
        description_words = set(rule.get("grammar_description", "").lower().split())
        score += len(input_words.intersection(description_words)) * 2
        for example in rule.get("examples", []):
            example_za_words = set(example.get("za", "").lower().split())
            score += len(input_words.intersection(example_za_words))
            rule_zhuang_keys = set(example.get("related_words", {}).keys())
            score += len(test_zhuang_keys.intersection(rule_zhuang_keys)) * 3
        scores.append(score)
    return np.array(scores, dtype=float)

def calculate_semantic_scores(queries: List[str], grammar_book: List[Dict], model_name: str) -> np.ndarray:
    """【V11修正】为所有语法规则计算语义分数，确保输入为列表。"""
    global GRAMMAR_BOOK_VECTORS
    from sklearn.metrics.pairwise import cosine_similarity
    
    model = load_embedding_model(model_name)
    
    if model_name not in GRAMMAR_BOOK_VECTORS:
        print(f">>> 正在使用 {model_name} 为语法书计算语义向量 (首次)...")
        rule_contents = [f"{rule.get('grammar_description', '')}\n{' '.join([ex.get('za', '') for ex in rule.get('examples', [])])}" for rule in grammar_book]
        GRAMMAR_BOOK_VECTORS[model_name] = model.encode(rule_contents, convert_to_tensor=True, show_progress_bar=True)
        print(">>> 语法书向量计算完成。")

    # 确保输入是一个列表，这样输出的向量总是2D的
    query_vectors = model.encode(queries, convert_to_tensor=True)
    # 返回的是一个 (len(queries), len(grammar_book)) 的相似度矩阵
    return cosine_similarity(query_vectors.cpu(), GRAMMAR_BOOK_VECTORS[model_name].cpu())

def find_best_rule(sentence: str, test_item_words: Dict, grammar_book: List[Dict], args: 'Args') -> Optional[Dict]:
    """【V11修正】调整对检索函数的调用和结果处理。"""
    if not grammar_book: return None

    # --- 关键词分数 ---
    keyword_scores = calculate_keyword_scores(sentence, test_item_words, grammar_book)
    
    # --- 语义分数 ---
    # 【修正】将单个句子放入列表中进行调用
    semantic_scores = calculate_semantic_scores([sentence], grammar_book, model_name=args.embedding_model_name)[0]

    # --- 分数融合 ---
    keyword_scores_norm = keyword_scores / (keyword_scores.max() + 1e-9)
    semantic_scores_norm = (semantic_scores + 1) / 2
    final_scores = (args.hybrid_alpha * keyword_scores_norm) + ((1 - args.hybrid_alpha) * semantic_scores_norm)
    top_k_indices = np.argsort(final_scores)[-args.rerank_top_k:][::-1]

    # --- 重排 (Re-rank) ---
    if args.use_reranker and len(top_k_indices) > 0:
        candidate_rules = [grammar_book[i] for i in top_k_indices]
        candidate_texts = [f"{rule.get('grammar_description', '')} {' '.join([ex.get('za', '') for ex in rule.get('examples', [])])}" for rule in candidate_rules]
        
        # 【修正】直接调用，返回一个(1, k)的矩阵，取第一行
        rerank_similarities = calculate_semantic_scores([sentence], candidate_rules, model_name=args.rerank_model_name)[0]
        
        best_candidate_index_in_top_k = rerank_similarities.argmax()
        best_rule_index = top_k_indices[best_candidate_index_in_top_k]
    elif len(top_k_indices) > 0:
        best_rule_index = top_k_indices[0]
    else:
        return None # 如果没有任何候选，直接返回
        
    return grammar_book[best_rule_index]

def find_best_examples(sentence: str, all_examples: List[Dict], args: 'Args') -> List[Dict]:
    """从所有例句中找到与输入最相似的k个作为few-shot示例。"""
    global ALL_EXAMPLES_VECTORS
    model_name = args.embedding_model_name
    model = load_embedding_model(model_name)

    if model_name not in ALL_EXAMPLES_VECTORS:
        print(">>> 正在为所有例句计算语义向量 (首次运行)...")
        example_texts = [ex['za'] for ex in all_examples]
        ALL_EXAMPLES_VECTORS[model_name] = model.encode(example_texts, convert_to_tensor=True, show_progress_bar=True)
        print(">>> 所有例句向量计算完成。")
        
    sentence_vector = model.encode([sentence], convert_to_tensor=True)
    similarities = cosine_similarity(sentence_vector.cpu(), ALL_EXAMPLES_VECTORS[model_name].cpu())[0]
    top_k_indices = np.argsort(similarities)[-args.num_few_shot:][::-1]

    return [all_examples[i] for i in top_k_indices]
def find_top_k_rules(sentence: str, test_item_words: Dict, grammar_book: List[Dict], args: 'Args') -> List[Dict]:
    """
    【Top-K升级】执行混合检索，并返回分数最高的 K 条规则。
    """
    if not grammar_book: return []

    keyword_scores = calculate_keyword_scores(sentence, test_item_words, grammar_book)
    semantic_scores = calculate_semantic_scores([sentence], grammar_book, model_name=args.embedding_model_name)[0]
    
    keyword_scores_norm = keyword_scores / (keyword_scores.max() + 1e-9)
    semantic_scores_norm = (semantic_scores + 1) / 2
    
    dynamic_alpha = 0.7 if len(sentence.split()) < 5 else 0.3
    final_scores = (dynamic_alpha * keyword_scores_norm) + ((1 - dynamic_alpha) * semantic_scores_norm)
    
    # 获取分数最高的 Top-K 个规则的索引
    # 注意：如果K大于规则总数，则返回所有规则
    k = min(args.num_retrieved_rules, len(grammar_book))
    top_k_indices = np.argsort(final_scores)[-k:][::-1]
    
    return [grammar_book[i] for i in top_k_indices]

def build_prompt(sentence: str, test_item_words: Dict, relevant_rule: Optional[Dict], few_shot_examples: List[Dict], args: 'Args') -> str:
    """
    【专家级Prompt构建器】
    能够动态整合词汇表、少样本示例、语法规则和思维链引导。
    """
    prompt = "你是一个专业的语言学家和翻译家，擅长将低资源语言\"壮语\"翻译成流畅、地道的\"中文\"。\n\n"
    
    # --- 1. 构建统一词汇表 ---
    vocabulary = {}
    vocabulary.update(test_item_words)
    if relevant_rule:
        for ex in relevant_rule.get("examples", []):
            vocabulary.update(ex.get("related_words", {}))
            
    if vocabulary:
        prompt += "--- 相关词汇表 ---\n"
        prompt += "以下是这个句子最可能涉及到的词汇及其翻译，请在翻译时优先参考和使用它们：\n"
        for word, meaning in vocabulary.items():
            prompt += f"  {word}: {meaning}\n"
        prompt += "\n"

    # --- 2. 动态添加少样本翻译示例 ---
    if few_shot_examples:
        prompt += "--- 翻译示例 ---\n"
        prompt += "这里有一些与待翻译句子最相似的范例，请模仿它们的风格和结构：\n"
        for ex in few_shot_examples:
            prompt += f"  壮语: {ex.get('za', '')}\n  中文: {ex.get('zh', '')}\n"
        prompt += "\n"

    # --- 3. 添加检索到的语法规则 ---
    if relevant_rule:
        prompt += "--- 相关语法规则 ---\n"
        prompt += f"为了更好地理解句子结构，请参考这条高度相关的语法规则：\n"
        prompt += f"语法描述: {relevant_rule.get('grammar_description', '无')}\n\n"
        prompt += "--- 背景知识结束 ---\n\n"

    # --- 4. 明确最终翻译任务，并可选地加入思维链引导 ---
    prompt += "--- 翻译任务 ---\n"
    if args.use_chain_of_thought:
        prompt += "请综合利用以上所有信息，遵循以下步骤完成任务：第一步，分析句子结构；第二步，结合词汇表和示例理解词义与风格；第三步，生成最终翻译。\n"
        prompt += "请只输出最终的中文翻译结果，不要包含任何分析过程。\n\n"
    else:
        prompt += "请综合利用以上提供的所有信息，将下面的壮语句子翻译成中文。\n"
        prompt += "请严格按照要求，只输出最终的中文翻译结果，不要包含任何额外的解释、标签或前缀。\n\n"
    
    prompt += f"待翻译的壮语句子: \"{sentence}\""
    return prompt

class Args(Tap):
    grammar_book_file: Path = Path('../../subtask2/data/grammar_book.json')  # 需要从Kaggle下载
    test_data_file: Path = Path('../../subtask2/data/test_data.json')
    output_dir: Path = Path('outputs/qwenlatest')

    use_retrieval: bool = True
    use_few_shot: bool = True
    use_self_correction: bool = True
    use_chain_of_thought: bool = True
    use_reranker: bool = False

    retrieval_mode: Literal["keyword", "semantic", "hybrid"] = "hybrid"
    hybrid_alpha: float = 0.5
    rerank_top_k: int = 5
    num_retrieved_rules: int = 3
    embedding_model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2' # 召回模型
    rerank_model_name: str = 'intfloat/e5-large-v2' # 更强的重排模型
    
    num_few_shot: int = 3
    
    def process_args(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

def main(args: Args):
    print(f"--- 开始运行壮语翻译脚本 (V10 - 专家级功能版) ---")
    
    try:
        grammar_book = json.loads(args.grammar_book_file.read_text(encoding='utf-8'))
        test_data = json.loads(args.test_data_file.read_text(encoding='utf-8'))
    except FileNotFoundError as e:
        print(f"\n错误: 找不到文件 {e.filename}。")
        sys.exit(1)
    print(f"成功加载 {len(grammar_book)} 条语法规则和 {len(test_data)} 条测试数据。")

    # 预加载所有例句，用于动态少样本检索
    all_examples_for_fewshot = [ex for rule in grammar_book for ex in rule.get('examples', [])]
    
    all_results = []
    for i, item in enumerate(tqdm(test_data, desc="翻译进度")):
        zhuang_sentence = item.get("za")
        test_item_words = item.get("related_words", {})
        if not zhuang_sentence: continue

        # 1. 检索背景知识
        relevant_rule = find_best_rule(zhuang_sentence, test_item_words, grammar_book, args) if args.use_retrieval else None
        few_shot_examples = find_best_examples(zhuang_sentence, all_examples_for_fewshot, args) if args.use_few_shot else []
        
        # 2. 构建用于"初翻"的Prompt
        initial_prompt = build_prompt(zhuang_sentence, test_item_words, relevant_rule, few_shot_examples, args)
        
        # 3. 【已修正】执行"初翻"，得到 initial_translation
        initial_translation = get_qwen_max_translation(initial_prompt)
        time.sleep(10)

        # 4. （可选）执行"自我修正"流程
        final_translation = initial_translation # 默认最终翻译等于初版翻译
        if args.use_self_correction:
            # 提取背景知识部分，用于构建修正Prompt
            context_for_correction = initial_prompt.split('--- 翻译任务 ---')[0]
            
            correction_prompt = f"""你是一位顶级的壮语翻译审校专家。这里有一个壮语句子、相关的背景知识，以及一个初步的翻译版本。
---
【壮语原文】: "{zhuang_sentence}"
{context_for_correction}
【初步翻译】: "{initial_translation}"
---
请你严格根据【背景知识】，审视【初步翻译】是否存在任何不准确或不通顺的地方。如果初步翻译是完美的，请直接重复一遍它。如果存在错误，请直接输出你修正后的最终版本。
你的任务是输出最完美的翻译，所以请只输出最终的中文句子，不要包含任何分析和解释。"""
            
            # 得到修正后的最终翻译
            final_translation = get_qwen_max_translation(correction_prompt)
            time.sleep(1)

        # 5. 在终端实时打印结果
        tqdm.write(f"\n{'='*20} 翻译进度 [{i + 1}/{len(test_data)}] {'='*20}")
        tqdm.write(f"  [壮语原文]: {zhuang_sentence}")
        tqdm.write(f"  [初版翻译]: {initial_translation}")
        if args.use_self_correction:
            tqdm.write(f"  [修正后翻译]: {final_translation}")
        tqdm.write("="*64 + "\n")

        # 6. 保存所有结果
        result_item = {
            "id": item.get("id"), 
            "source": zhuang_sentence, 
            "text": final_translation, # 保存最终结果用于评估
            "initial_translation": initial_translation, # 保存初版结果用于对比分析
            "prompt": initial_prompt, # 保存初翻的prompt用于提交
        }
        all_results.append(result_item)

    mode = f"RAG_{args.retrieval_mode}_alpha{args.hybrid_alpha}" if args.use_retrieval else "ZeroShot"
    output_filename = f"{args.test_data_file.stem}_results_{mode}.json"
    output_path = args.output_dir / output_filename
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"\n--- 翻译完成 ---")
    print(f"详细结果（含Prompt）已保存至: {output_path}")

    try:
        import pandas as pd
        submission_df = pd.DataFrame([{"id": r["id"], "translation": r["text"]} for r in all_results])
        submission_path = args.output_dir / f"submission_{mode}.csv"
        submission_df.to_csv(submission_path, index=False)
        print(f"Kaggle提交文件已生成: {submission_path}")
    except ImportError:
        print("\n提示: 未安装pandas库，无法自动生成Kaggle提交文件。")

if __name__ == "__main__":
    args = Args().parse_args()
    main(args)