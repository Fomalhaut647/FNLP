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

# Global variables for caching models and vectors to avoid reloading and improve efficiency
EMBEDDING_MODEL = None
EMBEDDING_RERANK_MODEL = None
GRAMMAR_BOOK_VECTORS = {} # Cache vectors for different models
ALL_EXAMPLES_VECTORS = {}   # Cache vectors for different models

# ----------------------------------------------------------------------------
# Core functionality section
# ----------------------------------------------------------------------------

def load_embedding_model(model_name: str) -> 'SentenceTransformer':
    """Load and cache the specified semantic embedding model."""
    global EMBEDDING_MODEL, EMBEDDING_RERANK_MODEL
    
    # Decide which global variable to use for caching
    if 'rerank' in model_name or 'e5' in model_name or 'mpnet' in model_name:
        model_cache = 'EMBEDDING_RERANK_MODEL'
    else:
        model_cache = 'EMBEDDING_MODEL'

    if globals()[model_cache] is None:
        try:
            from sentence_transformers import SentenceTransformer
            print(f">>> Loading semantic model: {model_name} (First run requires download, please be patient)...")
            globals()[model_cache] = SentenceTransformer(model_name)
            print(f">>> {model_name} loaded successfully.")
        except ImportError:
            print("\nError: Required libraries not found. Please run: pip install sentence-transformers torch scikit-learn")
            sys.exit(1)
    return globals()[model_cache]

def get_qwen_max_translation(prompt: str) -> str:
  
    api_key = os.getenv("API_KEY") # <--- Please replace with your actual, valid API Key
    if "sk-" not in api_key:
        raise ValueError("API Key format is incorrect or not set, please write it in the get_qwen_max_translation function.")

    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": "qwen-max", # This is an example, please replace as needed, e.g., "deepseek-moe-16b-chat"
        "input": {
            "messages": [
                {
                    "role": "user", # For direct tasks, a single "user" role is usually sufficient
                    "content": prompt
                }
            ]
        },
        "parameters": {}
    }
    # --- Fix ends ---

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        # First, try to parse the new format response
        if result.get("output") and result["output"].get("choices"):
            return result["output"]["choices"][0]["message"]["content"].strip()
        # If new format parsing fails, try to be compatible with old format (like qwen-max)
        elif result.get("output") and result["output"].get("text"):
            return result["output"]["text"].strip()
        # If both formats fail, report error
        else:
            error_code = result.get("code", "N/A")
            error_message = result.get("message", "Unknown API error")
            tqdm.write(f"API returned abnormal data format or error: Code: {error_code}, Message: {error_message}")
            return f"[API format error: {error_message}]"

    except requests.exceptions.RequestException as e:
        tqdm.write(f"Network error occurred while calling API: {e}")
        return "[API call failed]"

def calculate_keyword_scores(sentence: str, test_item_words: Dict, grammar_book: List[Dict]) -> np.ndarray:
    """Calculate keyword scores for all grammar rules."""
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
    """[V11 Fix] Calculate semantic scores for all grammar rules, ensuring input is a list."""
    global GRAMMAR_BOOK_VECTORS
    from sklearn.metrics.pairwise import cosine_similarity
    
    model = load_embedding_model(model_name)
    
    if model_name not in GRAMMAR_BOOK_VECTORS:
        print(f">>> Using {model_name} to calculate semantic vectors for grammar book (first time)...")
        rule_contents = [f"{rule.get('grammar_description', '')}\n{' '.join([ex.get('za', '') for ex in rule.get('examples', [])])}" for rule in grammar_book]
        GRAMMAR_BOOK_VECTORS[model_name] = model.encode(rule_contents, convert_to_tensor=True, show_progress_bar=True)
        print(">>> Grammar book vector calculation completed.")

    # Ensure input is a list so output vectors are always 2D
    query_vectors = model.encode(queries, convert_to_tensor=True)
    # Returns a (len(queries), len(grammar_book)) similarity matrix
    return cosine_similarity(query_vectors.cpu(), GRAMMAR_BOOK_VECTORS[model_name].cpu())

def find_best_rule(sentence: str, test_item_words: Dict, grammar_book: List[Dict], args: 'Args') -> Optional[Dict]:
    """[V11 Fix] Adjust calls to retrieval functions and result handling."""
    if not grammar_book: return None

    # --- Keyword scores ---
    keyword_scores = calculate_keyword_scores(sentence, test_item_words, grammar_book)
    
    # --- Semantic scores ---
    # [Fix] Put single sentence into a list for calling
    semantic_scores = calculate_semantic_scores([sentence], grammar_book, model_name=args.embedding_model_name)[0]

    # --- Score fusion ---
    keyword_scores_norm = keyword_scores / (keyword_scores.max() + 1e-9)
    semantic_scores_norm = (semantic_scores + 1) / 2
    final_scores = (args.hybrid_alpha * keyword_scores_norm) + ((1 - args.hybrid_alpha) * semantic_scores_norm)
    top_k_indices = np.argsort(final_scores)[-args.rerank_top_k:][::-1]

    # --- Re-ranking ---
    if args.use_reranker and len(top_k_indices) > 0:
        candidate_rules = [grammar_book[i] for i in top_k_indices]
        candidate_texts = [f"{rule.get('grammar_description', '')} {' '.join([ex.get('za', '') for ex in rule.get('examples', [])])}" for rule in candidate_rules]
        
        # [Fix] Direct call, returns a (1, k) matrix, take the first row
        rerank_similarities = calculate_semantic_scores([sentence], candidate_rules, model_name=args.rerank_model_name)[0]
        
        best_candidate_index_in_top_k = rerank_similarities.argmax()
        best_rule_index = top_k_indices[best_candidate_index_in_top_k]
    elif len(top_k_indices) > 0:
        best_rule_index = top_k_indices[0]
    else:
        return None # If no candidates, return directly
        
    return grammar_book[best_rule_index]

def find_best_examples(sentence: str, all_examples: List[Dict], args: 'Args') -> List[Dict]:
    """Find the k most similar examples from all examples as few-shot examples."""
    global ALL_EXAMPLES_VECTORS
    model_name = args.embedding_model_name
    model = load_embedding_model(model_name)

    if model_name not in ALL_EXAMPLES_VECTORS:
        print(">>> Calculating semantic vectors for all examples (first run)...")
        example_texts = [ex['za'] for ex in all_examples]
        ALL_EXAMPLES_VECTORS[model_name] = model.encode(example_texts, convert_to_tensor=True, show_progress_bar=True)
        print(">>> All example vectors calculation completed.")
        
    sentence_vector = model.encode([sentence], convert_to_tensor=True)
    similarities = cosine_similarity(sentence_vector.cpu(), ALL_EXAMPLES_VECTORS[model_name].cpu())[0]
    top_k_indices = np.argsort(similarities)[-args.num_few_shot:][::-1]

    return [all_examples[i] for i in top_k_indices]

def find_top_k_rules(sentence: str, test_item_words: Dict, grammar_book: List[Dict], args: 'Args') -> List[Dict]:
    """
    [Top-K upgrade] Perform hybrid retrieval and return the top K rules with highest scores.
    """
    if not grammar_book: return []

    keyword_scores = calculate_keyword_scores(sentence, test_item_words, grammar_book)
    semantic_scores = calculate_semantic_scores([sentence], grammar_book, model_name=args.embedding_model_name)[0]
    
    keyword_scores_norm = keyword_scores / (keyword_scores.max() + 1e-9)
    semantic_scores_norm = (semantic_scores + 1) / 2
    
    dynamic_alpha = 0.7 if len(sentence.split()) < 5 else 0.3
    final_scores = (dynamic_alpha * keyword_scores_norm) + ((1 - dynamic_alpha) * semantic_scores_norm)
    
    # Get indices of the top K rules with highest scores
    # Note: If K is greater than total rules, return all rules
    k = min(args.num_retrieved_rules, len(grammar_book))
    top_k_indices = np.argsort(final_scores)[-k:][::-1]
    
    return [grammar_book[i] for i in top_k_indices]

def build_prompt(sentence: str, test_item_words: Dict, relevant_rule: Optional[Dict], few_shot_examples: List[Dict], args: 'Args') -> str:
    """
    [Expert-level Prompt Builder]
    Can dynamically integrate vocabulary, few-shot examples, grammar rules, and chain-of-thought guidance.
    """
    prompt = "You are a professional linguist and translator, expert at translating the low-resource language \"Zhuang\" into fluent, natural \"Chinese\".\n\n"
    
    # --- 1. Build unified vocabulary ---
    vocabulary = {}
    vocabulary.update(test_item_words)
    if relevant_rule:
        for ex in relevant_rule.get("examples", []):
            vocabulary.update(ex.get("related_words", {}))
            
    if vocabulary:
        prompt += "--- Related Vocabulary ---\n"
        prompt += "The following are the most likely words involved in this sentence and their translations. Please refer to and use them when translating:\n"
        for word, meaning in vocabulary.items():
            prompt += f"  {word}: {meaning}\n"
        prompt += "\n"

    # --- 2. Dynamically add few-shot translation examples ---
    if few_shot_examples:
        prompt += "--- Translation Examples ---\n"
        prompt += "Here are some examples most similar to the sentence to be translated. Please imitate their style and structure:\n"
        for ex in few_shot_examples:
            prompt += f"  Zhuang: {ex.get('za', '')}\n  Chinese: {ex.get('zh', '')}\n"
        prompt += "\n"

    # --- 3. Add retrieved grammar rules ---
    if relevant_rule:
        prompt += "--- Related Grammar Rules ---\n"
        prompt += f"To better understand the sentence structure, please refer to this highly relevant grammar rule:\n"
        prompt += f"Grammar description: {relevant_rule.get('grammar_description', 'None')}\n\n"
        prompt += "--- Background knowledge ends ---\n\n"

    # --- 4. Clear final translation task with optional chain-of-thought guidance ---
    prompt += "--- Translation Task ---\n"
    if args.use_chain_of_thought:
        prompt += "Please use all the above information comprehensively and follow these steps to complete the task: Step 1, analyze sentence structure; Step 2, combine vocabulary and examples to understand word meanings and style; Step 3, generate the final translation.\n"
        prompt += "Please only output the final Chinese translation result without including any analysis process.\n\n"
    else:
        prompt += "Please use all the information provided above comprehensively to translate the following Zhuang sentence into Chinese.\n"
        prompt += "Please strictly follow the requirements and only output the final Chinese translation result without any additional explanations, tags, or prefixes.\n\n"
    
    prompt += f"Zhuang sentence to be translated: \"{sentence}\""
    return prompt

class Args(Tap):
    grammar_book_file: Path = Path('./data/grammar_book.json')
    test_data_file: Path = Path('./data/test_data.json')
    output_dir: Path = Path('./outputs')

    use_retrieval: bool = True
    use_few_shot: bool = True
    use_self_correction: bool = True
    use_chain_of_thought: bool = True
    use_reranker: bool = False

    retrieval_mode: Literal["keyword", "semantic", "hybrid"] = "hybrid"
    hybrid_alpha: float = 0.5
    rerank_top_k: int = 5
    num_retrieved_rules: int = 3
    embedding_model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2' # Retrieval model
    rerank_model_name: str = 'intfloat/e5-large-v2' # Stronger reranking model
    
    num_few_shot: int = 3
    
    def process_args(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

def main(args: Args):
    print(f"--- Starting Zhuang translation script (V10 - Expert feature version) ---")
    
    try:
        grammar_book = json.loads(args.grammar_book_file.read_text(encoding='utf-8'))
        test_data = json.loads(args.test_data_file.read_text(encoding='utf-8'))
    except FileNotFoundError as e:
        print(f"\nError: File not found {e.filename}.")
        sys.exit(1)
    print(f"Successfully loaded {len(grammar_book)} grammar rules and {len(test_data)} test data.")

    # Preload all examples for dynamic few-shot retrieval
    all_examples_for_fewshot = [ex for rule in grammar_book for ex in rule.get('examples', [])]
    
    all_results = []
    for i, item in enumerate(tqdm(test_data, desc="Translation progress")):
        zhuang_sentence = item.get("za")
        test_item_words = item.get("related_words", {})
        if not zhuang_sentence: continue

        # 1. Retrieve background knowledge
        relevant_rule = find_best_rule(zhuang_sentence, test_item_words, grammar_book, args) if args.use_retrieval else None
        few_shot_examples = find_best_examples(zhuang_sentence, all_examples_for_fewshot, args) if args.use_few_shot else []
        
        # 2. Build prompt for "initial translation"
        initial_prompt = build_prompt(zhuang_sentence, test_item_words, relevant_rule, few_shot_examples, args)
        
        # 3. [Fixed] Execute "initial translation" to get initial_translation
        initial_translation = get_qwen_max_translation(initial_prompt)
        time.sleep(10)

        # 4. (Optional) Execute "self-correction" process
        final_translation = initial_translation # Default final translation equals initial translation
        if args.use_self_correction:
            # Extract background knowledge part for building correction prompt
            context_for_correction = initial_prompt.split('--- Translation Task ---')[0]
            
            correction_prompt = f"""You are a top-level Zhuang translation proofreading expert. Here is a Zhuang sentence, related background knowledge, and a preliminary translation version.
---
【Zhuang Original】: "{zhuang_sentence}"
{context_for_correction}
【Preliminary Translation】: "{initial_translation}"
---
Please strictly based on the 【Background Knowledge】, review whether the 【Preliminary Translation】 has any inaccuracies or awkwardness. If the preliminary translation is perfect, please repeat it directly. If there are errors, please directly output your corrected final version.
Your task is to output the most perfect translation, so please only output the final Chinese sentence without any analysis and explanation."""
            
            # Get the corrected final translation
            final_translation = get_qwen_max_translation(correction_prompt)
            time.sleep(1)

        # 5. Print results in real-time in terminal
        tqdm.write(f"\n{'='*20} Translation progress [{i + 1}/{len(test_data)}] {'='*20}")
        tqdm.write(f"  [Zhuang Original]: {zhuang_sentence}")
        tqdm.write(f"  [Initial Translation]: {initial_translation}")
        if args.use_self_correction:
            tqdm.write(f"  [Corrected Translation]: {final_translation}")
        tqdm.write("="*64 + "\n")

        # 6. Save all results
        result_item = {
            "id": item.get("id"), 
            "source": zhuang_sentence, 
            "text": final_translation, # Save final result for evaluation
            "initial_translation": initial_translation, # Save initial result for comparison analysis
            "prompt": initial_prompt, # Save initial translation prompt for submission
        }
        all_results.append(result_item)

    mode = f"RAG_{args.retrieval_mode}_alpha{args.hybrid_alpha}" if args.use_retrieval else "ZeroShot"
    output_filename = f"{args.test_data_file.stem}_results_{mode}.json"
    output_path = args.output_dir / output_filename
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"\n--- Translation completed ---")
    print(f"Detailed results (including prompts) saved to: {output_path}")

    try:
        import pandas as pd
        submission_df = pd.DataFrame([{"id": r["id"], "translation": r["text"]} for r in all_results])
        submission_path = args.output_dir / f"submission_{mode}.csv"
        submission_df.to_csv(submission_path, index=False)
        print(f"Kaggle submission file generated: {submission_path}")
    except ImportError:
        print("\nNote: pandas library not installed, cannot automatically generate Kaggle submission file.")

if __name__ == "__main__":
    args = Args().parse_args()
    main(args)