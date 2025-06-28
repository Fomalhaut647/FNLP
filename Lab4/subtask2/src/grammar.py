import json
# 1. 导入新需要的库
from sentence_transformers import SentenceTransformer, util
import torch
from tokenizer import ZaTokenizer


class GrammarBook:
    """
    此类用于加载和查询语法书 (grammar_book.json)。
    这个最先进的版本使用基于句子嵌入的语义搜索来寻找最相关的规则。
    """

    def __init__(self, grammar_book_path: str):
        """
        初始化时加载语法书，并从本地路径加载句子嵌入模型，为所有规则描述创建向量索引。
        """
        self.rules = []
        self.rule_embeddings = None

        # --- 这里是核心修改点 ---
        # 不再使用网络名称，而是使用一个本地文件夹路径。
        # 请确保您已经将下载好的模型文件夹放在了这个路径。
        local_model_path = './downloaded_model/'  # 假设模型文件夹就在项目根目录

        print("Loading sentence transformer model from local path: {}...".format(local_model_path))
        try:
            self.model = SentenceTransformer(local_model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print("Error loading model from local path: {}".format(e))
            print(
                "Please ensure you have downloaded the model and placed it in the correct directory, e.g., './downloaded_model/'.")
            # 如果模型加载失败，则优雅地退出或禁用相关功能
            self.model = None

        print("Loading grammar book...")
        try:
            with open(grammar_book_path, 'r', encoding='utf-8') as f:
                self.rules = json.load(f)
            print("Grammar book loaded successfully.")
            # 仅在模型成功加载后才构建索引
            if self.model:
                self._build_semantic_index()
        except FileNotFoundError:
            print("Error: Grammar book not found at {}".format(grammar_book_path))
        except json.JSONDecodeError:
            print("Error: Could not decode JSON from {}".format(grammar_book_path))

    def _build_semantic_index(self):
        """
        为所有规则的 'grammar_description' 创建并缓存语义向量。
        """
        print("Building semantic index for grammar rules...")
        if not self.rules:
            print("Warning: No rules found in grammar book.")
            return

        corpus = [rule.get('grammar_description', '') for rule in self.rules]

        # 确保我们只为非空描述创建向量
        self.valid_indices = [i for i, desc in enumerate(corpus) if desc]
        valid_corpus = [corpus[i] for i in self.valid_indices]

        if not valid_corpus:
            print("Warning: No valid grammar descriptions found. Semantic search will be disabled.")
            return

        # 计算所有有效描述的向量，并将其移动到GPU（如果可用）以加速
        self.rule_embeddings = self.model.encode(valid_corpus, convert_to_tensor=True, show_progress_bar=True)
        if torch.cuda.is_available():
            self.rule_embeddings = self.rule_embeddings.to('cuda')
        print("Semantic index built successfully.")

    def search_rules_by_semantic_similarity(self, sentence: str, top_k: int = 1) -> list:
        """
        【新功能】使用语义搜索查找最相关的 top_k 条规则。
        """
        if self.rule_embeddings is None:
            return []

        # 1. 为查询句子创建向量
        query_embedding = self.model.encode(sentence, convert_to_tensor=True)
        if torch.cuda.is_available():
            query_embedding = query_embedding.to('cuda')

        # 2. 计算余弦相似度
        # 这个函数会快速计算查询向量与所有规则向量之间的相似度
        cos_scores = util.cos_sim(query_embedding, self.rule_embeddings)[0]

        # 3. 找到得分最高的 top_k 个结果
        top_results = torch.topk(cos_scores, k=min(top_k, len(cos_scores)))

        found_rules = []
        for score, idx in zip(top_results[0], top_results[1]):
            # 可以设置一个阈值，比如0.5，避免返回完全不相关的结果
            if score.item() > 0.5:
                original_index = self.valid_indices[idx.item()]
                rule = self.rules[original_index]

                rule_str = "语法说明: {}\n".format(rule.get('grammar_description', 'N/A'))

                examples = rule.get('examples', [])
                if examples:
                    rule_str += "示例:\n"
                    for ex in examples[:1]:  # 只展示一个最相关的例子
                        za_sent = ex.get('za', '')
                        zh_sent = ex.get('zh', '')
                        rule_str += "  - 壮语: {}\n    汉语: {}\n".format(za_sent, zh_sent)

                found_rules.append(rule_str)

        return found_rules