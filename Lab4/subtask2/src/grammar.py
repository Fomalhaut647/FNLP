import json
# 1. Import required libraries
from sentence_transformers import SentenceTransformer, util
import torch
from tokenizer import ZaTokenizer


class GrammarBook:
    """
    This class is used to load and query the grammar book (grammar_book.json).
    This advanced version uses semantic search based on sentence embeddings to find the most relevant rules.
    """

    def __init__(self, grammar_book_path: str):
        """
        Initialize by loading the grammar book and loading sentence embedding model from local path,
        creating vector index for all rule descriptions.
        """
        self.rules = []
        self.rule_embeddings = None

        # --- Core modification point ---
        # No longer using network names, but using a local folder path.
        # Please ensure you have placed the downloaded model folder at this path.
        local_model_path = './downloaded_model/'  # Assume model folder is in project root

        print("Loading sentence transformer model from local path: {}...".format(local_model_path))
        try:
            self.model = SentenceTransformer(local_model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print("Error loading model from local path: {}".format(e))
            print(
                "Please ensure you have downloaded the model and placed it in the correct directory, e.g., './downloaded_model/'.")
            # If model loading fails, gracefully exit or disable related functionality
            self.model = None

        print("Loading grammar book...")
        try:
            with open(grammar_book_path, 'r', encoding='utf-8') as f:
                self.rules = json.load(f)
            print("Grammar book loaded successfully.")
            # Only build index if model is successfully loaded
            if self.model:
                self._build_semantic_index()
        except FileNotFoundError:
            print("Error: Grammar book not found at {}".format(grammar_book_path))
        except json.JSONDecodeError:
            print("Error: Could not decode JSON from {}".format(grammar_book_path))

    def _build_semantic_index(self):
        """
        Create and cache semantic vectors for all rules' 'grammar_description'.
        """
        print("Building semantic index for grammar rules...")
        if not self.rules:
            print("Warning: No rules found in grammar book.")
            return

        corpus = [rule.get('grammar_description', '') for rule in self.rules]

        # Ensure we only create vectors for non-empty descriptions
        self.valid_indices = [i for i, desc in enumerate(corpus) if desc]
        valid_corpus = [corpus[i] for i in self.valid_indices]

        if not valid_corpus:
            print("Warning: No valid grammar descriptions found. Semantic search will be disabled.")
            return

        # Compute vectors for all valid descriptions and move to GPU (if available) for acceleration
        self.rule_embeddings = self.model.encode(valid_corpus, convert_to_tensor=True, show_progress_bar=True)
        if torch.cuda.is_available():
            self.rule_embeddings = self.rule_embeddings.to('cuda')
        print("Semantic index built successfully.")

    def search_rules_by_semantic_similarity(self, sentence: str, top_k: int = 1) -> list:
        """
        [New Feature] Use semantic search to find the most relevant top_k rules.
        """
        if self.rule_embeddings is None:
            return []

        # 1. Create vector for query sentence
        query_embedding = self.model.encode(sentence, convert_to_tensor=True)
        if torch.cuda.is_available():
            query_embedding = query_embedding.to('cuda')

        # 2. Calculate cosine similarity
        # This function quickly calculates similarity between query vector and all rule vectors
        cos_scores = util.cos_sim(query_embedding, self.rule_embeddings)[0]

        # 3. Find top_k results with highest scores
        top_results = torch.topk(cos_scores, k=min(top_k, len(cos_scores)))

        found_rules = []
        for score, idx in zip(top_results[0], top_results[1]):
            # Can set a threshold, e.g., 0.5, to avoid returning completely irrelevant results
            if score.item() > 0.5:
                original_index = self.valid_indices[idx.item()]
                rule = self.rules[original_index]

                rule_str = "Grammar description: {}\n".format(rule.get('grammar_description', 'N/A'))

                examples = rule.get('examples', [])
                if examples:
                    rule_str += "Examples:\n"
                    for ex in examples[:1]:  # Only show one most relevant example
                        za_sent = ex.get('za', '')
                        zh_sent = ex.get('zh', '')
                        rule_str += "  - Zhuang: {}\n    Chinese: {}\n".format(za_sent, zh_sent)

                found_rules.append(rule_str)

        return found_rules