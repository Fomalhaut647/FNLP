from corpus import lang2tokenizer
import random
import json

model_to_chat_template = {
    'qwen': "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
}


# --- Function signature modified: added grammar_book parameter ---
def construct_prompt_za2zh(src_sent, dictionary, parallel_corpus, grammar_book, args):
    """
    Construct translation prompt from Zhuang to Chinese, now using BM25 algorithm to retrieve grammar rules.
    """
    # 1. Retrieve similar examples (logic unchanged)
    if args.num_parallel_sent > 0:
        top_k_sentences_with_scores = parallel_corpus.search_by_bm25(src_sent, query_lang=args.src_lang,
                                                                     top_k=args.num_parallel_sent)
    else:
        top_k_sentences_with_scores = []

    # 2. Call new BM25 grammar retrieval method
    related_rules = grammar_book.search_rules_by_semantic_similarity(src_sent, top_k=2)  # Default retrieve 2 most relevant rules

    def get_word_explanation_prompt(text):
        prompt = "## In the above sentence, "
        tokens = lang2tokenizer[args.src_lang].tokenize(text, remove_punc=True)
        for word in tokens:
            exact_match_meanings = dictionary.get_meanings_by_exact_match(word, max_num_meanings=2)
            if exact_match_meanings is not None:
                combined_meaning = "" or "".join(exact_match_meanings)
                combined_meaning = """ + combined_meaning + """
                prompt += "the Zhuang word \"{}\" may translate to {} in Chinese;\n".format(word, combined_meaning)
            else:
                fuzzy_match_meanings = dictionary.get_meanings_by_fuzzy_match(word, top_k=2,
                                                                              max_num_meanings_per_word=2)
                for item in fuzzy_match_meanings[:2]:
                    combined_meaning = "" or "".join(item["meanings"])
                    combined_meaning = """ + combined_meaning + """
                    prompt += "the Zhuang word \"{}\" may translate to {} in Chinese;\n".format(item['word'], combined_meaning)
        return prompt

    prompt = ""

    # 3. Assemble prompt (logic unchanged)
    if args.num_parallel_sent > 0:
        prompt += "# Please follow the examples, refer to the given vocabulary and grammar, translate the Zhuang sentence into Chinese.\n\n"
        for i in range(len(top_k_sentences_with_scores)):
            item = top_k_sentences_with_scores[i]["pair"]
            prompt += "## Please translate the following Zhuang sentence into Chinese: {}\n".format(item[args.src_lang])
            prompt += get_word_explanation_prompt(item[args.src_lang])
            prompt += "## Therefore, the complete Chinese translation of this Zhuang sentence is: {}\n\n".format(item['zh'])

    prompt += "## Please translate the following Zhuang sentence into Chinese: {}\n".format(src_sent)
    prompt += get_word_explanation_prompt(src_sent)

    if related_rules:
        prompt += "## Relevant grammar rules:\n"
        for rule in related_rules:
            prompt += rule + "\n"

    prompt += "## Therefore, the complete Chinese translation of this Zhuang sentence is:"
    return prompt






if __name__ == '__main__':
    pass

    



