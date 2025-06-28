from corpus import lang2tokenizer
import random
import json

model_to_chat_template = {
    'qwen': "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
}


# --- 函数签名已修改：增加了 grammar_book 参数 ---
def construct_prompt_za2zh(src_sent, dictionary, parallel_corpus, grammar_book, args):
    """
    构建从壮语到汉语的翻译提示，现在使用 BM25 算法检索语法规则。
    """
    # 1. 检索相似例句 (逻辑不变)
    if args.num_parallel_sent > 0:
        top_k_sentences_with_scores = parallel_corpus.search_by_bm25(src_sent, query_lang=args.src_lang,
                                                                     top_k=args.num_parallel_sent)
    else:
        top_k_sentences_with_scores = []

    # 2. 调用新的 BM25 语法检索方法
    related_rules = grammar_book.search_rules_by_semantic_similarity(src_sent, top_k=2)  # 默认检索最相关的2条规则

    def get_word_explanation_prompt(text):
        prompt = "## 在上面的句子中，"
        tokens = lang2tokenizer[args.src_lang].tokenize(text, remove_punc=True)
        for word in tokens:
            exact_match_meanings = dictionary.get_meanings_by_exact_match(word, max_num_meanings=2)
            if exact_match_meanings is not None:
                concated_meaning = "”或“".join(exact_match_meanings)
                concated_meaning = "“" + concated_meaning + "”"
                prompt += "壮语词语“{}”在汉语中可能的翻译是{}；\n".format(word, concated_meaning)
            else:
                fuzzy_match_meanings = dictionary.get_meanings_by_fuzzy_match(word, top_k=2,
                                                                              max_num_meanings_per_word=2)
                for item in fuzzy_match_meanings[:2]:
                    concated_meaning = "”或“".join(item["meanings"])
                    concated_meaning = "“" + concated_meaning + "”"
                    prompt += "壮语词语“{}”在汉语中可能的翻译是{}；\n".format(item['word'], concated_meaning)
        return prompt

    prompt = ""

    # 3. 组装提示 (逻辑不变)
    if args.num_parallel_sent > 0:
        prompt += "# 请仿照样例，参考给出的词汇和语法，将壮语句子翻译成汉语。\n\n"
        for i in range(len(top_k_sentences_with_scores)):
            item = top_k_sentences_with_scores[i]["pair"]
            prompt += "## 请将下面的壮语句子翻译成汉语：{}\n".format(item[args.src_lang])
            prompt += get_word_explanation_prompt(item[args.src_lang])
            prompt += "## 所以，该壮语句子完整的汉语翻译是：{}\n\n".format(item['zh'])

    prompt += "## 请将下面的壮语句子翻译成汉语：{}\n".format(src_sent)
    prompt += get_word_explanation_prompt(src_sent)

    if related_rules:
        prompt += "## 相关语法规则：\n"
        for rule in related_rules:
            prompt += rule + "\n"

    prompt += "## 所以，该壮语句子完整的汉语翻译是："
    return prompt






if __name__ == '__main__':
    pass

    



