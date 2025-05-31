from transformers import AutoTokenizer
from collections import defaultdict


def wordpiece(training_corpus, vocab_size):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    word_freqs = defaultdict(int)
    for text in training_corpus:
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        new_words = [word for word, offset in words_with_offsets]
        for word in new_words:
            word_freqs[word] += 1

    alphabet = []
    for word in word_freqs.keys():
        if word[0] not in alphabet:
            alphabet.append(word[0])
        for letter in word[1:]:
            if f"##{letter}" not in alphabet:
                alphabet.append(f"##{letter}")

    alphabet.sort()

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()

    # Do NOT add your above this line.
    #======
    
    # Add your code here.
    # 初始化单词分割：第一个字符不带 ##，其余带 ##
    splits = {word: [word[0]] + [f"##{c}" for c in word[1:]] 
              for word in word_freqs}
    
    # 迭代合并 token 对直到达到目标词汇表大小
    while len(vocab) < vocab_size:
        # 计算 token 对和 token 的频率
        pairs = defaultdict(int)
        tokens = defaultdict(int)
        
        for word, freq in word_freqs.items():
            word_split = splits[word]
            # 计算 token 频率
            for token in word_split:
                tokens[token] += freq
            # 计算 token 对频率
            for i in range(len(word_split) - 1):
                pairs[(word_split[i], word_split[i + 1])] += freq
        
        if not pairs:
            break
        
        # 根据得分查找最佳 token 对：freq(A,B) / (freq(A) * freq(B))
        best = max(pairs, key=lambda p: pairs[p] / (tokens[p[0]] * tokens[p[1]]))
        
        # 创建合并后的 token
        merged = best[0] + (best[1][2:] if best[1].startswith("##") else best[1])
        vocab.append(merged)
        
        # 更新分割：用合并后的 token 替换最佳 token 对
        for word in splits:
            old_split = splits[word]
            new_split = []
            i = 0
            while i < len(old_split):
                if (i < len(old_split) - 1 and 
                    (old_split[i], old_split[i + 1]) == best):
                    new_split.append(merged)
                    i += 2
                else:
                    new_split.append(old_split[i])
                    i += 1
            splits[word] = new_split

    #======
    # Do NOT add your below this line.

    return vocab

if __name__ == "__main__":
    default_training_corpus = [
        "peking university is located in haidian district",
        "computer science is the flagship major of peking university",
        "the school of electronic engineering and computer science enrolls approximately five hundred new students each year"  
    ]

    default_vocab_size = 120

    my_vocab = wordpiece(default_training_corpus, default_vocab_size)

    print('The vocab:', my_vocab)

    def encode_word(custom_vocab, word):
        tokens = []
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in custom_vocab:
                i -= 1
            if i == 0:
                return ["[UNK]"]
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0:
                word = f"##{word}"
        return tokens

    def tokenize(custom_vocab, text):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
        pre_tokenized_text = [word for word, offset in pre_tokenize_result]
        encoded_words = [encode_word(custom_vocab, word) for word in pre_tokenized_text]
        return sum(encoded_words, [])

    print('Tokenization result:', tokenize(my_vocab, 'nous etudions a l universite de pekin'))
