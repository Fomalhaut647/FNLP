import re
import jieba
import jieba.posseg as pseg

class Tokenizer():
    def __init__(self):
        pass

    def tokenize(self, text, remove_punc=False):
        text = text.lower()    
        if remove_punc:
            # Remove Chinese punctuation marks
            for punc in "，。、；！？「」『』【】（）《》""…":
                text = text.replace(punc, " ")
            # Remove English punctuation marks
            for punc in ",.;?!":
                text = text.replace(punc, " ")
            # Remove numbers
            text = re.sub(r'\d+', '', text)
        else:
            for punc in "，。、；！？「」『』【】（）《》""…":
                text = text.replace(punc, " " + punc + " ")
            for punc in ",.;?!":
                text = text.replace(punc, " " + punc + " ")
        # Replace single quotes
        text = text.replace("'", "'").replace("'", "'")
        # Split by spaces
        tokenized_text = text.split(" ")
        # Remove empty strings
        tokenized_text = [word.strip() for word in tokenized_text if word.strip() != ""]
        return tokenized_text

class EngTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()


class KgvTokenizer(Tokenizer):
    def __init__(self):
        super().__init__() 

class ZaTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def tokenize(self, text, remove_punc=False):
        text = text.lower()    
        if remove_punc:
            # Remove Chinese punctuation marks
            for punc in "，。、；！？「」『』【】（）《》""…":
                text = text.replace(punc, " ")
            # Remove English punctuation marks
            for punc in ",.;?!":
                text = text.replace(punc, " ")
            # Remove numbers
            text = re.sub(r'\d+', '', text)
        else:
            for punc in "，。、；！？「」『』【】（）《》""…":
                text = text.replace(punc, " " + punc + " ")
            # Remove English punctuation marks
            for punc in ",.;?!":
                text = text.replace(punc, " " + punc + " ")
        # Replace single quotes
        text = text.replace("'", "'").replace("'", "'")
        # Split by spaces
        tokenized_text = text.split(" ")
        # Remove empty strings
        tokenized_text = [word.strip() for word in tokenized_text if word.strip() != ""]
        return tokenized_text


class ZhTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def tokenize(self, text, remove_punc=False, do_cut_all=False, cut_for_search=False):
        # Use jieba for Chinese word segmentation
        text = text.lower()
        if remove_punc:
            # Remove Chinese punctuation marks
            for punc in "，。、；！？「」『』【】（）《》""…":
                text = text.replace(punc, "")
            # Remove numbers
            text = re.sub(r'\d+', '', text)
        
        if cut_for_search:
            tokenized_text = jieba.lcut_for_search(text)
        else:
            tokenized_text = jieba.lcut(text, cut_all=do_cut_all)
        tokenized_text = [word.strip() for word in tokenized_text if word.strip() != ""]
        return tokenized_text
    
