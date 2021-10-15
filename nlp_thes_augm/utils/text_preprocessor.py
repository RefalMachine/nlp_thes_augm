import pymorphy2
from functools import lru_cache
from nlp_thes_augm.utils.common import Token
import re

class TextPreprocessor:
    def __init__(self):
        self.sep_reg = re.compile(r"[\w]+|[!\"#$%&\'()*+,-–—./:;<=>?@\[\\\]^_`{|}~“”«»]")
        self.morph_analyzer = pymorphy2.MorphAnalyzer()
    
    @lru_cache(maxsize=1000000)
    def get_normal_form(self, word):
        return self.morph_analyzer.parse(word)[0].normal_form
    
    def tokenize_text(self, text):
        text_splitted = re.findall(self.sep_reg, text)
        shift = 0
        result = []
        for i, w in enumerate(text_splitted):
            w_norm = self._normalize_word(w)
            if text[shift] == ' ':
                shift += 1
            w_shift = shift
            shift = shift + len(w)
            result.append(Token(w, w_norm, i, w_shift))

        return result
    
    def _normalize_word(self, word):
        if (not word.isalpha()) or (len(word) < 3) or word.isupper():
            return word

        return self.get_normal_form(word).lower()