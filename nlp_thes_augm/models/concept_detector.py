from nlp_thes_augm.utils.common import ConceptToken

class ConceptDetector:
    def __init__(self, wordnet, text_preprocessor):
        self.wordnet = wordnet
        self.senses = {}
        self.issensecontinuation = set()
        self.text_preprocessor = text_preprocessor
        self._transform_senses()

    def _transform_senses(self):
        for sense in self.wordnet.senses:
            splited_sense = sense.split('_')
            sense_len = len(splited_sense)
            for i in range(1, sense_len):
                self.issensecontinuation.add('_'.join(splited_sense[:i]))

            if sense_len not in self.senses:
                self.senses[sense_len] = set()

            self.senses[sense_len].add(sense)
    
    @staticmethod
    def _filter_tokens(tokens):
        return [token for token in tokens if token.token.isalpha() and not token.token.isupper()]
    
    @staticmethod
    def _sanity_check(tokens, text):
        for token in tokens:
            if token.word != text[token.shift:token.shift + token.word_len]:
                raise Exception(f'ERROR: {token.word} != {text[token.shift:token.shift + token.word_len]}')
                
    def detect(self, text):
        tokens = self._filter_tokens(self.text_preprocessor.tokenize_text(text))
        self._sanity_check(tokens, text)
        concepts = []

        start_w_i = 0
        while start_w_i < len(tokens):
            last_ok_tokens = [tokens[start_w_i]]
            for end_w_i in range(start_w_i + 1, len(tokens) + 1):
                if end_w_i - start_w_i > 1 and tokens[end_w_i-1].num - tokens[end_w_i-2].num != 1:
                    break
                word = '_'.join([t.token for t in tokens[start_w_i:end_w_i]])
                if word in self.senses[end_w_i - start_w_i]:
                    if end_w_i - start_w_i > 1:
                        last_ok_tokens = tokens[start_w_i:end_w_i]

                if word not in self.issensecontinuation:
                    break
            if '_'.join([t.token for t in last_ok_tokens]) in self.senses[len(last_ok_tokens)]:
                concepts.append(last_ok_tokens)
            start_w_i += len(last_ok_tokens)
        
        concepts = [ConceptToken(concept_tokens, self.wordnet.sense2synid['_'.join([t.token for t in concept_tokens])]) for concept_tokens in concepts]
        return concepts