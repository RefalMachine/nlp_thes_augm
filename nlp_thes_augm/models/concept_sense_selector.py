import numpy as np

class RandomConceptSenseSelector:
    def __init__(self, wordnet):
        self.wordnet = wordnet
        
    def select(self, synset_id):
        senses = list(self.wordnet.synsets[synset_id].synset_words)
        idx = np.random.choice(range(len(senses)), 1).tolist()[0]
        return senses[idx]