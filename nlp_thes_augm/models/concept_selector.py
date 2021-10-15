import numpy as np

class RandomConceptSelector:
    def __init__(self, wordnet):
        self.wordnet = wordnet
    
    def select(self, concept):
        sense = concept.get_sense_norm()
        synsets_ids = self.wordnet.sense2synid[sense]
        return np.random.choice(synsets_ids, 1).tolist()[0]