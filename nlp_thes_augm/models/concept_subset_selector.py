import numpy as np

class RandomConceptsSubsetSelector:
    def __init__(self):
        pass
    
    def select(self, concepts, max_concepts_count=5, only_deterministic=False):
        if only_deterministic:
            concepts = [c for c in concepts if len(c.synsets_ids) == 1]
        concept_count = min(max_concepts_count, len(concepts))
        if concept_count == 0:
            return []
        
        idx = np.random.choice(range(len(concepts)), concept_count, replace=False)
        selected_concepts = np.array(concepts)[idx].tolist()
        return sorted(selected_concepts, key=lambda x: x.tokens[0].num)