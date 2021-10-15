import copy
from tqdm import tqdm
from nlp_thes_augm.utils.common import Augmentation, inject_augmentation
from nlp_thes_augm.utils.text_preprocessor import TextPreprocessor
from nlp_thes_augm.models.concept_detector import ConceptDetector
from nlp_thes_augm.models.concept_selector import RandomConceptSelector
from nlp_thes_augm.models.concept_sense_selector import RandomConceptSenseSelector
from nlp_thes_augm.models.concept_subset_selector import RandomConceptsSubsetSelector
from nlp_thes_augm.models.text_restore import T5TextRestore
from nlp_thes_augm.models.augmentation_score import GPT2Score

class ConceptAugment:
    def __init__(self, wordnet, concept_subset_selector, concept_selector, sense_selector):
        self.wordnet = wordnet
        self.concept_subset_selector = concept_subset_selector
        self.concept_selector = concept_selector
        self.sense_selector = sense_selector
        
    def augment_one(self, concepts, text, max_concepts_count, only_deterministic=False):
        concepts_subset = self.concept_subset_selector.select(
            copy.deepcopy(concepts), max_concepts_count=max_concepts_count, 
            only_deterministic=only_deterministic
        )
        synset_ids = [self.concept_selector.select(c) for c in concepts_subset]
        senses = [self.sense_selector.select(synid) for synid in synset_ids]
        
        return inject_augmentation(concepts_subset, senses, text), concepts_subset, senses, synset_ids
    
    def augmentate(self, concepts, text, max_concepts_count, augmentations_count, only_deterministic=False):
        augmentations = []
        for i in range(augmentations_count):
            augmentation, concepts_subset, senses, synset_ids = self.augment_one(
                concepts, text, max_concepts_count, only_deterministic
            )
            augmentations.append(Augmentation(text, concepts_subset, senses, synset_ids, augmentation))
            
        return augmentations

class WordnetAugmentator:
    def __init__(self, wordnet, t5_model_path, gpt2_model_path):
        self.wordnet = wordnet
        self.text_preprocessor = TextPreprocessor()
        self.concept_detector = ConceptDetector(wordnet, self.text_preprocessor)
        
        concepts_subset_selector = RandomConceptsSubsetSelector()
        concept_selector = RandomConceptSelector(wordnet)
        sense_selector = RandomConceptSenseSelector(wordnet)
        self.concept_augmentator = ConceptAugment(wordnet, concepts_subset_selector, concept_selector, sense_selector)
        
        self.t5_text_resorer = T5TextRestore(t5_model_path)
        self.gpt2_score = GPT2Score(gpt2_model_path)
        
    def augmentate(self, text, augmentations_count, topk, max_concept_count=5, bs=1):
        detected_concepts = self.concept_detector.detect(text)
        augmentations = self.concept_augmentator.augmentate(
            detected_concepts, text, max_concept_count, augmentations_count
        )
        
        if bs > 1:
            restored_augmentations = self.t5_text_resorer.restore_batch(
                [augm.augmentation for augm in tqdm(augmentations)], bs=bs
            )
        else:
            restored_augmentations = []
            for augm in tqdm(augmentations):
                restored_augmentations.append(self.t5_text_resorer.restore(augm.augmentation))
                
        for i in range(len(restored_augmentations)):
            augmentations[i].restored_augmentation = restored_augmentations[i]
            
        for augm in tqdm(augmentations):
            augm.score = self.gpt2_score.score(augm.restored_augmentation)
            
        augmentations = sorted(augmentations, key=lambda x: x.score)
        augmentations = self._filter_duplicates(augmentations)
        return augmentations[:topk], detected_concepts

    @staticmethod
    def _filter_duplicates(augmentations):
        processed = set()
        filtered_augmentations = []
        for augm in augmentations:
            if augm.text == augm.restored_augmentation:
                continue
            if augm.restored_augmentation in processed:
                continue
            filtered_augmentations.append(augm)
            processed.add(augm.restored_augmentation)
            
        return filtered_augmentations