class Token:
    def __init__(self, word, token, num, shift):
        self.word = word
        self.word_len = len(word)
        self.token = token
        self.num = num
        self.shift = shift

class ConceptToken:
    def __init__(self, tokens, synsets_ids):
        self.tokens = tokens
        self.synsets_ids = synsets_ids
        
    def get_sense(self):
        return ' '.join([t.word for t in self.tokens])
    
    def get_sense_norm(self):
        return '_'.join([t.token for t in self.tokens])
        
    def __str__(self):
        sense = self.get_sense()
        sense_norm = self.get_sense_norm()
        synsets = ','.join(self.synsets_ids)
        return '{{' + sense + '|' + sense_norm + '|' + synsets + '}}'
        
class Augmentation:
    def __init__(self, text, concepts, senses, synsets_ids, augmentation, restored_augmentation='', score =1.0):
        self.text = text
        self.concepts = concepts
        self.senses = senses
        self.synsets_ids = synsets_ids
        self.augmentation = augmentation
        self.restored_augmentation = restored_augmentation
        self.score = score

def inject_concept_tokens(concept_tokens, text):
    shift = 0
    text_injected = ''
    for concept_token in concept_tokens:
        first_token = concept_token.tokens[0]
        last_token = concept_token.tokens[-1]

        info = str(concept_token)
        text_injected += text[shift:first_token.shift] + info

        shift = last_token.shift + last_token.word_len

    text_injected += text[shift:]
    return text_injected
    
def inject_augmentation(concept_tokens, senses, text):
    text_augmented = ''
    shift = 0

    for concept, sense in zip(*[concept_tokens, senses]):
        sense = sense.replace('_', ' ')

        first_token = concept.tokens[0]
        last_token = concept.tokens[-1]

        text_augmented += text[shift:first_token.shift] + f'{sense}'
        shift = last_token.shift + last_token.word_len

    text_augmented += text[shift:]
    return text_augmented