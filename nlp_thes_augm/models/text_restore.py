import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

class T5TextRestore:
    def __init__(self, model_path, device='cuda', prefix='thes_augm: '):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
        self.model.eval()
        self.device = device
        self.prefix = prefix
        
    def restore(self, text):
        text = self.prefix + text
        with torch.no_grad():
            model_input = self.tokenizer(text, return_tensors='pt', padding=True).to(self.device)
            max_size = int(model_input.input_ids.shape[1] * 1.5 + 10)
            out = self.model.generate(**model_input, max_length=max_size).detach().cpu()[0]
            result = self.tokenizer.decode(out, skip_special_tokens=True)
        return result
    
    def restore_batch(self, texts, bs=4):
        texts = [self.prefix + t for t in texts]
        batch_count = len(texts) // bs + int(len(texts) % bs != 0)
        results = []
        for i in tqdm(range(batch_count)):
            batch = texts[i * bs: (i + 1) * bs]
            model_input = self.tokenizer(batch, return_tensors='pt', padding=True).to(self.device)
            max_size = int(model_input.input_ids.shape[1] * 1.5 + 10)
            out = self.model.generate(**model_input, max_length=max_size).detach().cpu()
            result = [self.tokenizer.decode(o, skip_special_tokens=True) for o in out]
            results += result
        return results