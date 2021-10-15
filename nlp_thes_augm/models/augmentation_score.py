import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPT2Score:
    def __init__(self, model_path, device='cuda'):
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.device = device
        
    def score(self, text):
        max_length = self.model.config.n_positions
        stride = 512
        encodings = self.tokenizer(text, return_tensors='pt')
        nlls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i    # may be different from stride on last loop
            input_ids = encodings.input_ids[:,begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:,:-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len

            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        return float(ppl.detach().cpu())

    def score_batch(self, texts, bs=4):
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