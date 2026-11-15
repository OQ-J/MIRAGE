"""Full valid-token Qwen encoding; no dataset or shard pipeline."""
import torch

class TextEncoder:
    def __init__(self, path, device='cuda', max_tokens=32768, compute_dtype='float32'):
        from transformers import AutoModel, AutoTokenizer
        self.device = torch.device(device)
        self.compute_dtype = compute_dtype
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, padding_side='right')
        self.eos = self.tokenizer.convert_tokens_to_ids('<|endoftext|>')
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.eos
        self.model = AutoModel.from_pretrained(
            str(path), local_files_only=True,
            torch_dtype=getattr(torch, compute_dtype),
            attn_implementation='sdpa'
        ).to(self.device).eval().requires_grad_(False)
        assert self.model.config.hidden_size == 1024
        self.max_tokens = min(max_tokens, self.model.config.max_position_embeddings)

    def tokenize(self, texts):
        ids = self.tokenizer(list(texts), add_special_tokens=False, truncation=False, padding=False)['input_ids']
        result = [row + [self.eos] for row in ids]
        for index, row in enumerate(result):
            if len(row) > self.max_tokens:
                raise ValueError(f'Text {index}: {len(row)} tokens exceeds --max-text-tokens={self.max_tokens}; no silent truncation.')
        return result

    @torch.inference_mode()
    def encode_ids(self, sequences):
        batch = self.tokenizer.pad(dict(input_ids=sequences, attention_mask=[[1] * len(ids) for ids in sequences]),
                                   padding=True, return_tensors='pt')
        batch = {key: value.to(self.device) for key, value in batch.items()}
        # Explicit logical positions make valid-token positions independent of padding length.
        positions = batch['attention_mask'].cumsum(-1) - 1
        positions.masked_fill_(batch['attention_mask'] == 0, 0)
        hidden = self.model(**batch, position_ids=positions, use_cache=False).last_hidden_state
        return [hidden[i, :len(ids)].to(device='cpu', dtype=torch.float32).clone().contiguous()
                for i, ids in enumerate(sequences)]
