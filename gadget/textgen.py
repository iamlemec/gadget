# text generation

import numpy as np
from transformers import AutoTokenizer

from .loader import GgufFile
from .llama import LlamaModel

def pad_array(val, length, dtype=None, pad_val=0):
    if dtype is None:
        dtype = val.dtype
    arr = np.full((length,), pad_val, dtype=dtype)
    arr[:len(val)] = val
    return arr

def load_model(gguf_or_path, model_class, **kwargs):
    if type(gguf_or_path) is str:
        return model_class.from_path(gguf_or_path, **kwargs)
    elif type(gguf_or_path) is GgufFile:
        return model_class.from_gguf(gguf_or_path, **kwargs)
    else:
        raise ValueError('must specify gguf file or path')

class TextGen:
    def __init__(self, gguf_path, model_id, model_class=LlamaModel, **kwargs):
        self.model = load_model(gguf_path, model_class, **kwargs)
        self.toker = AutoTokenizer.from_pretrained(model_id)
        self.batch_size = self.model.params['batch_size']

    def tokenize(self, texts):
        return self.toker(texts)['input_ids']

    def detokenize(self, tokens):
        return self.toker.decode(tokens)

    def prepare_inputs(self, tokens):
        n_toks = len(tokens)
        tokids = pad_array(tokens, self.batch_size, dtype=np.int32)
        posids = pad_array(range(n_toks), self.batch_size, dtype=np.int32)
        mask   = np.where(posids[None, :] <= posids[:, None], 0.0, -np.inf).astype(np.float32)
        return tokids, posids, mask

    def sample(self, logits, temperature=0.7, top_p=0.9, top_k=50):
        probs = np.exp(logits / temperature)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs)

    def generate_next(self, tokens, **kwargs):
        tokids, posids, mask = self.prepare_inputs(tokens)
        logits = self.model(tokens=tokids, positions=posids, mask=mask)
        return self.sample(logits, **kwargs)

def test_textgen(gguf_path, model_id, prompt='The capital of France is', model_class=LlamaModel, **kwargs):
    model = TextGen(gguf_path, model_id, model_class=model_class, **kwargs)
    toks = model.tokenize(prompt)
    tokids, posids, mask = model.prepare_inputs(toks)
    logits = model.model(tokens=tokids, positions=posids, mask=mask)
    return logits
