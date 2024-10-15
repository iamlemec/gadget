# text generation

import numpy as np
from transformers import AutoTokenizer

from .loader import GgufFile
from .models.llama import LlamaModel

# we don't want to require torch for gadget
try:
    import torch
except ImportError:
    pass

def load_model(gguf_or_path, model_class, **kwargs):
    if type(gguf_or_path) is str:
        return model_class.from_path(gguf_or_path, **kwargs)
    elif type(gguf_or_path) is GgufFile:
        return model_class.from_gguf(gguf_or_path, **kwargs)
    else:
        raise ValueError('must specify gguf file or path')

def sample(logits, temperature=0.7, top_p=0.9, top_k=50):
    probs = torch.exp(logits / temperature)
    probs = probs / torch.sum(probs)
    return torch.multinomial(probs, num_samples=1).item()

def sprint(text):
    print(text, end='', flush=True)

class TextGen:
    def __init__(self, gguf_path, model_id, model_class=LlamaModel, **kwargs):
        self.model = load_model(gguf_path, model_class, framework='torch', **kwargs)
        self.toker = AutoTokenizer.from_pretrained(model_id)

    def reset(self):
        self.model.reset()

    def tokenize(self, texts, **kwargs):
        return self.toker(texts, **kwargs)['input_ids']

    def detokenize(self, tokens, **kwargs):
        return self.toker.decode(tokens, **kwargs)

    def logits(self, tokens):
        return torch.atleast_2d(self.model(tokens))

    def sample(self, tokens, **kwargs):
        logits = self.logits(tokens)
        return sample(logits[-1,:], **kwargs)

    def stream_tokens(self, tokens, max_gen=128, **kwargs):
        batch = tokens
        for _ in range(max_gen):
            tok = self.sample(batch, **kwargs)
            batch = [tok]
            yield tok

    def stream(self, text, **kwargs):
        tokens = self.tokenize(text)
        for tok in self.stream_tokens(tokens, **kwargs):
            yield self.detokenize([tok], add_special_tokens=False)

    def generate(self, text, **kwargs):
        tokens = self.tokenize(text)
        for tok in self.stream_tokens(tokens, **kwargs):
            tokens += [tok]
        return self.detokenize(tokens)

class TextChat(TextGen):
    def __init__(self, gguf_path, model_id, system=None, **kwargs):
        super().__init__(gguf_path, model_id, **kwargs)
        self.system = system
        self.reset(system)

    def reset(self, system=None):
        super().reset()
        system = self.system if system is None else system
        if system is not None:
            self.history = [{'role': 'system', 'content': system}]
        else:
            self.history = []

    def stream_chat(self, message, prefill=None, add_generation_prompt=True, **kwargs):
        self.history.append({'role': 'user', 'content': message})
        n_past = self.model.state['n_past']
        tokens_full = self.toker.apply_chat_template(self.history, add_generation_prompt=add_generation_prompt)
        if prefill is not None:
            tokens_full += self.tokenize(prefill, add_special_tokens=False)
            reply = prefill
        else:
            reply = ''
        tokens = tokens_full[n_past:]
        for tok in self.stream_tokens(tokens, **kwargs):
            if tok == self.toker.eos_token_id:
                break
            text = self.detokenize([tok], add_special_tokens=False)
            reply += text
            yield text
        self.history.append({'role': 'assistant', 'content': reply})

    def generate_chat(self, message, **kwargs):
        tokens = []
        for tok in self.stream_chat(message, **kwargs):
            tokens += [tok]
        return self.detokenize(tokens)

def test_logits(gguf_path, model_id, model_class=LlamaModel, batch_size=128, **kwargs):
    model = TextGen(gguf_path, model_id, model_class=model_class, batch_size=batch_size, **kwargs)
    prompt = 'The capital of France is'
    tokes = model.tokenize(prompt)
    logits = model.logits(tokes)
    return logits

def test_logits_hf(model_id, **kwargs):
    from transformers import AutoModelForCausalLM
    hf_prompt = 'The capital of France is'
    hf_toker = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, output_hidden_states=True)
    hf_input = hf_toker(hf_prompt, return_tensors='pt')['input_ids']
    hf_pos = torch.arange(len(hf_input[0])).unsqueeze(0)
    with torch.no_grad():
        hf_state = hf_model(hf_input)[0]
    return hf_state[0]

def test_textgen(
    gguf_path, model_id, prompt='The capital of France is', model_class=LlamaModel, batch_size=128,
    max_gen=128, **kwargs
):
    model = TextGen(gguf_path, model_id, model_class=model_class, batch_size=batch_size, **kwargs)
    for tok in model.stream(prompt, max_gen=max_gen):
        sprint(tok)
