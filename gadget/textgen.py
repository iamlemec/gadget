# text generation

import numpy as np
from transformers import AutoTokenizer

from .loader import GgufFile
from .llama import LlamaModel

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

class TextGen:
    def __init__(self, gguf_path, model_id, model_class=LlamaModel, **kwargs):
        self.model = load_model(gguf_path, model_class, framework='torch', **kwargs)
        self.toker = AutoTokenizer.from_pretrained(model_id)
        self.context_length = self.model.params['context_length']
        self.batch_size = self.model.params['batch_size']
        self.tokens = torch.zeros(self.context_length, dtype=torch.int32)

    def tokenize(self, texts):
        return self.toker(texts)['input_ids']

    def detokenize(self, tokens):
        return self.toker.decode(tokens)

    def sample(self, logits, temperature=0.7, top_p=0.9, top_k=50):
        probs = torch.exp(logits / temperature)
        probs = probs / torch.sum(probs)
        return torch.multinomial(probs, num_samples=1).item()

    def next_token(self, tokens, **kwargs):
        logits = self.model(self.tokens)
        return self.sample(logits[-1,:], **kwargs)

    def generate_next(self, text, **kwargs):
        toks = self.tokenize(text)
        gen = self.next_token(toks, **kwargs)
        return self.detokenize(gen)

def test_textgen(
    gguf_path, model_id, prompt='The capital of France is', model_class=LlamaModel,
    batch_size=128, **kwargs
):
    model = TextGen(
        gguf_path, model_id, model_class=model_class, batch_size=batch_size, **kwargs
    )
    toks = model.tokenize(prompt)
    tokens = torch.tensor(toks, dtype=torch.int32)
    output = model.model(tokens)
    return output

def test_huggingface(model_id, prompt='The capital of France is', **kwargs):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    hf_toker = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, output_hidden_states=True)

    hf_input = hf_toker(prompt, return_tensors='pt')['input_ids']
    hf_pos = torch.arange(len(hf_input[0])).unsqueeze(0)

    with torch.no_grad():
        hf_state = hf_model(hf_input)[0]

        # hf_state = hf_model.model.embed_tokens(hf_input)

        # hf_state = hf_layer(hf_state, position_ids=hf_pos)[0]

        # hf_layer = hf_model.model.layers[0]
        # residual = hf_state
        # hf_state = hf_layer.input_layernorm(hf_state)
        # hf_state = hf_layer.self_attn(hf_state, position_ids=hf_pos)[0]
        # hf_state = residual + hf_state
        # residual = hf_state
        # hf_state = hf_layer.post_attention_layernorm(hf_state)
        # hf_state = hf_layer.mlp(hf_state)
        # hf_state = residual + hf_state

    return hf_state[0]
