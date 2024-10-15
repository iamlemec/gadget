# shrek llama

import torch
from math import sqrt

from entropix.torch_sampler import sample, device
from .textgen import TextGen, TextChat

class ShrekMixin:
    def sample(self, tokens, **kwargs):
        head_dim = self.model.params['head_dim_kv']
        n_layers = self.model.params['llama.block_count']
        score_name = f'attn{n_layers-1}_pre_scores'

        tokens = torch.tensor(tokens, dtype=torch.int32, device=device)
        logits = self.logits(tokens)
        scores = self.model.get_named_node(score_name) / sqrt(head_dim)

        batch_tokens = tokens.unsqueeze(0)
        batch_logits = logits.unsqueeze(0)
        batch_scores = scores.unsqueeze(0)

        nexts = sample(batch_tokens, batch_logits, batch_scores, **kwargs)
        return nexts.squeeze(0).item()

class ShrekGen(ShrekMixin, TextGen):
    pass

class ShrekChat(ShrekMixin, TextChat):
    pass
