# high level embedding interface

import time
import numpy as np
from transformers import AutoTokenizer

from .ggml import LlamaPoolingType
from .models.bert import BertModel

# we don't want to require torch for gadget
try:
    import torch
except ImportError:
    pass

##
## embed
##

class EmbedBase:
    def __init__(self, gguf_path, model_id, pooling=None, model_class=BertModel, **kwargs):
        self.toker = AutoTokenizer.from_pretrained(model_id)
        self.model = model_class.from_path(gguf_path, **kwargs)
        self.batch_size = self.model.params['batch_size']

        # assign pooling type
        if pooling is None:
            pooling = self.model.params.get('bert.pooling_type', LlamaPoolingType.NONE)

        self.pooling = LlamaPoolingType(pooling)

    def tokenize(self, texts):
        if type(texts) is str:
            texts = [texts]
        return self.toker(texts)['input_ids']

    def embed(self, texts, pooling=None, normalize=True):
        # get tokens as list of lists
        tokens = self.tokenize(texts)
        total = sum([len(t) for t in tokens])

        # check for batch fit
        if total > self.batch_size:
            raise ValueError(f'Number of tokens ({total}) > batch_size ({self.batch_size})')

        # create input arrays and compute
        tokids, posids, seqids, mask = self.prepare_inputs(tokens)
        embeds = self.model(tokids, posids, mask, n_tokens=total)

        # do requested pooling
        pooling = self.pooling if pooling is None else pooling
        seqids = seqids.to(embeds.device) if self.model.framework == 'torch' else seqids
        embeds = self.pool_embeds(pooling, embeds[:total, :], seqids[:total])

        # return embeddings
        if normalize:
            embeds = self.norm_embeds(embeds)
        return embeds

class EmbedNumpy(EmbedBase):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, framework='numpy', **kwargs)

    @staticmethod
    def norm_embeds(embeds):
        return embeds / np.linalg.norm(embeds, axis=-1, keepdims=True)

    # this assumes that sequences are packed in order
    @staticmethod
    def pool_embeds(pooling, embeds, seqids):
        if pooling == LlamaPoolingType.NONE:
            _, first = np.unique(seqids, return_index=True)
            pooled = np.split(embeds, first)
        elif pooling == LlamaPoolingType.MEAN:
            uids, ntoks = np.unique(seqids, return_counts=True)
            weights = (uids[:, None] == seqids[None, :]).astype(np.float32)
            weights /= ntoks[:, None]
            pooled = weights @ embeds
        elif pooling == LlamaPoolingType.CLS:
            _, first = np.unique(seqids, return_index=True)
            pooled = embeds[first, :]
        elif pooling == LlamaPoolingType.LAST:
            _, rlast = np.unique(seqids[::-1], return_index=True)
            last = len(seqids) - rlast - 1
            pooled = embeds[last, :]
        else:
            raise ValueError('must specify pooling type')
        return pooled

    def prepare_inputs(self, tokens):
        ntoks = np.array([len(ts) for ts in tokens], dtype=np.int32)
        nseqs, total = len(tokens), ntoks.sum()
        padding = np.zeros(self.batch_size - total, dtype=np.int32)
        tokens = np.concatenate([*(np.array(t, dtype=np.int32) for t in tokens), padding])
        posits = np.concatenate([*(np.arange(n, dtype=np.int32) for n in ntoks), padding])
        seqids = np.concatenate([np.repeat(np.arange(nseqs, dtype=np.int32), ntoks), padding - 1])
        mask = np.where(seqids[:, None] == seqids[None, :], 0.0, -np.inf).astype(np.float32)
        return tokens, posits, seqids, mask

class EmbedTorch(EmbedBase):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, framework='torch', **kwargs)

    @staticmethod
    def norm_embeds(embeds):
        return embeds / embeds.norm(dim=-1, keepdim=True)

    # this assumes ordered and contiguous seqids
    @staticmethod
    def first_indices(values):
        uvals, indices = values.unique(return_inverse=True)
        first = torch.empty(len(uvals), device=values.device, dtype=torch.int32)
        order = torch.arange(len(values), device=values.device, dtype=torch.int32)
        first.scatter_reduce_(0, indices, order, reduce='amin')
        return first

    # this assumes that sequences are packed in order
    @classmethod
    def pool_embeds(cls, pooling, embeds, seqids):
        if pooling == LlamaPoolingType.NONE:
            first = cls.first_indices(seqids)
            pooled = torch.split(embeds, first, 0)
        elif pooling == LlamaPoolingType.MEAN:
            uids, ntoks = torch.unique(seqids, return_counts=True)
            weights = (uids[:, None] == seqids[None, :]).float()
            weights /= ntoks[:, None]
            pooled = weights @ embeds
        elif pooling == LlamaPoolingType.CLS:
            first = cls.first_indices(seqids)
            pooled = embeds[first, :]
        elif pooling == LlamaPoolingType.LAST:
            rlast = cls.first_indices(seqids[::-1])
            last = len(seqids) - rlast - 1
            pooled = embeds[last, :]
        else:
            raise ValueError('must specify pooling type')
        return pooled

    def prepare_inputs(self, tokens):
        ntoks = torch.tensor([len(ts) for ts in tokens], dtype=torch.int32)
        nseqs, total = len(tokens), ntoks.sum()
        padding = torch.zeros(self.batch_size - total, dtype=torch.int32)
        tokens = torch.cat([*(torch.tensor(t, dtype=torch.int32) for t in tokens), padding])
        posits = torch.cat([*(torch.arange(n, dtype=torch.int32) for n in ntoks), padding])
        seqids = torch.cat([torch.repeat_interleave(torch.arange(nseqs, dtype=torch.int32), ntoks), padding - 1])
        mask = torch.where(seqids[:, None] == seqids[None, :], 0.0, -torch.inf).float()
        return tokens, posits, seqids, mask

##
## test
##

def test_embed(gguf_path, model_id, prompt='hello world', embed_class=EmbedTorch, model_class=BertModel, **kwargs):
    from transformers import AutoModel

    # embed with gg model
    gg_model = embed_class(gguf_path, model_id, model_class=model_class, **kwargs)
    gg_embed = gg_model.embed(prompt)

    # bring to host numpy if needed
    if hasattr(gg_embed, 'numpy'):
        gg_embed = gg_embed.cpu().numpy()

    # embed with hf
    hf_toker = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModel.from_pretrained(model_id)
    hf_input = hf_toker(prompt, return_tensors='pt')['input_ids']
    hf_seqid = torch.zeros_like(hf_input)
    with torch.no_grad():
        hf_state = hf_model(hf_input).last_hidden_state[0]
    hf_poold = EmbedTorch.pool_embeds(gg_model.pooling, hf_state, hf_seqid)
    hf_embed = hf_poold / hf_poold.norm(dim=-1, keepdim=True)
    hf_embed = hf_embed.cpu().numpy()

    # embed with ggml
    simil = (hf_embed * gg_embed).sum()
    print(simil)

    return gg_model

def profile_embed(gguf_path, model_id, prompt=None, length=256, reps=25, embed_class=EmbedTorch, model_class=BertModel, **kwargs):
    if prompt is None:
        prompt = ' '.join('a' for _ in range(length-2))

    t0 = time.time()
    total = reps * length

    gg_model = embed_class(gguf_path, model_id, model_class=model_class, **kwargs)

    t1 = time.time()
    d1 = t1 - t0
    print(f'load: {d1}')

    for _ in range(reps):
        gg_embed = gg_model.embed(prompt)

    t2 = time.time()
    d2 = t2 - t1
    print(f'embd: {d2}')
    print(f'rps : {reps/d2}')
    print(f'tps : {total/d2}')
