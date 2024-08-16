# high level embedding interface

import numpy as np
from transformers import AutoTokenizer

from .ggml import LlamaPoolingType
from .bert import BertModel

##
## utils
##

def pad_concat(arrs, length, dtype, fill):
    full = np.full((length,), fill, dtype=dtype)
    pos = 0
    for arr in arrs:
        num = len(arr)
        full[pos:pos+num] = arr
        pos += num
    return full

# only need causal for now
def attention_matrix(seqids, null_id=-1):
    mask = seqids[:, None] == seqids[None, :]
    logits = np.where(mask, 0.0, -np.inf)
    return logits

def l2_normalize(values, axis=-1):
    return values / np.linalg.norm(values, axis=axis, keepdims=True)

# this assumes that sequences are packed in order
def pool_embeds(embeds, pooling, seqids):
    if pooling == LlamaPoolingType.NONE:
        pooled = embeds
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

##
## embed
##

class EmbedModel:
    def __init__(self, gguf_path, model_id, batch_size=512, pooling=None, model_class=BertModel, **kwargs):
        self.batch_size = batch_size
        self.toker = AutoTokenizer.from_pretrained(model_id)
        self.model = model_class.from_path(gguf_path, batch_size=batch_size)

        # assign pooling type
        if pooling is None:
            pooling = self.model.params.get('bert.pooling_type', LlamaPoolingType.NONE.value)
        self.pooling = pooling

    def encode(self, tokens, sequences=None, positions=None):
        # ensure ndarray inputs
        tokens = np.asarray(tokens, dtype=np.int32)
        sequences = np.asarray(sequences, dtype=np.int32)
        positions = np.asarray(positions, dtype=np.int32)

        # handle single sequence case
        if sequences is None:
            sequences = np.zeros_like(tokens, dtype=np.int32)
        if positions is None:
            positions = np.arange(self.batch_size, dtype=np.int32)

        # set up attention matrix and compute
        attention = attention_matrix(sequences).astype(np.float32)
        embed = self.model(tokens=tokens, positions=positions, attention=attention)
        return embed

    def embed(self, texts, pooling=None, normalize=True):
        # handle singleton case
        if type(texts) is str:
            texts = [texts]

        # get tokens as list of lists
        tokens = self.toker(texts)['input_ids']
        ntoks = [len(ts) for ts in tokens]

        # check for batch fit
        if (total := sum(ntoks)) > (batch_size := self.batch_size):
            raise ValueError(f'Number of tokens ({total}) > batch_size ({batch_size})')

        # make numpy inputs for model
        tokens = pad_concat(tokens, batch_size, np.int32, 0)
        posits = pad_concat([range(n) for n in ntoks], batch_size, np.int32, 0)
        seqids = pad_concat([n * [i] for i, n in enumerate(ntoks)], batch_size, np.int32, -1)

        # call model encoder
        embeds = self.encode(tokens, positions=posits, sequences=seqids)[:total,:]

        # do requested pooling
        pooling = self.pooling if pooling is None else pooling
        embeds = pool_embeds(embeds, pooling, seqids[:total])

        # return embeddings
        if normalize:
            embeds = l2_normalize(embeds)
        return embeds

##
## test
##

def test_bert(gguf_path, model_id, prompt='hello world', batch_size=512):
    import torch
    from transformers import AutoModel

    # load hf model
    hf_toker = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModel.from_pretrained(model_id)

    # load gg model
    gg_model = EmbedModel(gguf_path, model_id, batch_size)
    pooling = gg_model.pooling

    # embed with hf
    hf_input = hf_toker(prompt, return_tensors='pt')['input_ids']
    hf_seqid = torch.zeros_like(hf_input).numpy()
    hf_state = hf_model(hf_input).last_hidden_state[0].detach().numpy()
    hf_poold = pool_embeds(hf_state, pooling, hf_seqid)
    hf_embed = l2_normalize(hf_poold)

    # embed with ggml
    gg_embed = gg_model.embed(prompt)

    # check results
    simil = (hf_embed * gg_embed).sum()
    print(simil)

    return hf_embed, gg_embed
