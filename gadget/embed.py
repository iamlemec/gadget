# high level embedding interface

import numpy as np
from transformers import AutoTokenizer

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
def attention_matrix(seq_ids, null_id=-1):
    mask = seq_ids[:, None] == seq_ids[None, :]
    logits = np.where(mask, 0.0, -np.inf)
    return logits

def l2_normalize(values, axis=-1):
    return values / np.linalg.norm(values, axis=axis, keepdims=True)

##
## embed
##

class EmbedModel:
    def __init__(self, gguf_path, model_id, batch_size=512, model_class=BertModel, **kwargs):
        self.batch_size = batch_size
        self.toker = AutoTokenizer.from_pretrained(model_id)
        self.model = model_class.from_path(gguf_path, batch_size=batch_size)

    def encode(self, tokens, sequences=None, positions=None):
        # ensure ndarray inputs
        tokens = np.asarray(tokens, dtype=np.int32)

        # handle single sequence case
        if sequences is None:
            sequences = np.zeros_like(tokens, dtype=np.int32)
        if positions is None:
            positions = np.arange(self.batch_size, dtype=np.int32)

        # set up attention matrix
        attention = attention_matrix(sequences).astype(np.float32)

        # compute on input data
        embed = self.model(tokens=tokens, positions=positions, attention=attention)

        # return embedding
        return embed

    def embed(self, texts, normalize=True):
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
        tokens_np = pad_concat(tokens, batch_size, np.int32, 0)
        posits_np = pad_concat([range(n) for n in ntoks], batch_size, np.int32, 0)
        seqids_np = pad_concat([n * [i] for i, n in enumerate(ntoks)], batch_size, np.int32, -1)

        # call model encoder
        embeds = self.encode(tokens_np, sequences=seqids_np)[:total,:]

        # normalize if needed
        if normalize:
            embeds = l2_normalize(embeds)

        # return embeddings
        return embeds

##
## test
##

def test_bert(gguf_path, model_id, prompt='hello world', batch_size=512):
    # load hf model
    from transformers import AutoModel
    hf_toker = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModel.from_pretrained(model_id)

    # embed with hf
    hf_input = hf_toker(prompt, return_tensors='pt')['input_ids']
    hf_state = hf_model(hf_input).last_hidden_state[0]
    hf_embed = l2_normalize(hf_state.detach().numpy())

    # embed with ggml
    gg_model = EmbedModel(gguf_path, model_id, batch_size)
    gg_embed = gg_model.embed(prompt)

    # check results
    match = np.allclose(hf_embed, gg_embed, atol=1e-3)
    print(match)

    return hf_embed, gg_embed
