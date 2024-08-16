# bert implementation

import numpy as np

from .ggml import (
    ggml_add,
    ggml_add_inplace,
    ggml_get_rows,
    ggml_view_1d,
)
from .layers import (
    linear_layer,
    norm_layer,
    attention_layer,
    feed_forward_layer,
)
from .model import GgmlModel, Tensor

##
## utils
##

# only need causal for now
def attention_matrix(seq_ids, null_id=-1):
    mask = seq_ids[:, None] == seq_ids[None, :]
    logits = np.where(mask, 0.0, -np.inf)
    return logits

def normalize(values, axis=-1):
    return values / np.linalg.norm(values, axis=axis, keepdims=True)

def padded_array(dtype, length, values, fill):
    arr = np.full((length,), fill, dtype=dtype)
    arr[:len(values)] = values
    return arr

##
## bert model
##

class BertModel(GgmlModel):
    tokens   : Tensor('I32', ('batch_size',))
    positions: Tensor('I32', ('batch_size',))
    attention: Tensor('F32', ('batch_size', 'batch_size'))

    def __init__(self, params, tensors, backend=None):
        # validate batch_size
        if (bs := params['batch_size']) > (cl := params['bert.context_length']):
            raise ValueError('batch_size ({bs}) > context_length ({cl})')

        # pass to model constructor
        super().__init__(params, tensors, backend=backend)

    # bert model function
    def forward(self):
        ctx = self.ctx_graph

        # get params
        n_layers, n_heads, embed_dim, layer_norm_eps = self.params[
            'bert.block_count', 'bert.attention.head_count',
            'bert.embedding_length', 'bert.attention.layer_norm_epsilon'
        ]

        # get weights
        etok, etyp, epos, tnw, tnb = self.tensors[
            'token_embd.weight', 'token_types.weight', 'position_embd.weight',
            'token_embd_norm.weight', 'token_embd_norm.bias',
        ]

        # get inputs
        tokens, positions, mask = self.tensors['tokens', 'positions', 'attention']

        # get token embeddings (token+type+position+norm)
        cur = ggml_get_rows(ctx, etok, tokens, name='embed=tok')
        cur = ggml_add_inplace(ctx, cur, ggml_view_1d(ctx, etyp, embed_dim, 0), name='embed=tok+typ')
        cur = ggml_add_inplace(ctx, cur, ggml_get_rows(ctx, epos, positions), name='embed=tok+typ+pos')
        cur = norm_layer(ctx, cur, tnw, tnb, layer_norm_eps, inplace=True, name='embed_norm')

        # loop over layers
        for i in range(n_layers):
            # get layer tensors
            wq, bq, wk, bk, wv, bv, wo, bo, wan, ban, wu, bu, wd, bd, wln, bln = self.tensors[
                f'blk.{i}.attn_q.weight'           , f'blk.{i}.attn_q.bias'           ,
                f'blk.{i}.attn_k.weight'           , f'blk.{i}.attn_k.bias'           ,
                f'blk.{i}.attn_v.weight'           , f'blk.{i}.attn_v.bias'           ,
                f'blk.{i}.attn_output.weight'      , f'blk.{i}.attn_output.bias'      ,
                f'blk.{i}.attn_output_norm.weight' , f'blk.{i}.attn_output_norm.bias' ,
                f'blk.{i}.ffn_up.weight'           , f'blk.{i}.ffn_up.bias'           ,
                f'blk.{i}.ffn_down.weight'         , f'blk.{i}.ffn_down.bias'         ,
                f'blk.{i}.layer_output_norm.weight', f'blk.{i}.layer_output_norm.bias',
            ]

            # get attention interactions
            att = attention_layer(
                ctx, cur, n_heads, mask, wq, bq, wk, bk, wv, bv, wo, bo,
                eps=layer_norm_eps, name=f'attn{i}'
            )

            # add attention output to current then normalize
            att = ggml_add_inplace(ctx, cur, att)
            att = norm_layer(ctx, att, wan, ban, layer_norm_eps, inplace=True, name=f'attn{i}_norm')

            # feed forward network on current
            cur = feed_forward_layer(ctx, att, wu, bu, wd, bd, act='gelu', name=f'ffn{i}')

            # add attention output to current tensor and normalize
            cur = ggml_add_inplace(ctx, cur, att, name=f'add{i}')
            cur = norm_layer(ctx, cur, wln, bln, layer_norm_eps, inplace=True, name=f'norm{i}')

        # return embedding
        return cur

    def encode(self, tokens, positions=None, sequences=None):
        # get runtime parameters
        batch_size = self.params['batch_size']

        # validate input tokens
        tokens = np.asarray(tokens, dtype=np.int32)
        if tokens.shape != (batch_size,):
            raise ValueError('tokens must be an array of shape (batch_size,)')

        # handle single sequence case
        if sequences is None:
            sequences = np.zeros_like(tokens, dtype=np.int32)

        # generate token positions
        if positions is None:
            positions = np.arange(batch_size, dtype=np.int32)

        # set up attention matrix
        attention = attention_matrix(sequences).astype(np.float32)

        # compute on input data
        embed = self(tokens=tokens, positions=positions, attention=attention)

        # return embedding
        return embed

def test_bert(gguf_path, model_id, prompt='hello world', batch_size=512):
    import torch
    from transformers import AutoTokenizer, AutoModel

    # load tokenizer
    toker = AutoTokenizer.from_pretrained(model_id)
    tokens = toker(prompt)['input_ids']
    n_tokens = len(tokens)
    seqids = n_tokens * [0]

    # load hf model
    hf_model = AutoModel.from_pretrained(model_id)
    hf_tokens = torch.tensor(tokens, dtype=torch.int64).unsqueeze(0)
    hf_embed = normalize(hf_model(hf_tokens).last_hidden_state.squeeze(0).detach().numpy())

    # load gguf model
    gg_model = BertModel.from_path(gguf_path, batch_size=batch_size)
    gg_tokens = padded_array(np.int32, batch_size, tokens, 0)
    gg_seqids = padded_array(np.int32, batch_size, seqids, -1)
    gg_embed = normalize(gg_model.encode(gg_tokens, sequences=gg_seqids)[:n_tokens,:])

    # check results
    match = np.allclose(hf_embed, gg_embed, atol=1e-3)
    print(match)

    return hf_embed, gg_embed

