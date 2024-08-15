# bert implementation

import numpy as np

from .ggml import (
    GGMLQuantizationType,
    ggml_get_rows,
    ggml_add,
    ggml_add_inplace,
    ggml_mul,
    ggml_mul_inplace,
    ggml_norm,
    ggml_mul_mat,
    ggml_view_1d,
)
from .layers import linear_layer, norm_layer, attention_layer
from .loader import GgufFile
from .compute import set_tensor_name
from .model import GgmlModel, Tensor

##
## bert model
##

# only need causal for now
def attention_matrix(sequences):
    mask = sequences[:, None] == sequences[None, :]
    logits = np.where(mask, 0.0, -np.inf)
    return logits

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
        cur = ggml_add(ctx, cur, ggml_view_1d(ctx, etyp, embed_dim, 0), name='embed=tok+typ')
        cur = ggml_add(ctx, cur, ggml_get_rows(ctx, epos, positions), name='embed=tok+typ+pos')
        cur = norm_layer(ctx, cur, tnw, tnb, layer_norm_eps, name='embed_norm')

        # loop over layers
        for i in range(n_layers):
            # get layer tensors
            wq, bq, wk, bk, wv, bv, wao, bao, wan, ban, wu, bu, wd, bd, wln, bln = self.tensors[
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
            lay = attention_layer(ctx, cur, n_heads, mask, wq, bq, wk, bk, wv, bv, name=f'attn{i}')

            # apply attention output and normalize
            lay = linear_layer(ctx, lay, wao, bao, name=f'attn{i}_out')
            lay = norm_layer(ctx, lay, wan, ban, layer_norm_eps, name=f'attn{i}_norm')

            # feed forward network
            lay = linear_layer(ctx, lay, wu, bu, name=f'ffn{i}_up')
            lay = linear_layer(ctx, lay, wd, bd, name=f'ffn{i}_down')

            # add to current tensor and normalize
            cur = ggml_add(ctx, cur, lay, name=f'add{i}')
            cur = norm_layer(ctx, cur, wln, bln, layer_norm_eps, name=f'norm{i}')

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

def test_bert(gguf_path, model_id, batch_size=512):
    import torch
    from transformers import AutoModel

    # load hf model
    hf_model = AutoModel.from_pretrained(model_id)
    hf_tokens = torch.arange(batch_size, dtype=torch.int64).unsqueeze(0)
    hf_embed = hf_model.embeddings(hf_tokens).squeeze(0).detach().numpy()

    # load gguf model
    gg_model = BertModel.from_path(gguf_path, batch_size=batch_size)
    gg_tokens = np.arange(batch_size, dtype=np.int32)
    gg_embed = gg_model.encode(gg_tokens)

    # check results
    match = np.allclose(hf_embed, gg_embed, atol=1e-6)
    print(match)

    # return results
    return (
        gg_model, gg_tokens, gg_embed,
        hf_model, hf_tokens, hf_embed
    )

