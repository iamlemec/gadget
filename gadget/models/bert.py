# bert implementation

import numpy as np

from ..ggml import (
    ggml_add,
    ggml_add_inplace,
    ggml_get_rows,
    ggml_view_1d,
    ggml_view_2d,
    ggml_cont,
    ggml_element_size,
)
from ..model import GgmlModel, Parameter, State, Tensor
from .layers import (
    linear_layer,
    norm_layer,
    attention_layer,
    feed_forward_layer,
)

##
## bert model
##

class BertModel(GgmlModel):
    batch_size: Parameter('bert.context_length')
    n_tokens  : State(None)
    tokens    : Tensor('I32', ('batch_size',))
    positions : Tensor('I32', ('batch_size',))
    mask     : Tensor('F32', ('batch_size', 'batch_size'))

    # perform param validation here
    def __init__(self, params, tensors, states, **kwargs):
        # validate batch_size
        if (bs := params['batch_size']) > (cl := params['bert.context_length']):
            raise ValueError('batch_size ({bs}) > context_length ({cl})')

        # pass to model constructor
        super().__init__(params, tensors, states, **kwargs)

    # update state with number of tokens
    # NOTE: this still needs inputs of size batch_size
    #       because we can't set 2d input tensor slices yet
    def __call__(self, tokens, positions, mask, n_tokens=None):
        self.state['n_tokens'] = n_tokens if n_tokens is not None else len(tokens)
        return super().__call__(tokens=tokens, positions=positions, mask=mask)

    # bert model function
    def forward(self):
        ctx = self.ctx_graph

        # get params
        n_layers, n_heads, embed_dim, batch_size, layer_norm_eps = self.params[
            'bert.block_count'     , 'bert.attention.head_count',
            'bert.embedding_length', 'batch_size'               ,
            'bert.attention.layer_norm_epsilon'
        ]

        # get weights
        etok, etyp, epos, tnw, tnb = self.tensors[
            'token_embd.weight', 'token_types.weight', 'position_embd.weight',
            'token_embd_norm.weight', 'token_embd_norm.bias',
        ]

        # get state
        n_tokens = self.state['n_tokens']

        # get input tensors
        tokens, positions, mask = self.tensors['tokens', 'positions', 'mask']

        # get just this batch of tokens
        mask_stride = ggml_element_size(mask) * batch_size
        tokens = ggml_view_1d(ctx, tokens, n_tokens, 0, name='tokens_batch')
        positions = ggml_view_1d(ctx, positions, n_tokens, 0, name='positions_batch')
        mask = ggml_cont(ctx, ggml_view_2d(ctx, mask, n_tokens, n_tokens, mask_stride, 0, name='mask_batch'))

        # get token embeddings (token+type+position+norm)
        cur = ggml_get_rows(ctx, etok, tokens, name='embed=tok')
        cur = ggml_add_inplace(ctx, cur, ggml_view_1d(ctx, etyp, embed_dim, 0), name='embed=tok+typ')
        cur = ggml_add_inplace(ctx, cur, ggml_get_rows(ctx, epos, positions), name='embed=tok+typ+pos')
        cur = norm_layer(ctx, cur, tnw, tnb, eps=layer_norm_eps, inplace=True, name='embed_norm')

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
                ctx, cur, n_heads, mask, wq, wk, wv, wo, bq=bq, bk=bk, bv=bv, bo=bo,
                eps=layer_norm_eps, name=f'attn{i}'
            )

            # add attention output to current then normalize
            att = ggml_add_inplace(ctx, cur, att)
            att = norm_layer(ctx, att, wan, ban, eps=layer_norm_eps, inplace=True, name=f'attn{i}_norm')

            # feed forward network on current
            cur = feed_forward_layer(ctx, att, wu, wd, bu=bu, bd=bd, act='gelu', name=f'ffn{i}')

            # add attention output to current tensor and normalize
            cur = ggml_add_inplace(ctx, cur, att, name=f'add{i}')
            cur = norm_layer(ctx, cur, wln, bln, eps=layer_norm_eps, inplace=True, name=f'norm{i}')

        # return embeddings
        return cur
