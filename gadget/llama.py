# llama implementation

import numpy as np

from .ggml import (
    ggml_add_inplace,
    ggml_get_rows,
)
from .layers import (
    linear_layer,
    norm_layer,
    attention_layer,
    feed_forward_layer,
)
from .model import GgmlModel, Parameter, Tensor

##
## bert model
##

class LlamaModel(GgmlModel):
    batch_size: Parameter('llama.context_length')
    tokens    : Tensor('I32', ('batch_size',))
    attention : Tensor('F32', ('batch_size', 'batch_size'))

    # perform param validation here
    def __init__(self, params, tensors, **kwargs):
        # validate batch_size
        if (bs := params['batch_size']) > (cl := params['llama.context_length']):
            raise ValueError('batch_size ({bs}) > context_length ({cl})')

        # pass to model constructor
        super().__init__(params, tensors, **kwargs)

    # llama model function
    def forward(self):
        ctx = self.ctx_graph

        # get params
        n_layers, n_heads_q, n_heads_kv, layer_norm_rms_eps = self.params[
            'llama.block_count'            , 'llama.attention.head_count'            ,
            'llama.attention.head_count_kv', 'llama.attention.layer_norm_rms_epsilon',
        ]

        # get embed tensors
        etok = self.tensors['token_embd.weight']

        # get input tensors
        tokens, attention = self.tensors['tokens', 'attention']

        # get token embeddings
        cur = ggml_get_rows(ctx, etok, tokens, name='embed=tok')

        # loop over layers
        for i in range(n_layers):
            last = cur

            # get layer tensors
            wq, wk, wv, wo, wan, wu, wd, wg, wn, = self.tensors[
                f'blk.{i}.attn_q.weight'     , f'blk.{i}.attn_k.weight'   , f'blk.{i}.attn_v.weight'  ,
                f'blk.{i}.attn_output.weight', f'blk.{i}.attn_norm.weight', f'blk.{i}.ffn_up.weight'  ,
                f'blk.{i}.ffn_down.weight'   , f'blk.{i}.ffn_gate.weight' , f'blk.{i}.ffn_norm.weight',
            ]

            # get attention interactions
            att = norm_layer(ctx, cur, wan, eps=layer_norm_rms_eps, inplace=True, name=f'attn{i}_norm')
            att = attention_layer(
                ctx, att, n_heads_q, attention, wq, wk, wv, wo,
                n_heads_kv=n_heads_kv, eps=layer_norm_rms_eps, name=f'attn{i}'
            )

            # add layer input to attention
            att = ggml_add_inplace(ctx, att, last)

            # feed forward network on current
            cur = norm_layer(ctx, att, wn, eps=layer_norm_rms_eps, name=f'ffn{i}_norm')
            cur = feed_forward_layer(ctx, cur, wu, wd, act='gelu', name=f'ffn{i}')

            # add attention output to current tensor
            cur = ggml_add_inplace(ctx, cur, att)

        # get output tensors
        ow, onw = self.tensors['output.weight', 'output_norm.weight']

        # generate output
        cur = norm_layer(ctx, cur, onw, eps=layer_norm_rms_eps, inplace=True, name='output_norm')
        cur = linear_layer(ctx, cur, ow, name='output')

        # return logits
        return cur
