# bert implementation

import numpy as np

from .ggml import (
    GGMLQuantizationType,
    ggml_get_rows,
    ggml_add,
    ggml_add_inplace,
    ggml_view_1d,
)
from .loader import GgufFile
from .compute import set_tensor_name
from .model import GgmlModel, Tensor

##
## bert model
##

class BertModel(GgmlModel):
    tokens   : Tensor('I32', ('batch_size',))
    positions: Tensor('I32', ('batch_size',))

    # model function (comments are numpy shapes)
    def forward(self):
        ctx = self.ctx_graph

        # get hparams
        embed_dim = self.params['bert.embedding_length']

        # get weights
        token_embd = self.tensors['token_embd.weight']
        token_types = self.tensors['token_types.weight']
        position_embd = self.tensors['position_embd.weight']

        # get inputs
        tokens = self.tensors['tokens']
        positions = self.tensors['positions']

        # get token embeddings
        embed = ggml_get_rows(
            ctx, token_embd, tokens, name='embed'
        ) # [batch_size, embed_dim]

        # get token type embeddings
        embed = ggml_add_inplace(ctx, embed, ggml_view_1d(
            ctx, token_types, embed_dim, 0, name='typ_embed'
        ))

        # get positional embeddings
        embed = ggml_add_inplace(ctx, embed, ggml_get_rows(
            ctx, position_embd, positions, name='pos_embed'
        ))

        # return embedding
        return embed

    def embed(self, tokens):
        batch_size = self.params['batch_size']

        # validate input tokens
        tokens = np.asarray(tokens, dtype=np.int32)
        if tokens.shape != (batch_size,):
            raise ValueError('tokens must be an array of shape (batch_size,)')

        # generate token positions
        positions = np.arange(batch_size, dtype=np.int32)

        # compute on input data
        embed = self.compute(tokens=tokens, positions=positions)

        # return embedding
        return embed

def test_bert(gguf_path, batch_size=512):
    # load model
    model = BertModel.from_path(gguf_path, batch_size=batch_size)

    # make some tokens
    tokens = np.arange(batch_size, dtype=np.int32)

    # embed tokens
    embed = model.embed(tokens)

    # return model
    return model
