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
from .model import GgmlModel

##
## bert model
##

class BertModel(GgmlModel):
    @classmethod
    def from_gguf(cls, gguf, batch_size=None):
        # get hparams
        if batch_size is None:
            batch_size = gguf.get_field('bert.context_length')
        embed_dim = gguf.get_field('bert.embedding_length')

        # model inputs
        inputs = dict(
            tokens = (GGMLQuantizationType.I32, (batch_size,)),
            positions = (GGMLQuantizationType.I32, (batch_size,)),
        )

        # load model (this sets params)
        self = super().from_gguf(
            gguf, inputs, batch_size=batch_size, embed_dim=embed_dim
        )

        # return model
        return self

    # model function (comments are numpy shapes)
    def forward(self, ctx, inp):
        # get token embeddings
        embed = ggml_get_rows(
            ctx, inp['token_embd.weight'], inp.tokens, name='embed'
        ) # [batch_size, embed_dim]

        # get token type embeddings
        embed = ggml_add_inplace(ctx, embed, ggml_view_1d(
            ctx, inp['token_types.weight'], self.embed_dim, 0, name='typ_embed'
        ))

        # get positional embeddings
        embed = ggml_add_inplace(ctx, embed, ggml_get_rows(
            ctx, inp['position_embd.weight'], inp.positions, name='pos_embed'
        ))

        # return embedding
        return embed

    def embed(self, tokens):
        # validate input tokens
        tokens = np.asarray(tokens, dtype=np.int32)
        if tokens.shape != (self.batch_size,):
            raise ValueError('tokens must be an array of shape (batch_size,)')

        # generate token positions
        positions = np.arange(self.batch_size, dtype=np.int32)

        # compute on input data
        embed = self.compute(tokens=tokens, positions=positions)

        # return embedding
        return embed
