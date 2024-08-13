# gguf + ggml models

import numpy as np
from typing import get_type_hints

from .ggml import GGMLQuantizationType, ggml_mul_mat, ggml_add
from .loader import GgufFile
from .compute import GgmlCompute, set_tensor_name

class GgmlModel(GgmlCompute):
    def __init__(self, params, inputs, backend=None, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        super().__init__(params | inputs, self.forward, backend=backend)

    @classmethod
    def from_gguf(cls, gguf, inputs, **kwargs):
        # extract params metadata from gguf
        params = {
            key: (ttype, tensor.shape)
            for key, (ttype, tensor) in gguf.tensors.items()
        }

        # create model and graph
        self = cls(params, inputs, **kwargs)

        # assign tensors on backend
        for name, (ttype, tensor) in gguf.tensors.items():
            self.set_input(name, tensor)

        # return model
        return self

    @classmethod
    def from_path(cls, path, *args, **kwargs):
        gguf = GgufFile.from_path(path)
        return cls.from_gguf(gguf, *args, **kwargs)

    def forward(self, ctx, inputs):
        raise NotImplementedError('forward method must be implemented')

def test_model():
    class TestModel(GgmlModel):
        @classmethod
        def from_gguf(cls, gguf, batch_size):
            # get hparams
            n_layers = gguf.get_field('n_layers')
            input_dim = gguf.get_field('embed_dim')

            # model inputs
            inputs = dict(
                x = (GGMLQuantizationType.F32, (batch_size, embed_dim)),
            )

            # load model (this sets params)
            return super().from_gguf(
                gguf, inputs, n_layers=n_layers, embed_dim=embed_dim
            )

        def forward(self, ctx, inp):
            x = inp.x
            for i in range(self.n_layers):
                x = ggml_mul_mat(ctx, inp[f'weight{i}'], x, name=f'a{i}')
                x = ggml_add(ctx, x, inp[f'bias{i}'], name=f'b{i}')
            return x

    # define model hparams
    n_layers, embed_dim, batch_size = 50, 32, 16

    # create dummy gguf
    gguf = GgufFile()
    gguf.set_field('name', b'test')
    gguf.set_field('n_layers', n_layers, dtype=np.uint64)
    gguf.set_field('embed_dim', embed_dim, dtype=np.uint64)

    # add layers
    for i in range(n_layers):
        weight = np.random.randn(embed_dim, embed_dim).astype(np.float32)
        gguf.set_tensor(f'weight{i}', weight)
    for i in range(n_layers):
        bias = np.random.randn(embed_dim).astype(np.float32)
        gguf.set_tensor(f'bias{i}', bias)

    # load gguf as model
    model = TestModel.from_gguf(gguf, batch_size)

    # compute on input data
    x = np.random.randn(batch_size, embed_dim).astype(np.float32)
    output_np = model(x=x)

    # return model
    return model
