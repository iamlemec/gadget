# gguf + ggml models

import numpy as np
from typing import get_type_hints

from .ggml import GGMLQuantizationType, ggml_mul_mat, ggml_add
from .loader import GgufFile
from .compute import GgmlCompute, set_tensor_name

##
## type decorators
##

class Tensor:
    def __init__(self, ttype, shape):
        if type(ttype) is str:
            ttype = GGMLQuantizationType[ttype]
        self.ttype = ttype
        self.shape = shape

    def to_tuple(self):
        return self.ttype, self.shape

def resolve_field(key, *dicts):
    if type(key) is str:
        for d in dicts:
            if key in d:
                return d[key]
        raise KeyError(f'key {key} not found')
    else:
        return key

##
## model interface
##

class GgmlModel(GgmlCompute):
    def __init__(self, params, weights, inputs, backend=None):
        def forward(*args):
            return self.forward()
        super().__init__(params, weights | inputs, forward, backend=backend)

    @classmethod
    def from_gguf(cls, gguf, backend=None, **kwargs):
        # get hparams (shallow copy)
        params = gguf.fields | kwargs

        # get metadata from gguf
        weights = {
            key: (ttype, tensor.shape)
            for key, (ttype, tensor) in gguf.tensors.items()
        }

        # get type hints for model
        hints = {
            k: v.to_tuple() for k, v in get_type_hints(cls).items()
        }

        # resolve string fields
        inputs = {
            k: (t, [resolve_field(x, params) for x in s])
            for k, (t, s) in hints.items()
        }

        # create model and graph
        self = cls(params, weights, inputs, backend=backend)

        # assign tensors on backend
        for name, (ttype, tensor) in gguf.tensors.items():
            self.set_input(name, tensor)

        # return model
        return self

    @classmethod
    def from_path(cls, path, *args, **kwargs):
        gguf = GgufFile.from_path(path)
        return cls.from_gguf(gguf, *args, **kwargs)

    def forward(self):
        raise NotImplementedError('forward method must be implemented')

##
## testing
##

def test_model(n_layers=50, embed_dim=32, batch_size=16):
    # simple model interface
    class TestModel(GgmlModel):
        # strings dimensions are filled in dynamically
        x: Tensor('F32', ('batch_size', 'embed_dim'))

        def forward(self):
            # get contexta and inputs
            ctx, x = self.ctx_graph, self.tensors['x']
            n_layers = self.params['n_layers']

            # loop through layers
            for i in range(n_layers):
                # get layer weights
                weight, bias = self.tensors[f'weight{i}'], self.tensors[f'bias{i}']

                # apply layer function
                x = ggml_mul_mat(ctx, weight, x, name=f'a{i}')
                x = ggml_add(ctx, x, bias, name=f'b{i}')

            # return final embed
            return x

    # create dummy gguf
    gguf = GgufFile()
    gguf.set_field('name', b'test')
    gguf.set_field('n_layers', n_layers, dtype=np.uint64)
    gguf.set_field('embed_dim', embed_dim, dtype=np.uint64)

    # add layers
    for i in range(n_layers):
        weight = np.random.randn(embed_dim, embed_dim).astype(np.float32)
        bias = np.random.randn(embed_dim).astype(np.float32)
        gguf.set_tensor(f'weight{i}', weight)
        gguf.set_tensor(f'bias{i}', bias)

    # load gguf as model
    model = TestModel.from_gguf(gguf, batch_size=batch_size)

    # compute on input data
    x = np.random.randn(batch_size, embed_dim).astype(np.float32)
    output_np = model(x=x)

    # return model
    return model
