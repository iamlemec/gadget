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
    def __init__(self, hparams, weights, inputs, backend=None):
        def forward(*args):
            return self.forward()
        super().__init__(hparams, weights | inputs, forward, backend=backend)

    @classmethod
    def from_gguf(cls, gguf, backend=None, **kwargs):
        # get hparams (shallow copy)
        hparams = gguf.fields | kwargs

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
            k: (t, [resolve_field(x, hparams) for x in s])
            for k, (t, s) in hints.items()
        }

        # create model and graph
        self = cls(hparams, weights, inputs, backend=backend)

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

def test_model():
    class TestModel(GgmlModel):
        x: Tensor('F32', ('batch_size', 'embed_dim'))

        def forward(self):
            x = self.inputs.x
            for i in range(self.hparams['n_layers']):
                weight, bias = self.inputs[f'weight{i}'], self.inputs[f'bias{i}']
                x = ggml_mul_mat(self.ctx_graph, weight, x, name=f'a{i}')
                x = ggml_add(self.ctx_graph, x, bias, name=f'b{i}')
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
