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

def freeze(func):
    def wrapper(*args, **kwargs):
        return func()
    return wrapper

class GgmlModel(GgmlCompute):
    def __init__(self, params, weights, inputs, backend=None):
        super().__init__(
            params, weights | inputs, freeze(self.forward), backend=backend
        )

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

def test_model(input_dim=64, output_dim=32, batch_size=16):
    # simple model interface
    class TestModel(GgmlModel):
        # strings dimensions are filled in dynamically
        x: Tensor('F32', ('batch_size', 'input_dim'))

        def forward(self):
            # get contexts and inputs
            ctx, x, a, b = (
                self.ctx_graph, self.tensors['x'], 
                self.tensors['a'], self.tensors['b']
            )

            # apply function
            x1 = ggml_mul_mat(ctx, a, x, name=f'x1')
            x2 = ggml_add(ctx, x1, b, name=f'x2')

            # return result
            return x2

    # generate weights
    a_np = np.random.randn(output_dim, input_dim).astype(np.float32)
    b_np = np.random.randn(output_dim).astype(np.float32)

    # create dummy gguf
    gguf = GgufFile()
    gguf.set_field('name', b'test')
    gguf.set_field('input_dim', input_dim, dtype=np.int64)
    gguf.set_field('output_dim', output_dim, dtype=np.int64)
    gguf.set_tensor('a', a_np)
    gguf.set_tensor('b', b_np)

    # load gguf as model
    model = TestModel.from_gguf(gguf, batch_size=batch_size)

    # compute on input data
    x_np = np.random.randn(batch_size, input_dim).astype(np.float32)
    y_np = model(x=x_np)

    # get numpy results
    y0_np = (x_np @ a_np.T) + b_np[None,:]
    np.allclose(y_np, y0_np)

    # return model
    return model
