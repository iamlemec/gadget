# gguf + ggml models

import numpy as np
from typing import get_type_hints

from .ggml import GGMLQuantizationType
from .tensor import set_tensor_name
from .loader import GgufFile
from .compute import GgmlCompute

##
## type decorators
##

class Parameter:
    def __init__(self, field):
        self.field = field

class Tensor:
    def __init__(self, ttype, shape):
        if type(ttype) is str:
            ttype = GGMLQuantizationType[ttype]
        self.ttype = ttype
        self.shape = shape

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
    def __init__(self, params, tensors, backend=None):
        super().__init__(params, tensors, freeze(self.forward), backend=backend)

    @classmethod
    def from_gguf(cls, gguf, backend=None, **params):
        # get metadata from gguf
        weights = {
            key: (ttype, shape)
            for key, (ttype, shape, array) in gguf.tensors.items()
        }

        # get type hints for model
        hints = get_type_hints(cls)

        # sub in default parameters
        params0 = {
            k: gguf.get_field(v.field)
            for k, v in hints.items() if type(v) is Parameter
        }

        # resolve string fields
        inputs = {
            k: (t.ttype, [resolve_field(x, params, params0, gguf.fields) for x in t.shape])
            for k, t in hints.items() if type(t) is Tensor
        }

        # create model and graph
        self = cls(gguf.fields | params0 | params, weights | inputs, backend=backend)

        # assign tensors on backend
        for name, (ttype, shape, tensor) in gguf.tensors.items():
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

def test_linear(input_dim=64, output_dim=32, batch_size=16):
    from .ggml import ggml_mul_mat, ggml_add

    # simple model interface
    class TestModel(GgmlModel):
        # strings dimensions are filled in dynamically
        x: Tensor('F32', ('batch_size', 'input_dim'))

        def forward(self):
            # get contexts and inputs
            ctx = self.ctx_graph
            a, b, x = self.tensors['a', 'b', 'x']

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
    match = np.allclose(y_np, y0_np, atol=1e-5)

    # return result
    return match

def test_getrows(output_dim=32, vocab_size=1024, batch_size=16):
    from .ggml import ggml_get_rows

    # simple model interface
    class TestModel(GgmlModel):
        # strings dimensions are filled in dynamically
        x: Tensor('I32', ('batch_size',))

        def forward(self):
            # get contexts and inputs
            ctx = self.ctx_graph
            m, x = self.tensors['m', 'x']

            # apply function
            x1 = ggml_get_rows(ctx, m, x, name=f'x1')

            # return result
            return x1

    # generate weights
    m_np = np.random.randn(vocab_size, output_dim).astype(np.float32)

    # create dummy gguf
    gguf = GgufFile()
    gguf.set_field('name', b'test')
    gguf.set_field('vocab_size', vocab_size, dtype=np.int64)
    gguf.set_field('output_dim', output_dim, dtype=np.int64)
    gguf.set_tensor('m', m_np)

    # load gguf as model
    model = TestModel.from_gguf(gguf, batch_size=batch_size)

    # compute on input data
    x_np = np.random.randint(0, vocab_size, size=(batch_size,), dtype=np.int32)
    y_np = model(x=x_np)

    # get numpy results
    y0_np = m_np.take(x_np, axis=0)
    match = np.allclose(y_np, y0_np, atol=1e-5)

    # return result
    return match
