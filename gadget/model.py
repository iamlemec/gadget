# gguf + ggml models

import numpy as np

from .constants import GGMLQuantizationType
from .ggml import ggml_mul_mat, ggml_add
from .loader import GgufFile
from .compute import GgmlCompute

class GgmlModel(GgmlCompute):
    def __init__(self, params, inputs, model, backend=None):
        # merge inputs and params
        inputs_all = {**params, **inputs}
        super().__init__(inputs_all, model, backend=backend)

    @classmethod
    def from_gguf(cls, gguf, inputs, model, backend=None):
        # extract params metadata from gguf
        params = {
            key: (ttype, tensor.shape)
            for key, (ttype, tensor) in gguf.tensors.items()
        }

        # create model and graph
        self = cls(params, inputs, model, backend=backend)
    
        # assign tensors on backend
        for name, (ttype, tensor) in gguf.tensors.items():
            self.set_input(name, tensor.T)

        # return model
        return self

    @classmethod
    def from_file(cls, filename, inputs, model, backend=None):
        gguf = GgufFile.from_file(filename)
        return cls.from_gguf(gguf, inputs, model, backend=backend)

    def set_params(self, values):
        for name, value in values.items():
            self.set_input(name, value)

def test_model():
    # model metadata
    input_dim, output_dim, batch_size = 32, 16, 8

    # model params
    weight = np.ones((input_dim, output_dim), dtype=np.float32)
    bias = np.ones((output_dim, 1), dtype=np.float32)

    # model gguf
    gguf = GgufFile()
    gguf.set_field('name', b'test')
    gguf.set_field('input_dim', np.uint64(input_dim))
    gguf.set_field('output_dim', np.uint64(output_dim))
    gguf.set_tensor('weight', weight)
    gguf.set_tensor('bias', bias)

    # model inputs
    inputs = {
        'x': (GGMLQuantizationType.F32, (input_dim, batch_size))
    }

    # model function
    def forward(ctx, inputs):
        weight = inputs['weight'] # [input_dim, output_dim]
        bias = inputs['bias'] # [output_dim, 1]
        x = inputs['x'] # [input_dim, batch_size]
        a = ggml_mul_mat(ctx, weight, x) # [batch_size, output_dim]
        b = ggml_add(ctx, a, bias) # [batch_size, output_dim]
        return b

    # load model (this sets params)
    model = GgmlModel.from_gguf(gguf, inputs, forward)

    # define input data
    x = np.ones((input_dim, batch_size), dtype=np.float32)
    data = {'x': x.T}

    # compute
    y = model.compute(data)

    # test results
    y0 = x.T @ weight + bias.T
    assert np.allclose(y, y0)

    # return model and data
    return weight, bias, x, y
