# gguf + ggml models

import numpy as np
from typing import get_type_hints

from .constants import GGMLQuantizationType
from .ggml import ggml_mul_mat, ggml_add
from .loader import GgufFile
from .compute import GgmlCompute, set_tensor_name

class GgmlModel(GgmlCompute):
    def __init__(self, params, inputs, **kwargs):
        super().__init__(params | inputs, self.forward, **kwargs)

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
    # define model hparams
    input_dim, output_dim, batch_size = 32, 16, 8

    class TestModel(GgmlModel):
        @classmethod
        def from_gguf(cls, gguf, batch_size):
            # get hparams
            input_dim = gguf.get_field('input_dim')
            output_dim = gguf.get_field('output_dim')

            # model inputs
            inputs = dict(
                x = (GGMLQuantizationType.F32, (batch_size, input_dim)),
            )

            # load model (this sets params)
            self = super().from_gguf(gguf, inputs)
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.batch_size = batch_size

            # return model
            return self

        def forward(self, ctx, inp):
            a = ggml_mul_mat(ctx, inp.weight, inp.x, name='a') # [batch_size, output_dim]
            b = ggml_add(ctx, a, inp.bias, name='b') # [batch_size, output_dim]
            return b

    # define true model parameters
    weight = np.ones((output_dim, input_dim), dtype=np.float32)
    bias = np.ones(output_dim, dtype=np.float32)

    # create dummy gguf
    gguf = GgufFile()
    gguf.set_field('name', b'test')
    gguf.set_field('input_dim', input_dim, dtype=np.uint64)
    gguf.set_field('output_dim', output_dim, dtype=np.uint64)
    gguf.set_tensor('weight', weight)
    gguf.set_tensor('bias', bias)

    # load gguf as model
    model = TestModel.from_gguf(gguf, batch_size)

    # compute on input data
    x = np.ones((batch_size, input_dim), dtype=np.float32)
    y = model(x=x)

    # test results
    y0 = (weight @ x.T + bias[:,None]).T
    assert np.allclose(y, y0)

    # return model
    return model
