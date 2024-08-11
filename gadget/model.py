# gguf + ggml models

import numpy as np

from .constants import GGMLQuantizationType
from .ggml import ggml_mul_mat, ggml_add
from .loader import GgufFile
from .compute import GgmlCompute, get_tensor_info, set_tensor_name

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
            # ggml uses reverse shape convension from numpy
            self.set_input(name, tensor)

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
    weight = np.ones((output_dim, input_dim), dtype=np.float32)
    bias = np.ones((output_dim,), dtype=np.float32)

    # model gguf
    gguf = GgufFile()
    gguf.set_field('name', b'test')
    gguf.set_field('input_dim', input_dim, dtype=np.uint64)
    gguf.set_field('output_dim', output_dim, dtype=np.uint64)
    gguf.set_tensor('weight', weight)
    gguf.set_tensor('bias', bias)

    # model inputs
    inputs = dict(
        x = (GGMLQuantizationType.F32, (batch_size, input_dim)),
    )

    # model function (comments are ggml shapes)
    def forward(ctx, inputs):
        # load params
        weight = inputs['weight'] # [input_dim, output_dim]
        bias = inputs['bias'] # [output_dim]

        # load inputs
        x = inputs['x'] # [input_dim, batch_size]

        # do computation
        a = ggml_mul_mat(ctx, weight, x) # [output_dim, batch_size]
        b = ggml_add(ctx, a, bias) # [output_dim, batch_size]

        # set tensor names
        set_tensor_name(a, 'a')
        set_tensor_name(b, 'b')

        # return results
        return b

    # load model (this sets params)
    model = GgmlModel.from_gguf(gguf, inputs, forward)
    print(model)

    # compute on input data
    x = np.ones((batch_size, input_dim), dtype=np.float32)
    y = model.compute(x=x)

    # test results
    y0 = (weight @ x.T + bias[:,None]).T
    assert np.allclose(y, y0)
