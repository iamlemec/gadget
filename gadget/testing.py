# simple ggml test

import ctypes
import numpy as np

from .lib import (
    ggml_type_size, ggml_tensor_overhead, ggml_graph_overhead,
    ggml_init_params, ggml_init, ggml_new_tensor_2d,
    GGML_TYPE_F16, GGML_TYPE_F32
)

gtype_to_ctype = {
    GGML_TYPE_F32: ctypes.c_float,
}

def tensor_to_numpy(tensor, dims=2):
    assert dims <= 4
    assert tensor.contents.type in gtype_to_ctype
    ctype = gtype_to_ctype[tensor.contents.type]
    shape = tuple(tensor.contents.ne[:dims])[::-1]
    p = ctypes.cast(tensor.contents.data, ctypes.POINTER(ctype))
    return np.ctypeslib.as_array(p, shape=shape)

def test_ggml_ctx():
    # tensor shapes
    rows_A, cols_A = 4, 2
    rows_B, cols_B = 3, 2

    # compute memory requirements
    mem_usage = (
        ggml_type_size(GGML_TYPE_F32) * (rows_A * cols_A) +
        ggml_type_size(GGML_TYPE_F32) * (rows_B * cols_B) +
        ggml_tensor_overhead() +
        ggml_graph_overhead() +
        1024
    )
    print(mem_usage)

    # initialize ggml
    ggml_params = ggml_init_params(mem_usage, None, False)
    ggml_ctx = ggml_init(ggml_params)

    # create tensors
    a = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_F32, cols_A, rows_A)
    b = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_F32, cols_B, rows_B)
    print(a.contents.data, b.contents.data)

    # create numpy arrays
    a_np = tensor_to_numpy(a)
    b_np = tensor_to_numpy(b)

    # fill arrays
    a_np[:,:] = [[2 , 8], [5, 1], [4, 2], [8, 6]]
    b_np[:,:] = [[10, 5], [9, 9], [5, 4]]

    return a_np, b_np
