# tensors

import ctypes
import numpy as np

from .ggml import (
    GGMLQuantizationType as T,
    ggml_set_name,
    ggml_new_tensor_1d,
    ggml_new_tensor_2d,
    ggml_new_tensor_3d,
    ggml_new_tensor_4d,
    ggml_internal_get_type_traits,
    ggml_backend_buffer_is_host,
)

##
## type conversion
##

ttype_to_ntype = {
    T.F32: 'float32',
    T.F16: 'float16',
    T.Q4_0: 'uint8',
    T.Q4_1: 'uint8',
    T.Q5_0: 'uint8',
    T.Q5_1: 'uint8',
    T.Q8_0: 'uint8',
    T.Q8_1: 'uint8',
    T.Q2_K: 'uint8',
    T.Q3_K: 'uint8',
    T.Q4_K: 'uint8',
    T.Q5_K: 'uint8',
    T.Q6_K: 'uint8',
    T.Q8_K: 'uint8',
    T.IQ2_XXS: 'uint8',
    T.IQ2_XS: 'uint8',
    T.IQ3_XXS: 'uint8',
    T.IQ1_S: 'uint8',
    T.IQ4_NL: 'uint8',
    T.IQ3_S: 'uint8',
    T.IQ2_S: 'uint8',
    T.IQ4_XS: 'uint8',
    T.I8: 'int8',
    T.I16: 'int16',
    T.I32: 'int32',
    T.I64: 'int64',
    T.F64: 'float64',
    T.IQ1_M: 'uint8',
    T.BF16: 'bfloat16',
}

ttype_to_dtype = {
    k: getattr(np, v) for k, v in ttype_to_ntype.items() if hasattr(np, v)
}

ntype_width = {
    'uint8': 1,
    'int8': 1,
    'int16': 2,
    'int32': 4,
    'int64': 8,
    'float16': 2,
    'float32': 4,
    'float64': 8,
    'bfloat16': 2,
}

##
## array functions (array framework agnostic)
##

def get_framework(framework):
    if framework == 'numpy':
        lib = np
    elif framework == 'torch':
        import torch
        lib = torch
    else:
        raise ValueError(f'unknown array framework {library}')
    return lib

def create_array(ntype, shape, framework='numpy', device='cpu'):
    fw = get_framework(framework)
    if not hasattr(fw, ntype):
        raise ValueError(f'dtype {ntype} not supported by framework {framework}')
    dtype = getattr(fw, ntype)
    array = fw.empty(shape, dtype=dtype, device=device)
    return array

def get_array_ntype(array):
    return str(array.dtype).removeprefix('torch.')

def get_array_data(array):
    if hasattr(array, 'ctypes'):
        return array.ctypes.data
    elif hasattr(array, 'data_ptr'):
        return array.data_ptr()
    else:
        raise TypeError(f'unknown array type {type(array)}')

##
## tensor functions
##

def is_half(ttype):
    return ttype in (T.F16, T.BF16)

def trim_nelem(shape):
    dims = 1 + max([
        i for i, d in enumerate(shape) if d > 1
    ], default=0)
    return shape[:dims]

def get_type_traits(ttype):
    traits = ggml_internal_get_type_traits(ttype)
    return traits.blck_size, traits.type_size

def get_tensor_name(tensor):
    value = tensor.contents
    return value.name.decode('utf-8')

def get_tensor_is_host(tensor):
    buffer = tensor.contents.buffer
    is_host = ggml_backend_buffer_is_host(buffer)
    return is_host

def get_tensor_shape(tensor):
    value = tensor.contents
    nelem = tuple(value.ne[:4])
    return trim_nelem(nelem)[::-1]

def get_tensor_type(tensor):
    value = tensor.contents
    return T(value.type)

def get_tensor_info(tensor):
    name = get_tensor_name(tensor)
    ttype = get_tensor_type(tensor)
    shape = get_tensor_shape(tensor)
    stat = f'{name}: {ttype.name} × {shape}'
    return stat

def get_block_shape(tensor):
    ttype = get_tensor_type(tensor)
    shape = get_tensor_shape(tensor)
    block_size, type_size = get_type_traits(ttype)
    dims = len(shape)
    bshape = tuple(
        s // block_size if i == dims - 1 else s for i, s in enumerate(shape)
    )
    return bshape

def get_data_shape(tensor):
    ttype = get_tensor_type(tensor)
    shape = get_tensor_shape(tensor)
    ntype = ttype_to_ntype[ttype]
    ntype_size = ntype_width[ntype]
    block_size, type_size = get_type_traits(ttype)
    dims = len(shape)
    dshape = tuple(
        (s // block_size) * (type_size // ntype_size) if i == dims - 1 else s
        for i, s in enumerate(shape)
    )
    return dshape

##
## tensor creation
##

# dispatch create functions
create_funcs = {
    1: ggml_new_tensor_1d,
    2: ggml_new_tensor_2d,
    3: ggml_new_tensor_3d,
    4: ggml_new_tensor_4d,
}

# we reverse shape to match numpy convention
def create_tensor(ctx, typ, shp, nam=None):
    if (dims := len(shp)) not in create_funcs:
        raise ValueError(f'unsupported shape: {shp}')
    tensor = create_funcs[dims](ctx, typ, *shp[::-1])
    if nam is not None:
        ggml_set_name(tensor, nam.encode('utf-8'))
    return tensor

def set_tensor_name(tensor, name):
    ggml_set_name(tensor, name.encode('utf-8'))
