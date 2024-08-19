# tensors

import ctypes
import numpy as np

from .ggml import (
    GGMLQuantizationType,
    ggml_set_name,
    ggml_new_tensor_1d,
    ggml_new_tensor_2d,
    ggml_new_tensor_3d,
    ggml_new_tensor_4d,
    ggml_internal_get_type_traits,
)

##
## type conversion
##

ttype_to_dtype = {
    GGMLQuantizationType.F32: np.float32,
    GGMLQuantizationType.F16: np.float16,
    GGMLQuantizationType.Q4_0: np.uint8,
    GGMLQuantizationType.Q4_1: np.uint8,
    GGMLQuantizationType.Q5_0: np.uint8,
    GGMLQuantizationType.Q5_1: np.uint8,
    GGMLQuantizationType.Q8_0: np.uint8,
    GGMLQuantizationType.Q8_1: np.uint8,
    GGMLQuantizationType.Q2_K: np.uint8,
    GGMLQuantizationType.Q3_K: np.uint8,
    GGMLQuantizationType.Q4_K: np.uint8,
    GGMLQuantizationType.Q5_K: np.uint8,
    GGMLQuantizationType.Q6_K: np.uint8,
    GGMLQuantizationType.Q8_K: np.uint8,
    GGMLQuantizationType.IQ2_XXS: np.uint8,
    GGMLQuantizationType.IQ2_XS: np.uint8,
    GGMLQuantizationType.IQ3_XXS: np.uint8,
    GGMLQuantizationType.IQ1_S: np.uint8,
    GGMLQuantizationType.IQ4_NL: np.uint8,
    GGMLQuantizationType.IQ3_S: np.uint8,
    GGMLQuantizationType.IQ2_S: np.uint8,
    GGMLQuantizationType.IQ4_XS: np.uint8,
    GGMLQuantizationType.I8: np.int8,
    GGMLQuantizationType.I16: np.int16,
    GGMLQuantizationType.I32: np.int32,
    GGMLQuantizationType.I64: np.int64,
    GGMLQuantizationType.F64: np.float64,
    GGMLQuantizationType.IQ1_M: np.uint8,
    # GGMLQuantizationType.BF16: np.bfloat16, # not supported by ctypes
}

##
## tensor functions
##

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

def get_tensor_shape(tensor):
    value = tensor.contents
    nelem = tuple(value.ne[:4])
    return trim_nelem(nelem)[::-1]

def get_tensor_type(tensor):
    value = tensor.contents
    return GGMLQuantizationType(value.type)

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
    dtype = ttype_to_dtype[ttype]
    dtype_size = np.dtype(dtype).itemsize
    block_size, type_size = get_type_traits(ttype)
    dims = len(shape)
    dshape = tuple(
        (s // block_size) * (type_size // dtype_size) if i == dims - 1 else s
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
