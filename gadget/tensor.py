# tensors

import ctypes
import numpy as np

from .ggml import (
    GGMLQuantizationType,
    GGML_QUANT_SIZES,
    ggml_set_name,
    ggml_new_tensor_1d,
    ggml_new_tensor_2d,
    ggml_new_tensor_3d,
    ggml_new_tensor_4d,
)

##
## type conversion
##

ttype_to_ctype = {
    GGMLQuantizationType.F32: ctypes.c_float,
    # GGMLQuantizationType.F16: ctypes.c_half, # not supported by ctypes
    GGMLQuantizationType.Q4_0: ctypes.c_uint8,
    GGMLQuantizationType.Q4_1: ctypes.c_uint8,
    GGMLQuantizationType.Q5_0: ctypes.c_uint8,
    GGMLQuantizationType.Q5_1: ctypes.c_uint8,
    GGMLQuantizationType.Q8_0: ctypes.c_uint8,
    GGMLQuantizationType.Q8_1: ctypes.c_uint8,
    GGMLQuantizationType.Q2_K: ctypes.c_uint8,
    GGMLQuantizationType.Q3_K: ctypes.c_uint8,
    GGMLQuantizationType.Q4_K: ctypes.c_uint8,
    GGMLQuantizationType.Q5_K: ctypes.c_uint8,
    GGMLQuantizationType.Q6_K: ctypes.c_uint8,
    GGMLQuantizationType.Q8_K: ctypes.c_uint8,
    GGMLQuantizationType.IQ2_XXS: ctypes.c_uint8,
    GGMLQuantizationType.IQ2_XS: ctypes.c_uint8,
    GGMLQuantizationType.IQ3_XXS: ctypes.c_uint8,
    GGMLQuantizationType.IQ1_S: ctypes.c_uint8,
    GGMLQuantizationType.IQ4_NL: ctypes.c_uint8,
    GGMLQuantizationType.IQ3_S: ctypes.c_uint8,
    GGMLQuantizationType.IQ2_S: ctypes.c_uint8,
    GGMLQuantizationType.IQ4_XS: ctypes.c_uint8,
    GGMLQuantizationType.I8: ctypes.c_int8,
    GGMLQuantizationType.I16: ctypes.c_int16,
    GGMLQuantizationType.I32: ctypes.c_int32,
    GGMLQuantizationType.I64: ctypes.c_int64,
    GGMLQuantizationType.IQ1_M: ctypes.c_uint8,
    # GGMLQuantizationType.BF16: ctypes.c_bfloat16, # not supported by ctypes
}

ttype_to_dtype = {
    GGMLQuantizationType.F32: np.float32,
    # GGMLQuantizationType.F16: np.float16, # not supported by ctypes
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

def get_quant_shape(tensor):
    ttype = get_tensor_type(tensor)
    shape = get_tensor_shape(tensor)
    dims = len(shape)
    block_size, type_size = GGML_QUANT_SIZES[ttype]
    shape1 = tuple(
        s // block_size if i == dims - 1 else s for i, s in enumerate(shape)
    )
    return shape1

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
