# high leve gguf interface

import numpy as np
from operator import itemgetter

from .constants import (
    GGUF_MAGIC, GGUF_VERSION, GGUF_DEFAULT_ALIGNMENT,
    GGUFValueType, GGMLQuantizationType
)

# map for scalar types
gtype_to_dtype = {
    GGUFValueType.UINT8  : np.uint8  ,
    GGUFValueType.INT8   : np.int8   ,
    GGUFValueType.UINT16 : np.uint16 ,
    GGUFValueType.INT16  : np.int16  ,
    GGUFValueType.UINT32 : np.uint32 ,
    GGUFValueType.INT32  : np.int32  ,
    GGUFValueType.FLOAT32: np.float32,
    GGUFValueType.UINT64 : np.uint64 ,
    GGUFValueType.INT64  : np.int64  ,
    GGUFValueType.FLOAT64: np.float64,
    GGUFValueType.BOOL   : np.bool_  ,
}

# map for tensor types (missing bfloat16)
ttype_to_dtype = {
    GGMLQuantizationType.F16    : np.float16,
    GGMLQuantizationType.F32    : np.float32,
    GGMLQuantizationType.F64    : np.float64,
    GGMLQuantizationType.I8     : np.int8   ,
    GGMLQuantizationType.I16    : np.int16  ,
    GGMLQuantizationType.I32    : np.int32  ,
    GGMLQuantizationType.I64    : np.int64  ,
    GGMLQuantizationType.Q4_1   : np.uint8  ,
    GGMLQuantizationType.Q5_0   : np.uint8  ,
    GGMLQuantizationType.Q5_1   : np.uint8  ,
    GGMLQuantizationType.Q8_0   : np.uint8  ,
    GGMLQuantizationType.Q8_1   : np.uint8  ,
    GGMLQuantizationType.Q2_K   : np.uint8  ,
    GGMLQuantizationType.Q3_K   : np.uint8  ,
    GGMLQuantizationType.Q4_K   : np.uint8  ,
    GGMLQuantizationType.Q5_K   : np.uint8  ,
    GGMLQuantizationType.Q6_K   : np.uint8  ,
    GGMLQuantizationType.Q8_K   : np.uint8  ,
    GGMLQuantizationType.IQ2_XXS: np.uint8  ,
    GGMLQuantizationType.IQ2_XS : np.uint8  ,
    GGMLQuantizationType.IQ3_XXS: np.uint8  ,
    GGMLQuantizationType.IQ1_S  : np.uint8  ,
    GGMLQuantizationType.IQ4_NL : np.uint8  ,
    GGMLQuantizationType.IQ3_S  : np.uint8  ,
    GGMLQuantizationType.IQ2_S  : np.uint8  ,
    GGMLQuantizationType.IQ4_XS : np.uint8  ,
    GGMLQuantizationType.IQ1_M  : np.uint8  ,
}

class GgufModel:
    def __init__(self):
        self.fields  = {}
        self.weights = {}

    @classmethod
    def load(cls, fname, **kwargs):
        self = cls(**kwargs)
    
        # load model from file
        self.data = np.memmap(fname, mode='r')
        self.offset = 0

        # check magic
        if (magic := self.read_uint32()) != GGUF_MAGIC:
            raise ValueError(f'Invalid GGUF magic: {hex(magic)}')

        # check version
        if (version := self.read_uint32()) != GGUF_VERSION:
            raise ValueError(f'Invalid GGUF version: {version}')

        # get data counts
        self.n_tensors = self.read_uint64()
        self.n_fields = self.read_uint64()

        # read fields
        self.fields = {}
        for _ in range(self.n_fields):
            # get metadata
            name = self.read_str()
            gtype = self.read_uint32()

            # parse value by gtype
            if gtype == GGUFValueType.STRING:
                value = self.read_str()
            elif gtype in gtype_to_dtype:
                dtype = gtype_to_dtype[gtype]
                value = self.read_scalar(dtype)
            elif gtype == GGUFValueType.ARRAY:
                itype = self.read_uint32()
                size = self.read_uint64()
                if itype == GGUFValueType.STRING:
                    value = [self.read_str() for _ in range(size)]
                elif itype in gtype_to_dtype:
                    dtype = gtype_to_dtype.get(itype)
                    value = self.read(dtype, size).tolist()
                else:
                    raise ValueError(f'Invalid array type: {itype}')
            else:
                raise ValueError(f'Invalid field type: {gtype}')

            # store field
            self.fields[name] = value

        # read tensor metadata
        self.meta = {}
        for _ in range(self.n_tensors):
            name = self.read_str()
            dims = self.read_uint32()
            shape = tuple(map(int, self.read(np.uint64, dims).tolist()))
            ttype = self.read_uint32()
            offset = self.read_uint64()
            self.meta[name] = shape, ttype, offset

        # read weights
        self.tensors = {}
        for name, (shape, ttype, offset) in self.meta.items():
            self.offset = offset
            dtype = ttype_to_dtype[ttype]
            count = np.prod(shape)
            data = self.read(dtype, count=count)
            self.tensors[name] = data.reshape(shape)

        # return model
        return self

    def __repr__(self):
        width = max([len(f) for f in self.fields])
        max_length = 8
        lines = []
        for key, value in self.fields.items():
            if isinstance(value, list):
                prev = ' , '.join(map(str, value[:max_length]))
                value = f'[ {len(value)} ] → ( {prev} , ... )'
            lines.append(f'{key:{width}} = {value}')
        return '\n'.join(lines)

    def set_field(self, name, value):
        self.fields[name] = value

    def set_tensor(self, name, value):
        self.tensors[name] = value

    def get_field(self, name):
        return self.fields.get(name)

    def get_tensor(self, name):
        return self.tensors.get(name)

    def read(self, dtype, count=1):
        width = np.dtype(dtype).itemsize
        off, size = self.offset, count * width
        bdat = self.data[off:off+size]
        data = bdat.view(dtype)[:count]
        self.offset += size
        return data

    def read_scalar(self, dtype):
        return self.read(dtype)[0]

    def read_uint32(self):
        return int(self.read_scalar(np.uint32))

    def read_uint64(self):
        return int(self.read_scalar(np.uint64))

    def read_str(self):
        size = self.read_uint64()
        data = self.read(np.uint8, size)
        return data.tobytes().decode('utf-8')
