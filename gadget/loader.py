# high leve gguf interface

import numpy as np
from operator import itemgetter

from .constants import (
    GGUF_MAGIC, GGUF_VERSION, GGUF_DEFAULT_ALIGNMENT,
    GGUFValueType, GGMLQuantizationType
)

# map for scalar types (invertible)
gtype_to_type = {
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
gtype_to_dtype = {k: np.dtype(v) for k, v in gtype_to_type.items()}
type_to_gtype = {v: k for k, v in gtype_to_type.items()}
dtype_to_gtype = {v: k for k, v in gtype_to_dtype.items()}

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

# map for tensor types (unquantized only)
type_to_ttype = {
    np.float16: GGMLQuantizationType.F16  ,
    np.float32: GGMLQuantizationType.F32  ,
    np.float64: GGMLQuantizationType.F64  ,
    np.int8   : GGMLQuantizationType.I8 ,
    np.int16  : GGMLQuantizationType.I16,
    np.int32  : GGMLQuantizationType.I32,
    np.int64  : GGMLQuantizationType.I64,
}
dtype_to_ttype = {np.dtype(k): v for k, v in type_to_ttype.items()}

class MmapWriter:
    def __init__(self, fname, size):
        self.data = np.memmap(fname, mode='w+', shape=(size,))
        self.offset = 0

    def __del__(self):
        self.data.flush()

    def write_array(self, data):
        off, size = self.offset, data.nbytes
        self.data[off:off+size] = data.view(np.uint8)
        self.offset += size

    def write_scalar(self, value, dtype):
        data = np.array([value], dtype=dtype)
        self.write_array(data)

    def write_uint32(self, value):
        self.write_scalar(value, np.uint32)

    def write_uint64(self, value):
        self.write_scalar(value, np.uint64)

    def write_str(self, value):
        data = value.encode('utf-8')
        self.write_scalar(len(data), np.uint64)
        self.write_array(np.frombuffer(data, dtype=np.uint8))

class GgufModel:
    def __init__(self):
        self.fields  = {}
        self.tensors = {}

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
        n_tensors = self.read_uint64()
        n_fields = self.read_uint64()

        # read fields
        self.fields = {}
        for _ in range(n_fields):
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
                    value = np.array(self.read(dtype, size))
                else:
                    raise ValueError(f'Invalid array type: {itype}')
            else:
                raise ValueError(f'Invalid field type: {gtype}')

            # store field
            self.fields[name] = value

        # read tensor metadata
        metadata = {}
        for _ in range(n_tensors):
            name = self.read_str()
            dims = self.read_uint32()
            shape = tuple(map(int, self.read(np.uint64, dims).tolist()))
            ttype = GGMLQuantizationType(self.read_uint32())
            offset = self.read_uint64()
            metadata[name] = shape, ttype, offset

        # read weights
        self.tensors = {}
        for name, (shape, ttype, offset) in metadata.items():
            self.offset = offset
            dtype = ttype_to_dtype[ttype]
            count = np.prod(shape)
            vals = self.read(dtype, count=count)
            self.tensors[name] = ttype, vals.reshape(shape)

        # return model
        return self

    def save(self, fname):
        # create output file
        gguf_size = self.gguf_size()
        writer = MmapWriter(fname, gguf_size)

        # write magic
        writer.write_uint32(GGUF_MAGIC)

        # write version
        writer.write_uint32(GGUF_VERSION)

        # write counts
        writer.write_uint64(len(self.tensors))
        writer.write_uint64(len(self.fields))

        # write fields
        for name, value in self.fields.items():
            # write name
            writer.write_str(name)

            # write value (no subclassing allowed)
            vtype = type(value)
            if vtype is str:
                writer.write_uint32(GGUFValueType.STRING)
                writer.write_str(value)
            elif vtype is np.ndarray:
                dtype = value.dtype
                gtype = dtype_to_gtype[dtype]
                if value.ndim == 0:
                    writer.write_uint32(gtype)
                    writer.write_scalar(value, dtype)
                elif value.ndim == 1:
                    writer.write_uint32(GGUFValueType.ARRAY)
                    writer.write_uint32(gtype)
                    writer.write_uint64(len(value))
                    if itype is str:
                        for item in value:
                            writer.write_str(item)
                    else:
                        writer.write_array(value)
                else:
                    raise ValueError(f'Fields must be 0- or 1-dimensional arrays')

        # get tensor starting offset
        offset = self.base_size() + self.field_size() + self.meta_size()

        # write tensor metadata
        for name, (ttype, tensor) in self.tensors.items():
            shape = tensor.shape
            writer.write_str(name)
            writer.write_uint32(len(shape))
            writer.write_array(np.array(shape, dtype=np.uint64))
            writer.write_uint32(ttype)
            writer.write_uint64(offset)

            # update offset
            count = np.prod(shape)
            width = np.dtype(ttype_to_dtype[ttype]).itemsize
            offset += count * width

        # write weights
        for ttype, tensor in self.tensors.values():
            writer.write_array(tensor)

    def __repr__(self):
        width = max(
            [len(f) for f in self.fields ] +
            [len(t) for t in self.tensors]
        )
        max_length = 8
        lines = ['FIELDS']
        for key, value in self.fields.items():
            if type(value) is np.ndarray:
                value = value.tolist()
            if type(value) is list:
                prev = ' , '.join(map(str, value[:max_length]))
                if len(value) > max_length:
                    value = f'[ {prev} , ... ] ({len(value)})'
                else:
                    value = f'[ {prev} ] ({len(value)})'
            lines.append(f'{key:{width}} = {value}')
        lines += ['', 'TENSORS']
        for key, (ttype, tensor) in self.tensors.items():
            lines.append(f'{key:{width}} = {ttype.name} {tensor.shape}')
        return '\n'.join(lines)

    def base_size(self):
        return 4 + 4 + 8 + 8 # magic + version + n_tensors + n_fields

    def field_size(self, name=None):
        if name is None:
            return sum(self.field_size(f) for f in self.fields)
        base = 8 + len(name) # length + name
        value = self.fields[name]
        vtype = type(value)
        if vtype is str:
            data = 4 + 8 + len(value) # type + length + string
        elif vtype is np.ndarray:
            if value.ndim == 0:
                width = np.dtype(value.dtype).itemsize
                data = 4 + width # type + scalar
            else:
                data = 4 + 4 + 8 + value.nbytes # type + ttype + size + array
        else:
            raise ValueError(f'Invalid field type: {vtype}')
        return base + data

    def meta_size(self, name=None):
        if name is None:
            return sum(self.meta_size(t) for t in self.tensors)
        ttype, tensor = self.tensors[name]
        base = 8 + len(name) # length + name
        shape = 4 + 8 * tensor.ndim # dims + shape
        return base + shape + 4 + 8 # name + shape + ttype + offset

    def tensor_size(self, name=None):
        if name is None:
            return sum(self.tensor_size(t) for t in self.tensors)
        ttype, tensor = self.tensors[name]
        return tensor.nbytes

    def gguf_size(self):
        return self.base_size() + self.field_size() + self.meta_size() + self.tensor_size()

    def get_field(self, name):
        return self.fields.get(name)

    def set_field(self, name, value, dtype=None):
        if type(value) is list and set(map(type, value)) == {str}:
            pass
        elif type(value) is not np.ndarray:
            if dtype is None:
                raise ValueError(f'Must specify dtype for non-ndarray fields')
            value = np.array(value, dtype=dtype)
        if type(value) is np.ndarray and value.ndim > 1:
            raise ValueError(f'Fields must be 0- or 1-dimensional arrays')
        self.fields[name] = value

    def set_tensor(self, name, value, ttype=None):
        if ttype is None:
            if value.dtype in dtype_to_ttype:
                ttype = dtype_to_ttype[value.dtype]
            else:
                raise ValueError(f'Must specify tensor type for {value.dtype}')
        self.tensors[name] = ttype, value

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

def test_model():
    # make data
    data = np.arange(12).reshape((3, 4))

    # create model
    gf = GgufModel()
    gf.set_field('name', 'test')
    gf.set_field('value', 42)
    gf.set_tensor('data', data)

    return gf
