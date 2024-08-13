# higher level ggml interface

import math
import ctypes
import numpy as np

from .utils import AttrDict
from .ggml import (
    GGMLQuantizationType,
    ggml_tensor_overhead,
    ggml_graph_overhead,
    ggml_init_params,
    ggml_init,
    ggml_new_tensor_1d,
    ggml_new_tensor_2d,
    ggml_new_tensor_3d,
    ggml_new_tensor_4d,
    ggml_set_name,
    ggml_mul_mat,
    ggml_add,
    ggml_new_graph,
    ggml_build_forward_expand,
    ggml_backend_cpu_init,
    ggml_backend_free,
    ggml_backend_alloc_ctx_tensors,
    ggml_free,
    ggml_backend_get_default_buffer_type,
    ggml_gallocr_new,
    ggml_gallocr_reserve,
    ggml_gallocr_alloc_graph,
    ggml_gallocr_get_buffer_size,
    ggml_backend_cpu_set_n_threads,
    ggml_backend_graph_compute,
    GGML_DEFAULT_GRAPH_SIZE,
)

##
## type conversion
##

gtype_to_ctype = {
    GGMLQuantizationType.F32: ctypes.c_float,
    # GGMLQuantizationType.F16: ctypes.c_half, # not supported by ctypes
    GGMLQuantizationType.I8: ctypes.c_int8,
    GGMLQuantizationType.I16: ctypes.c_int16,
    GGMLQuantizationType.I32: ctypes.c_int32,
    GGMLQuantizationType.I64: ctypes.c_int64,
}

gtype_to_dtype = {
    GGMLQuantizationType.F32: np.float32,
    # GGMLQuantizationType.F16: np.float16, # not supported by ctypes
    GGMLQuantizationType.I8: np.int8,
    GGMLQuantizationType.I16: np.int16,
    GGMLQuantizationType.I32: np.int32,
    GGMLQuantizationType.I64: np.int64,
}

##
## tensor utilities
##

def trim_nelem(shape):
    dims = 1 + max([
        i for i, d in enumerate(shape) if d > 1
    ], default=0)
    return shape[:dims]

def get_tensor_name(tensor):
    value = tensor.contents
    return value.name.decode('utf-8')

def get_tensor_shape(tensor, raw=False):
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

# this assumes the data is contiguous
# will implicity squeeze unit dimensions
def array_to_tensor(array, tensor):
    # check tensor type
    ttype = get_tensor_type(tensor)
    if ttype not in gtype_to_ctype:
        raise ValueError(f'unsupported type: {ttype}')

    # get data pointers
    src = array.ctypes.data
    dst = tensor.contents.data
    size = array.nbytes

    # copy data
    ctypes.memmove(dst, src, size)

# this makes a new array and copies
# we want to avoid deallocating ggml buffers
def tensor_to_array(tensor):
    # check tensor type
    ttype = get_tensor_type(tensor)
    if ttype not in gtype_to_dtype:
        raise ValueError(f'unsupported type: {ttype}')

    # get data pointers
    src = tensor.contents.data
    shape = get_tensor_shape(tensor)
    dtype = gtype_to_dtype[ttype]

    # create numpy array
    array = np.empty(shape, dtype=dtype)
    dst = array.ctypes.data
    size = array.nbytes

    # copy data
    ctypes.memmove(src, dst, size)

##
## context sizing and creation
##

def create_tensor_context(num_tensors):
    mem_tensors = ggml_tensor_overhead() * num_tensors
    par_tensors = ggml_init_params(mem_tensors, None, True)
    ctx_tensors = ggml_init(par_tensors)
    return ctx_tensors

def create_graph_context(graph_size=GGML_DEFAULT_GRAPH_SIZE):
    # compute memory requirements for graph
    mem_graph = (
        ggml_graph_overhead() + ggml_tensor_overhead() * graph_size
    )
    arr_graph = ctypes.create_string_buffer(mem_graph)
    buf_graph = ctypes.cast(arr_graph, ctypes.c_void_p)

    # create graph context
    par_graph = ggml_init_params(mem_graph, buf_graph, True)
    ctx_graph = ggml_init(par_graph)

    # return context
    return ctx_graph

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

##
## compute interface
##

class GgmlCompute:
    def __init__(self, specs, model, backend=None):
        # zero out model elements
        self.backend = None
        self.tensors = None
        self.graph = None

        # construct model elements
        self.create_backend(backend)
        self.create_tensors(specs)
        self.create_graph(model)

        # set backend runtime options
        ggml_backend_cpu_set_n_threads(self.backend, 1)

    def __del__(self):
        if self.graph is not None:
            # ggml_free(self.graph)
            pass
        if self.tensors is not None:
            ggml_free(self.tensors)
        if self.backend is not None:
            ggml_backend_free(self.backend)

    def create_backend(self, name):
        if name is None or name == 'cpu':
            self.backend_name = 'cpu'
            self.backend = ggml_backend_cpu_init()
        elif name == 'cuda':
            raise ValueError('cuda support not implemented yet')
            # self.backend = ggml_backend_cuda_init()
        else:
            raise ValueError(f'unknown backend: {name}')

    def create_tensors(self, specs):
        # create tensor context
        self.tensors = create_tensor_context(len(specs))

        # create tensors
        self.inputs = AttrDict({
            nam: create_tensor(self.tensors, typ, shp, nam=nam)
            for nam, (typ, shp) in specs.items()
        })

        # assign tensors on backend
        ggml_backend_alloc_ctx_tensors(self.tensors, self.backend)

    # get tensor values as numpy (copy)
    def get_input(self, name):
        tensor = self.inputs[name]
        return tensor_to_array(tensor)

    # set tensor values using numpy
    def set_input(self, name, array):
        tensor = self.inputs[name]
        if type(array) is not np.ndarray:
            ttype = get_tensor_type(tensor)
            dtype = gtype_to_dtype[ttype]
            array = np.asarray(array, dtype=dtype)
        array_to_tensor(array, tensor)

    # create computational graph
    def create_graph(self, model):
        # create graph context (this could be a context manager)
        ctx_graph = create_graph_context()

        # create graph and expand
        self.graph = ggml_new_graph(ctx_graph)
        self.output = model(ctx_graph, self.inputs)
        ggml_build_forward_expand(self.graph, self.output)

        # free graph context
        ggml_free(ctx_graph)

        # allocate buffers for graph (worst case scenario)
        buf_type = ggml_backend_get_default_buffer_type(self.backend)
        self.alloc = ggml_gallocr_new(buf_type)

        # allocate tensors to buffers for graph
        ggml_gallocr_reserve(self.alloc, self.graph)
        ggml_gallocr_alloc_graph(self.alloc, self.graph)

        # print memory summary
        n_buffers = self.alloc.contents.n_buffers
        mem_usage = ggml_gallocr_get_buffer_size(self.alloc, 0)

    def compute(self, **values):
        # set input values
        for name, value in values.items():
            self.set_input(name, value)

        # do computation
        ggml_backend_graph_compute(self.backend, self.graph)

        # get results
        output_np = tensor_to_array(self.output)

        # return results
        return output_np

    def __repr__(self):
        name = self.__class__.__name__
        graph = self.graph.contents
        lines = (
            [f'{name}(backend={self.backend_name})'] + ['', 'INPUTS'] +
            [get_tensor_info(tensor) for tensor in self.inputs.values()] + ['', 'GRAPH'] +
            [get_tensor_info(graph.nodes[i]) for i in range(graph.n_nodes)]
        )
        return '\n'.join(lines)

    def __call__(self, **values):
        return self.compute(**values)

##
## testing
##

def test_compute():
    # tensor shapes
    n_layers, embed_dim, batch_size = 50, 32, 16

    # tensor specifications
    spec_weight = {
        f'weight{i}': (GGMLQuantizationType.F32, (embed_dim, embed_dim))
        for i in range(n_layers)
    }
    spec_bias = {
        f'bias{i}': (GGMLQuantizationType.F32, (embed_dim,))
        for i in range(n_layers)
    }
    spec_input = {
        'x': (GGMLQuantizationType.F32, (batch_size, embed_dim))
    }
    spec = spec_weight | spec_bias | spec_input

    # define model function
    def test_model(ctx, inp):
        x = inp.x
        for i in range(n_layers):
            x = ggml_mul_mat(ctx, inp[f'weight{i}'], x, name=f'a{i}')
            x = ggml_add(ctx, x, inp[f'bias{i}'], name=f'b{i}')
        return x

    # create model graph
    model = GgmlCompute(spec, test_model)

    # set weights and biases
    for i in range(n_layers):
        weight = np.random.randn(embed_dim, embed_dim).astype(np.float32)
        bias = np.random.randn(embed_dim).astype(np.float32)
        model.set_input(f'weight{i}', weight)
        model.set_input(f'bias{i}', bias)

    # compute on input data
    x = np.random.randn(batch_size, embed_dim).astype(np.float32)
    output_np = model.compute(x=x)

    # return model
    return model
