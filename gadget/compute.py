# higher level ggml interface

import math
import ctypes
import numpy as np

from .utils import AttrDict
from .ggml import (
    GGMLQuantizationType,
    GGML_QUANT_SIZES,
    ggml_tensor_overhead,
    ggml_graph_overhead,
    ggml_init_params,
    ggml_init,
    ggml_new_tensor_1d,
    ggml_new_tensor_2d,
    ggml_new_tensor_3d,
    ggml_new_tensor_4d,
    ggml_set_name,
    ggml_nelements,
    ggml_is_quantized,
    ggml_internal_get_type_traits,
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
from .libs.general import malloc, free
from .tensor import (
    ttype_to_ctype,
    ttype_to_dtype,
    get_tensor_shape,
    get_tensor_type,
    get_tensor_info,
    get_data_shape,
    create_tensor,
)

##
## tensor-array serialization
##

# this assumes the data is contiguous
# will implicity squeeze unit dimensions
def array_to_tensor(array, tensor):
    # get expected dtype
    ttype = get_tensor_type(tensor)
    dtype = ttype_to_dtype[ttype]
    shape = get_tensor_shape(tensor)

    # get quantization situation
    quant = ggml_is_quantized(ttype)
    dshape = get_data_shape(tensor)
    is_quantized = quant and array.dtype == np.uint8
    will_quantize = quant and array.dtype == np.float32

    # check type match
    if not will_quantize and dtype != array.dtype:
        raise ValueError(f'array dtype ({array.dtype}) does not match expected dtype ({dtype})')

    # check shape match
    if is_quantized and array.shape != dshape:
        raise ValueError(f'input shape {array.shape} does not match target (quantized) shape {dshape}')
    if not is_quantized and array.shape != shape:
        raise ValueError(f'input shape {array.shape} does not match target shape {shape}')

    # get data pointers
    src = array.ctypes.data
    dst = tensor.contents.data

    # do quant conversion if needed
    if will_quantize:
        src_p = ctypes.cast(src, ctypes.POINTER(ctypes.c_float))
        dst_p = ctypes.cast(dst, ctypes.c_void_p)
        size = ggml_nelements(tensor)
        traits = ggml_internal_get_type_traits(ttype)
        traits.from_float(src_p, dst_p, size)
    else:
        ctypes.memmove(dst, src, array.nbytes)

# this makes a new array and copies
# we want to avoid deallocating ggml buffers
def tensor_to_array(tensor):
    # check ctype support
    ttype = get_tensor_type(tensor)
    if ttype not in ttype_to_dtype:
        raise ValueError(f'unsupported type: {ttype}')

    # get data pointers
    src = tensor.contents.data
    shape = get_tensor_shape(tensor)
    dtype = ttype_to_dtype[ttype]

    # create numpy array
    array = np.empty(shape, dtype=dtype)
    dst = array.ctypes.data
    size = array.nbytes

    # copy data
    ctypes.memmove(dst, src, size)

    # return array
    return array

##
## compute interface
##

class GgmlCompute:
    def __init__(self, params, tensors, model, backend=None):
        # initialize empty
        self.backend = None
        self.ctx_tensors = None
        self.ctx_graph = None

        # create tensors and graph
        self.create_params(params)
        self.create_backend(backend)
        self.create_tensors(tensors)
        self.create_graph(model)

    def __del__(self):
        if self.ctx_graph is not None:
            ggml_free(self.ctx_graph)
        if self.ctx_tensors is not None:
            ggml_free(self.ctx_tensors)
        if self.backend is not None:
            ggml_backend_free(self.backend)

    def create_params(self, params):
        self.params = AttrDict(params)

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
        num_tensors = len(specs)
        mem_tensors = ggml_tensor_overhead() * num_tensors
        par_tensors = ggml_init_params(mem_tensors, None, True)
        self.ctx_tensors = ggml_init(par_tensors)

        # create tensors
        self.tensors = AttrDict({
            nam: create_tensor(self.ctx_tensors, typ, shp, nam=nam)
            for nam, (typ, shp) in specs.items()
        })

        # assign tensors on backend
        self.backend_buf = ggml_backend_alloc_ctx_tensors(self.ctx_tensors, self.backend)

    # get tensor values as numpy (copy)
    def get_input(self, name):
        tensor = self.tensors[name]
        return tensor_to_array(tensor)

    # set tensor values using numpy
    def set_input(self, name, array):
        tensor = self.tensors[name]
        try:
            array_to_tensor(array, tensor)
        except ValueError as e:
            raise ValueError(f'error setting input "{name}":\n{e}')

    # create computational graph
    def create_graph(self, model, graph_size=GGML_DEFAULT_GRAPH_SIZE):
        # compute memory requirements for graph
        # NOTE: need to keep reference to arr_graph around to prevent garbage collect!!!
        mem_graph = (
            ggml_graph_overhead() + ggml_tensor_overhead() * graph_size
        )
        self.arr_graph = ctypes.create_string_buffer(mem_graph)

        # create graph context
        buf_graph = ctypes.cast(self.arr_graph, ctypes.c_void_p)
        par_graph = ggml_init_params(mem_graph, buf_graph, True)
        self.ctx_graph = ggml_init(par_graph)

        # create graph and expand
        self.graph = ggml_new_graph(self.ctx_graph)
        self.output = model(self.ctx_graph, self.params, self.tensors)
        ggml_build_forward_expand(self.graph, self.output)

        # allocate buffers for graph (worst case scenario)
        self.buf_type = ggml_backend_get_default_buffer_type(self.backend)
        self.alloc = ggml_gallocr_new(self.buf_type)

        # allocate tensors to buffers for graph
        ggml_gallocr_reserve(self.alloc, self.graph)
        ggml_gallocr_alloc_graph(self.alloc, self.graph)

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
            [get_tensor_info(tensor) for tensor in self.tensors.values()] + ['', 'GRAPH'] +
            [get_tensor_info(graph.nodes[i]) for i in range(graph.n_nodes)]
        )
        return '\n'.join(lines)

    def __call__(self, **values):
        return self.compute(**values)

##
## testing
##

def test_compute(input_dim=64, output_dim=32, batch_size=16):
    from .ggml import ggml_mul_mat, ggml_add

    # model parameters
    params = dict(
        input_dim=input_dim, output_dim=output_dim, batch_size=batch_size
    )

    # tensor specifications
    tensors = dict(
        a = (GGMLQuantizationType.F32, (output_dim, input_dim)),
        b = (GGMLQuantizationType.F32, (output_dim,)),
        x = (GGMLQuantizationType.F32, (batch_size, input_dim)),
    )

    # define model function
    def test_model(ctx, par, ten):
        n, m = par['input_dim'], par['output_dim']
        a, b, x = ten['a'], ten['b'], ten['x']
        x1 = ggml_mul_mat(ctx, a, x, name=f'x1')
        x2 = ggml_add(ctx, x1, b, name=f'x2')
        return x2

    # create model graph
    model = GgmlCompute(params, tensors, test_model)

    # set weights
    a_np = np.random.randn(output_dim, input_dim).astype(np.float32)
    b_np = np.random.randn(output_dim).astype(np.float32)
    model.set_input('a', a_np)
    model.set_input('b', b_np)

    # compute on input data
    x_np = np.random.randn(batch_size, input_dim).astype(np.float32)
    y_np = model(x=x_np)

    # get numpy results
    y0_np = (x_np @ a_np.T) + b_np[None,:]
    match = np.allclose(y_np, y0_np, atol=1e-5)

    # return result
    return match

def test_quant(input_dim=64, output_dim=32, batch_size=16):
    from .ggml import ggml_mul_mat, ggml_add

    # model parameters
    params = dict(
        input_dim=input_dim, output_dim=output_dim, batch_size=batch_size
    )

    # tensor specifications
    tensors = dict(
        a = (GGMLQuantizationType.Q8_0, (output_dim, input_dim)),
        b = (GGMLQuantizationType.F32, (output_dim,)),
        x = (GGMLQuantizationType.F32, (batch_size, input_dim)),
    )

    # define model function
    def test_model(ctx, par, ten):
        n, m = par['input_dim'], par['output_dim']
        a, b, x = ten['a'], ten['b'], ten['x']
        x1 = ggml_mul_mat(ctx, a, x, name=f'x1')
        x2 = ggml_add(ctx, x1, b, name=f'x2')
        return x2

    # create model graph
    model = GgmlCompute(params, tensors, test_model)

    # set weights
    a_np = np.random.randn(output_dim, input_dim).astype(np.float32)
    b_np = np.random.randn(output_dim).astype(np.float32)
    model.set_input('a', a_np)
    model.set_input('b', b_np)

    # compute on input data
    x_np = np.random.randn(batch_size, input_dim).astype(np.float32)
    y_np = model(x=x_np)

    # get numpy results
    y0_np = (x_np @ a_np.T) + b_np[None,:]
    match = np.allclose(y_np, y0_np, atol=1e-5)

    # get rms and abs proportional errors
    rmse = np.sqrt(np.square(y_np-y0_np).mean()) / np.abs(y0_np).mean()
    abse = np.abs(y_np-y0_np).mean() / np.abs(y0_np).mean()

    # return result
    return match, rmse, abse
