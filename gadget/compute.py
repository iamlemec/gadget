# higher level ggml interface

import ctypes
import numpy as np

from .constants import GGMLQuantizationType
from .ggml import (
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
    ggml_backend_cpu_set_n_threads,
    ggml_backend_graph_compute,
    GGML_DEFAULT_GRAPH_SIZE,
)

gtype_to_ctype = {
    GGMLQuantizationType.F32: ctypes.c_float,
}

def tensor_to_numpy(tensor, dims=2):
    assert dims <= 4
    assert tensor.contents.type in gtype_to_ctype
    ctype = gtype_to_ctype[tensor.contents.type]
    shape = tuple(tensor.contents.ne[:dims])[::-1]
    p = ctypes.cast(tensor.contents.data, ctypes.POINTER(ctype))
    return np.ctypeslib.as_array(p, shape=shape)

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

def get_tensor_info(tensor):
    value = tensor.contents
    name = value.name.decode('utf-8')
    ttype = GGMLQuantizationType(value.type)
    shape = tuple(value.ne[:4])
    stat = f'{name}: {ttype.name} × {shape}'
    return stat

def set_tensor_name(tensor, name):
    ggml_set_name(tensor, name.encode('utf-8'))

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
        self.inputs = {
            nam: create_tensor(self.tensors, typ, shp, nam=nam)
            for nam, (typ, shp) in specs.items()
        }

        # assign tensors on backend
        ggml_backend_alloc_ctx_tensors(self.tensors, self.backend)

    # return tensor numpy view
    def get_input(self, name):
        ten = self.inputs[name]
        return tensor_to_numpy(ten)

    # set tensor values using numpy
    def set_input(self, name, value):
        ten = self.inputs[name]
        ten_np = tensor_to_numpy(ten)
        ten_np[:] = value

    # create computational graph
    def create_graph(self, model):
        # create graph context (this could be a context manager)
        ctx_graph = create_graph_context()

        # create graph and expand
        self.graph = ggml_new_graph(ctx_graph)
        output = model(ctx_graph, self.inputs)
        ggml_build_forward_expand(self.graph, output)

        # free graph context
        ggml_free(ctx_graph)

        # allocate buffers for graph (worst case scenario)
        buf_type = ggml_backend_get_default_buffer_type(self.backend)
        allocr = ggml_gallocr_new(buf_type)
        ggml_gallocr_reserve(allocr, self.graph)
        ggml_gallocr_alloc_graph(allocr, self.graph)

    def print_inputs(self):
        for name, tensor in self.inputs.items():
            print(get_tensor_info(tensor))

    def print_graph(self):
        n_nodes = self.graph.contents.n_nodes
        for i in range(n_nodes):
            tensor = self.graph.contents.nodes[i]
            print(get_tensor_info(tensor))

    def compute(self, **values):
        # set input values
        for name, value in values.items():
            self.set_input(name, value)

        # do computation
        ggml_backend_graph_compute(self.backend, self.graph)

        # get results
        n_nodes = self.graph.contents.n_nodes
        out_tensor = self.graph.contents.nodes[n_nodes-1]
        out_np = tensor_to_numpy(out_tensor)

        # return results
        return out_np

def test_compute():
    # define inputs: name -> (type, shape)
    spec = {
        'a': (GGMLQuantizationType.F32, (4, 2)),
        'b': (GGMLQuantizationType.F32, (3, 2)),
    }

    # define model function
    def test_model(ctx, inputs):
        a, b = inputs['a'], inputs['b']
        c = ggml_mul_mat(ctx, a, b)
        return c

    # create model graph
    model = GgmlCompute(spec, test_model)

    # compute on input data
    a = [[2 , 8], [5, 1], [4, 2], [8, 6]]
    b = [[10, 5], [9, 9], [5, 4]]
    c_np = model.compute(a=a, b=b)

    # test results
    a_np = np.array(a, dtype=np.float32)
    b_np = np.array(b, dtype=np.float32)
    c0_np = (a_np @ b_np.T).T
    assert np.allclose(c_np, c0_np)
