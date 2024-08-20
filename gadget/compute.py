# higher level ggml interface

import re
import ctypes
import numpy as np

from .utils import AttrDict
from .ggml import (
    GGML_DEFAULT_GRAPH_SIZE,
    GGMLQuantizationType as T,
    ggml_tensor_overhead,
    ggml_graph_overhead,
    ggml_init_params,
    ggml_init,
    ggml_free,
    ggml_new_graph,
    ggml_build_forward_expand,
    ggml_backend_cpu_init,
    ggml_backend_cuda_init,
    ggml_backend_free,
    ggml_backend_alloc_ctx_tensors,
    ggml_backend_get_default_buffer_type,
    ggml_backend_cpu_set_n_threads,
    ggml_backend_graph_compute,
    ggml_gallocr_new,
    ggml_gallocr_reserve,
    ggml_gallocr_alloc_graph,
)
from .tensor import (
    get_framework,
    get_tensor_name,
    get_tensor_info,
    create_tensor,
    array_to_tensor,
    tensor_to_array,
)

##
## compute interface
##

class GgmlCompute:
    def __init__(self, params, tensors, model, backend=None, framework=None):
        # initialize empty
        self.backend = None
        self.ctx_tensors = None
        self.ctx_graph = None

        # create tensors and graph
        self.create_params(params)
        self.create_backend(backend)
        self.create_tensors(tensors)
        self.create_graph(model)

        # other options
        self.framework = 'numpy' if framework is None else framework

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
            self.backend = ggml_backend_cpu_init()
            self.backend_type = 'cpu'
        elif (reg := re.match(r'^cuda(?::(\d+))?$', 'cuda')) is not None:
            num, = reg.groups()
            num = 0 if num is None else str(num)
            self.backend = ggml_backend_cuda_init(num)
            self.backend_type = 'cuda'
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

    def get_tensor(self, tensor, framework=None, device=None):
        if framework is None:
            framework = self.framework
        if device is None:
            device = self.backend_type
        return tensor_to_array(tensor, framework=framework, device=device)

    # get tensor values as numpy (copy)
    def get_input(self, name, **kwargs):
        tensor = self.tensors[name]
        return self.get_tensor(tensor, **kwargs)

    # set tensor values using numpy
    def set_input(self, name, array):
        tensor = self.tensors[name]
        try:
            array_to_tensor(array, tensor)
        except ValueError as e:
            raise ValueError(f'error setting input "{name}":\n{e}')

    def get_node(self, index, **kwargs):
        n_nodes = self.graph.contents.n_nodes
        if index >= n_nodes:
            raise ValueError(f'index ({index}) >= n_nodes ({n_nodes})')
        tensor = self.graph.contents.nodes[index]
        return self.get_tensor(tensor, **kwargs)

    def get_named_node(self, name, **kwargs):
        n_nodes = self.graph.contents.n_nodes
        for i in range(n_nodes):
            tensor = self.graph.contents.nodes[i]
            tname = get_tensor_name(tensor)
            if tname == name:
                return self.get_tensor(tensor, **kwargs)
        raise ValueError(f'node named "{name}" not found')

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

    # do computation
    def compute(self):
        ggml_backend_graph_compute(self.backend, self.graph)

    def __call__(self, **values):
        # set input values
        for name, value in values.items():
            self.set_input(name, value)

        # run that baby
        self.compute()

        # get results
        output_np = self.get_tensor(self.output)

        # return results
        return output_np

    def __repr__(self):
        name = self.__class__.__name__
        graph = self.graph.contents
        lines = (
            [f'{name}(backend={self.backend_type})'] + ['', 'INPUTS'] +
            [get_tensor_info(tensor) for tensor in self.tensors.values()] + ['', 'GRAPH'] +
            [get_tensor_info(graph.nodes[i]) for i in range(graph.n_nodes)]
        )
        return '\n'.join(lines)

##
## testing
##

def test_compute(input_dim=256, output_dim=32, batch_size=16, qtype=T.F32, **kwargs):
    from .ggml import ggml_mul_mat, ggml_add

    # model parameters
    params = dict(
        input_dim=input_dim, output_dim=output_dim, batch_size=batch_size
    )

    # tensor specifications
    tensors = dict(
        a = (qtype, (output_dim, input_dim)),
        b = (T.F32, (output_dim,)),
        x = (T.F32, (batch_size, input_dim)),
    )

    # define model function
    def test_model(ctx, par, ten):
        n, m = par['input_dim'], par['output_dim']
        a, b, x = ten['a'], ten['b'], ten['x']
        x1 = ggml_mul_mat(ctx, a, x, name=f'x1')
        x2 = ggml_add(ctx, x1, b, name=f'x2')
        return x2

    # create model graph
    model = GgmlCompute(params, tensors, test_model, **kwargs)

    # set weights
    a_dtype = np.float16 if qtype == T.F16 else np.float32
    a_np = np.random.randn(output_dim, input_dim).astype(a_dtype)
    b_np = np.random.randn(output_dim).astype(np.float32)
    model.set_input('a', a_np)
    model.set_input('b', b_np)

    # compute on input data
    x_np = np.random.randn(batch_size, input_dim).astype(np.float32)
    y_np = model(x=x_np)

    # bring to host numpy if needed
    if hasattr(y_np, 'numpy'):
        y_np = y_np.cpu().numpy()

    # get numpy results
    y0_np = (x_np @ a_np.T) + b_np[None,:]
    match = np.allclose(y_np, y0_np, atol=1e-5)

    # get rms and abs proportional errors
    rmse = np.sqrt(np.square(y_np-y0_np).mean()) / np.abs(y0_np).mean()
    abse = np.abs(y_np-y0_np).mean() / np.abs(y0_np).mean()
    print(match, rmse, abse)

    # return result
    return model
