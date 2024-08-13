# simple ggml test

import ctypes
import numpy as np

from .ggml import (
    ggml_type_size,
    ggml_tensor_overhead,
    ggml_graph_overhead,
    ggml_init_params,
    ggml_init,
    ggml_new_tensor_1d,
    ggml_new_tensor_2d,
    ggml_mul_mat,
    ggml_add,
    ggml_new_graph,
    ggml_build_forward_expand,
    ggml_graph_compute_with_ctx,
    ggml_backend_cpu_init,
    ggml_backend_alloc_ctx_tensors,
    ggml_free,
    ggml_backend_get_default_buffer_type,
    ggml_gallocr_new,
    ggml_gallocr_reserve,
    ggml_gallocr_get_buffer_size,
    ggml_gallocr_alloc_graph,
    ggml_backend_cpu_set_n_threads,
    ggml_backend_graph_compute,
    GGMLQuantizationType,
    GGML_DEFAULT_GRAPH_SIZE,
)

gtype_to_ctype = {
    GGMLQuantizationType.F32: ctypes.c_float
}

def trim_nelem(shape):
    dims = 1 + max([
        i for i, d in enumerate(shape) if d > 1
    ], default=0)
    return shape[:dims]

def get_tensor_shape(tensor, raw=False):
    value = tensor.contents
    nelem = tuple(value.ne[:4])
    return trim_nelem(nelem)[::-1]

def tensor_to_numpy(tensor):
    value = tensor.contents
    if value.type not in gtype_to_ctype:
        raise ValueError(f'unsupported type: {value.type}')
    ctype = gtype_to_ctype[value.type]
    shape = get_tensor_shape(tensor)
    p = ctypes.cast(value.data, ctypes.POINTER(ctype))
    return np.ctypeslib.as_array(p, shape=shape)

def test_context():
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

    # initialize ggml
    params = ggml_init_params(mem_usage, None, False)
    ctx = ggml_init(params)

    # create tensors
    a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_A, rows_A)
    b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_B, rows_B)

    # do multiplication
    c = ggml_mul_mat(ctx, a, b)

    # create graph
    gf = ggml_new_graph(ctx)
    ggml_build_forward_expand(gf, c)

    # create numpy arrays
    a_np = tensor_to_numpy(a)
    b_np = tensor_to_numpy(b)

    # fill arrays
    a_np[:,:] = [[2 , 8], [5, 1], [4, 2], [8, 6]]
    b_np[:,:] = [[10, 5], [9, 9], [5, 4]]

    # compute graph
    ggml_graph_compute_with_ctx(ctx, gf, 1)

    # get results
    n_nodes = gf.contents.n_nodes
    c_tensor = gf.contents.nodes[n_nodes-1]
    c_np = tensor_to_numpy(c_tensor)

    # test results
    assert np.allclose(c_np, (a_np @ b_np.T).T)

    # free memory
    ggml_free(ctx)

    # return arrays
    return a_np, b_np, c_np

def test_backend():
    # tensor shapes
    n_layers, embed_dim, batch_size = 50, 32, 16

    # initialize cpu backend
    backend = ggml_backend_cpu_init()

    # create tensor context
    num_tensors = 1 + 2 * n_layers # input + weights + biases
    mem_tensors = ggml_tensor_overhead() * num_tensors
    par_tensors = ggml_init_params(mem_tensors, None, True)
    ctx_tensors = ggml_init(par_tensors)

    # create tensors
    inputs = ggml_new_tensor_2d(ctx_tensors, GGMLQuantizationType.F32, embed_dim, batch_size)
    weights, biases = zip(*[
        (
            ggml_new_tensor_2d(ctx_tensors, GGMLQuantizationType.F32, embed_dim, embed_dim),
            ggml_new_tensor_1d(ctx_tensors, GGMLQuantizationType.F32, embed_dim)
        ) for _ in range(n_layers)
    ])

    # assign tensors to backend
    ggml_backend_alloc_ctx_tensors(ctx_tensors, backend)

    # compute memory requirements for graph
    mem_graph = (
        ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE +
        ggml_graph_overhead()
    )
    arr_graph = ctypes.create_string_buffer(mem_graph)
    buf_graph = ctypes.cast(arr_graph, ctypes.c_void_p)

    # create graph context
    par_graph = ggml_init_params(mem_graph, buf_graph, True)
    ctx_graph = ggml_init(par_graph)
    graph = ggml_new_graph(ctx_graph)

    # create network function
    x = inputs
    for i in range(n_layers):
        x = ggml_mul_mat(ctx_graph, weights[i], x, name=f'a{i}')
        x = ggml_add(ctx_graph, x, biases[i], name=f'b{i}')
    outputs = x

    # build graph from network
    ggml_build_forward_expand(graph, outputs)

    # free graph context
    ggml_free(ctx_graph)

    # allocate buffers for graph (worst case scenario)
    buf_type = ggml_backend_get_default_buffer_type(backend)
    allocr = ggml_gallocr_new(buf_type)
    ggml_gallocr_reserve(allocr, graph)
    ggml_gallocr_alloc_graph(allocr, graph)
    mem_worst = ggml_gallocr_get_buffer_size(allocr, 0)
    print(f'compute buffer size: {mem_worst/1024:.4f} KB\n')

    # set backend runtime options
    ggml_backend_cpu_set_n_threads(backend, 1)

    # set weights and biases
    for i in range(n_layers):
        weights_np = tensor_to_numpy(weights[i])
        biases_np = tensor_to_numpy(biases[i])
        weights_np[...] = np.random.randn(embed_dim, embed_dim).astype(np.float32)
        biases_np[...] = np.random.randn(embed_dim).astype(np.float32)

    # set input data
    inputs_np = tensor_to_numpy(inputs)
    inputs_np[...] = np.random.randn(batch_size, embed_dim).astype(np.float32)

    # do computation
    ggml_backend_graph_compute(backend, graph)

    # get results
    outputs_np = tensor_to_numpy(outputs)

    return graph, inputs, outputs, weights, biases
