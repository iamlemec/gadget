# simple ggml test

import ctypes
import numpy as np

from .ggml import (
    ggml_type_size, ggml_tensor_overhead, ggml_graph_overhead,
    ggml_init_params, ggml_init, ggml_new_tensor_2d, ggml_mul_mat,
    ggml_new_graph, ggml_build_forward_expand, ggml_graph_compute_with_ctx,
    ggml_backend_cpu_init, ggml_backend_alloc_ctx_tensors, ggml_free,
    ggml_backend_get_default_buffer_type, ggml_gallocr_new, ggml_gallocr_reserve,
    ggml_gallocr_get_buffer_size, ggml_gallocr_alloc_graph, ggml_backend_cpu_set_n_threads,
    ggml_backend_graph_compute, GGML_TYPE_F32, GGML_DEFAULT_GRAPH_SIZE
)

gtype_to_ctype = {
    GGML_TYPE_F32: ctypes.c_float,
}

def tensor_to_numpy(tensor, dims=2):
    assert dims <= 4
    assert tensor.contents.type in gtype_to_ctype
    ctype = gtype_to_ctype[tensor.contents.type]
    shape = tuple(tensor.contents.ne[:dims])[::-1]
    p = ctypes.cast(tensor.contents.data, ctypes.POINTER(ctype))
    return np.ctypeslib.as_array(p, shape=shape)

def test_ggml_ctx():
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

def test_ggml_backend():
    # tensor shapes
    rows_A, cols_A = 4, 2
    rows_B, cols_B = 3, 2

    ## load model

    # initialize cpu backend
    backend = ggml_backend_cpu_init()

    # create tensor context
    num_tensors = 2
    mem_tensors = ggml_tensor_overhead() * num_tensors
    par_tensors = ggml_init_params(mem_tensors, None, True)
    ctx_tensors = ggml_init(par_tensors)

    # create tensors
    a = ggml_new_tensor_2d(ctx_tensors, GGML_TYPE_F32, cols_A, rows_A)
    b = ggml_new_tensor_2d(ctx_tensors, GGML_TYPE_F32, cols_B, rows_B)

    # assign tensors to backend
    ggml_backend_alloc_ctx_tensors(ctx_tensors, backend)

    ## build_graph

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

    # create graph and expand
    gf = ggml_new_graph(ctx_graph)
    c = ggml_mul_mat(ctx_graph, a, b)
    ggml_build_forward_expand(gf, c)

    # free graph context
    ggml_free(ctx_graph)

    # allocate buffers for graph (worst case scenario)
    buf_type = ggml_backend_get_default_buffer_type(backend)
    allocr = ggml_gallocr_new(buf_type)
    ggml_gallocr_reserve(allocr, gf)
    mem_worst = ggml_gallocr_get_buffer_size(allocr, 0)
    print(f'compute buffer size: {mem_worst/1024:.4f} KB\n')

    # allocate tensors to buffers for graph
    ggml_gallocr_alloc_graph(allocr, gf)

    # set backend runtime options
    ggml_backend_cpu_set_n_threads(backend, 1)

    # set input data
    a_np = tensor_to_numpy(a)
    b_np = tensor_to_numpy(b)
    a_np[:,:] = [[2 , 8], [5, 1], [4, 2], [8, 6]]
    b_np[:,:] = [[10, 5], [9, 9], [5, 4]]

    # do computation
    ggml_backend_graph_compute(backend, gf)

    # get results
    n_nodes = gf.contents.n_nodes
    c_tensor = gf.contents.nodes[n_nodes-1]
    c_np = tensor_to_numpy(c_tensor)

    # test results
    assert np.allclose(c_np, (a_np @ b_np.T).T)

    # free memory
    ggml_free(ctx_tensors)

    # return arrays
    return a_np, b_np, c_np
