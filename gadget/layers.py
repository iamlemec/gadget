# common layers for reuse

from math import sqrt

from .ggml import (
    LlamaPoolingType,
    ggml_norm,
    ggml_norm_inplace,
    ggml_rms_norm,
    ggml_rms_norm_inplace,
    ggml_mul,
    ggml_mul_inplace,
    ggml_add,
    ggml_add_inplace,
    ggml_gelu,
    ggml_mul_mat,
    ggml_permute,
    ggml_transpose,
    ggml_cont,
    ggml_cont_2d,
    ggml_reshape_2d,
    ggml_reshape_3d,
    ggml_soft_max_ext,
)
from .compute import get_tensor_shape

def linear_layer(ctx, x, weight, bias=None, name=None):
    x = ggml_mul_mat(ctx, weight, x, name=f'{name}_mul')
    if bias is not None:
        x = ggml_add_inplace(ctx, x, bias, name=f'{name}_add')
    return x

def norm_layer(ctx, x, weight, bias, eps=0.0, rms=False, inplace=False, name=None):
    if inplace:
        norm_func = ggml_rms_norm_inplace if rms else ggml_norm_inplace
        mul_func, add_func = ggml_mul_inplace, ggml_add_inplace
    else:
        norm_func = ggml_rms_norm if rms else ggml_norm
        mul_func, add_func = ggml_mul, ggml_add
    x = norm_func(ctx, x, eps, name=f'{name}_norm')
    x = mul_func(ctx, x, weight, name=f'{name}_mul')
    x = add_func(ctx, x, bias, name=f'{name}_add')
    return x

def attention_layer(ctx, x, n_heads, mask, wq, bq, wk, bk, wv, bv, wo, bo, eps=0.0, alibi=0.0, name=None):
    # get dimensions
    batch_size, embed_dim= get_tensor_shape(x)
    if embed_dim % n_heads != 0:
        raise ValueError(f'embed_dim ({embed_dim}) must be divisble by n_heads ({n_heads})')

    # get attention head_dim
    head_dim = embed_dim // n_heads
    head_wgt = 1.0/sqrt(head_dim)

    # compute query, key, value
    q = linear_layer(ctx, x, wq, bq, name=f'{name}_q')
    k = linear_layer(ctx, x, wk, bk, name=f'{name}_k')
    v = linear_layer(ctx, x, wv, bv, name=f'{name}_v')

    # reshape to head_dim
    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, batch_size)
    k = ggml_reshape_3d(ctx, k, head_dim, n_heads, batch_size)

    # permute dimensions
    q = ggml_permute(ctx, q, 0, 2, 1, 3)
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3))

    # compute interactions
    kq = ggml_mul_mat(ctx, k, q)
    kq = ggml_soft_max_ext(ctx, kq, mask, head_wgt, alibi)

    # pull in values
    v = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, v, embed_dim, batch_size)))
    kqv = ggml_mul_mat(ctx, ggml_reshape_3d(ctx, v, batch_size, head_dim, n_heads), kq)

    # merge dimensions
    kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3)
    kqv = ggml_cont_2d(ctx, kqv, embed_dim, batch_size)

    # apply output layer
    out = linear_layer(ctx, kqv, wo, bo, name=f'{name}_out')

    # return output
    return out

activations = {
    'gelu': ggml_gelu,
}

def feed_forward_layer(ctx, x, wu, bu, wd, bd, act='gelu', name=None):
    x = linear_layer(ctx, x, wu, bu, name=f'{name}_up')
    x = activations[act](ctx, x)
    x = linear_layer(ctx, x, wd, bd, name=f'{name}_down')
    return x
