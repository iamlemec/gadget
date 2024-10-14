# kv cache class

from ..ggml import ggml_view_3d, ggml_element_size, ggml_cpy, ggml_build_forward_expand
from ..tensor import get_tensor_shape

## kv cache layout (f32)
# k : [n_embd, n_head, n_ctx, n_layer]
# v : [n_embd, n_head, n_ctx, n_layer]

# tensor: [n_embd, n_head, n_ctx, n_layer]
def get_layer_range(ctx, tensor, il, pos0, pos1):
    tsize = ggml_element_size(tensor)
    n_embd, n_head, n_ctx, n_layer = get_tensor_shape(tensor, trim=4)
    length = pos1 - pos0
    offset = (il * n_ctx + pos0) * tsize * n_embd * n_head
    nb1, nb2 = tsize * n_embd, tsize * n_embd * n_head
    return ggml_view_3d(ctx, tensor, n_embd, n_head, length, nb1, nb2, offset)

# src: [n_embd, n_head, length]
# dst: [n_embd, n_head, n_ctx, n_layer]
def set_layer_range(ctx, src, dst, il, pos0, pos1):
    dst1 = get_layer_range(ctx, dst, il, pos0, pos1)
    cpy1 = ggml_cpy(ctx, src, dst1)
    return cpy1

class KVLayerView:
    def __init__(self, cache, ctx, graph, layer, n_past):
        self.cache = cache
        self.ctx = ctx
        self.graph = graph
        self.layer = layer
        self.n_past = n_past

    def get(self, num):
        return self.cache.get_layer_range(self.ctx, self.layer, 0, self.n_past + num)

    def append(self, num, k, v):
        self.cache.set_layer_range(self.ctx, self.graph, self.layer, self.n_past, self.n_past + num, k, v)

    def update(self, k, v):
        _, _, batch_size_k = get_tensor_shape(k, trim=3)
        _, _, batch_size_v = get_tensor_shape(v, trim=3)
        assert batch_size_k == batch_size_v
        batch_size = batch_size_k
        self.append(batch_size, k, v)
        return self.get(batch_size)

class KVCache:
    def __init__(self, k_tensor, v_tensor):
        self.k_tensor = k_tensor
        self.v_tensor = v_tensor

    def get_layer_range(self, ctx, layer, pos0, pos1):
        return (
            get_layer_range(ctx, self.k_tensor, layer, pos0, pos1),
            get_layer_range(ctx, self.v_tensor, layer, pos0, pos1),
        )

    def set_layer_range(self, ctx, graph, layer, pos0, pos1, k, v):
        k1 = set_layer_range(ctx, k, self.k_tensor, layer, pos0, pos1)
        v1 = set_layer_range(ctx, v, self.v_tensor, layer, pos0, pos1)
        ggml_build_forward_expand(graph, k1)
        ggml_build_forward_expand(graph, v1)

    def layer_view(self, ctx, graph, layer, n_past):
        return KVLayerView(self, ctx, graph, layer, n_past)
