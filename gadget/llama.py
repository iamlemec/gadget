# wrappers around clib functions

import ctypes
import numpy as np

##
## imports
##

from .libs import _libllama

from .libs._libllama import (
    LLAMA_POOLING_TYPE_UNSPECIFIED,
    LLAMA_POOLING_TYPE_NONE,
    LLAMA_POOLING_TYPE_MEAN,
    LLAMA_POOLING_TYPE_CLS,
    LLAMA_POOLING_TYPE_LAST,
    llama_model_params,
    llama_context_params,
    llama_model_default_params,
    llama_context_default_params,
    llama_new_context_with_model,
    llama_backend_init,
    llama_numa_init,
    llama_backend_free,
    llama_free_model,
    llama_free,
    llama_batch_free,
    llama_decode,
    llama_pooling_type,
    llama_set_causal_attn,
    llama_n_embd,
)

##
## wrappers
##

def llama_numa_init(numa=_libllama.GGML_NUMA_STRATEGY_DISABLED):
    _libllama.llama_numa_init(numa)

def llama_load_model_from_file(path_model, params):
    return _libllama.llama_load_model_from_file(path_model.encode('utf-8'), params)

def llama_batch_init(n_tokens, embd=0, n_seq_max=1):
    return _libllama.llama_batch_init(n_tokens, embd, n_seq_max)

def llama_tokenize(model, text, max_tokens, add_special=True, parse_special=False):
    text_bytes = text.encode('utf-8')
    text_len = len(text_bytes)
    tokens = np.zeros(max_tokens, dtype=np.int32)
    tokens_p = tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    n_tokens = _libllama.llama_tokenize(
        model, text_bytes, text_len, tokens_p, max_tokens, add_special, parse_special
    )
    return tokens[:n_tokens].tolist()

def llama_get_embeddings(ctx, n_tokens, n_embd):
    p_embd = _libllama.llama_get_embeddings(ctx)
    return np.ctypeslib.as_array(p_embd, shape=(n_tokens, n_embd))

def llama_get_embeddings_seq(ctx, seq_id, n_embd):
    p_embd = _libllama.llama_get_embeddings_seq(ctx, seq_id)
    return np.ctypeslib.as_array(p_embd, shape=(n_embd,))
