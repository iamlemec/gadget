### llama.cpp bindings for embeddings
### mostly copied from llama-cpp-python

import ctypes

##
## library
##

# for testing
class DummyLib:
    def __getattr__(self, name):
        return DummyLib()
    def __setattr(self, name, value):
        pass

# load ctypes lib
def load_llama_lib(lib_path):
    try:
        return ctypes.CDLL(lib_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load shared library '{lib_path}': {e}")

# load shared library
# _lib = DummyLib()
_lib = load_llama_lib('build/llama.cpp/src/libllama.so')

##
## utils
##

def ctypes_function(argtypes=None, restype=None):
    if argtypes is None:
        argtypes = []
    def decorator(func):
        name = func.__name__
        func = getattr(_lib, name)
        func.argtypes = argtypes
        func.restype = restype
        return func
    return decorator

##
## types
##

# basic types
llama_pos = ctypes.c_int32
llama_token = ctypes.c_int32
llama_seq_id = ctypes.c_int32

# struct pointers
llama_model_p = ctypes.c_void_p
llama_context_p = ctypes.c_void_p

# callback types
llama_progress_callback = ctypes.CFUNCTYPE(
    ctypes.c_bool, ctypes.c_float, ctypes.c_void_p
)
ggml_backend_sched_eval_callback = ctypes.CFUNCTYPE(
    ctypes.c_bool, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p
)
ggml_abort_callback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p)

##
## enums
##

# numa strategy
GGML_NUMA_STRATEGY_DISABLED = 0
GGML_NUMA_STRATEGY_DISTRIBUTE = 1
GGML_NUMA_STRATEGY_ISOLATE = 2
GGML_NUMA_STRATEGY_NUMACTL = 3
GGML_NUMA_STRATEGY_MIRROR = 4
GGML_NUMA_STRATEGY_COUNT = 5

# kv_override
LLAMA_KV_OVERRIDE_TYPE_INT = 0
LLAMA_KV_OVERRIDE_TYPE_FLOAT = 1
LLAMA_KV_OVERRIDE_TYPE_BOOL = 2
LLAMA_KV_OVERRIDE_TYPE_STR = 3

# pooling type
LLAMA_POOLING_TYPE_UNSPECIFIED = -1
LLAMA_POOLING_TYPE_NONE = 0
LLAMA_POOLING_TYPE_MEAN = 1
LLAMA_POOLING_TYPE_CLS = 2
LLAMA_POOLING_TYPE_LAST = 3   

##
## options
##

class llama_model_kv_override_value(ctypes.Union):
    _fields_ = [
        ("val_i64", ctypes.c_int64),
        ("val_f64", ctypes.c_double),
        ("val_bool", ctypes.c_bool),
        ("val_str", ctypes.c_char * 128),
    ]

class llama_model_kv_override(ctypes.Structure):
    _fields_ = [
        ("tag", ctypes.c_int),
        ("key", ctypes.c_char * 128),
        ("value", llama_model_kv_override_value),
    ]

##
## stucts
##

class llama_batch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(llama_token)),
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(llama_pos)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(llama_seq_id))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
        ("all_pos_0", llama_pos),
        ("all_pos_1", llama_pos),
        ("all_seq_id", llama_seq_id),
    ]

class llama_model_params(ctypes.Structure):
    _fields_ = [
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.POINTER(ctypes.c_float)),
        ("rpc_servers", ctypes.c_char_p),
        ("progress_callback", llama_progress_callback),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.POINTER(llama_model_kv_override)),
        ("vocab_only", ctypes.c_bool),
        ("use_mmap", ctypes.c_bool),
        ("use_mlock", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
    ]

class llama_context_params(ctypes.Structure):
    _fields_ = [
        ("seed", ctypes.c_uint32),
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_threads", ctypes.c_uint32),
        ("n_threads_batch", ctypes.c_uint32),
        ("rope_scaling_type", ctypes.c_int),
        ("pooling_type", ctypes.c_int),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ggml_backend_sched_eval_callback),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int),
        ("type_v", ctypes.c_int),
        ("logits_all", ctypes.c_bool),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("flash_attn", ctypes.c_bool),
        ("abort_callback", ggml_abort_callback),
        ("abort_callback_data", ctypes.c_void_p),
    ]

##
## functions
##

@ctypes_function()
def llama_backend_init(): ...

@ctypes_function(
    [ctypes.c_int]
)
def llama_numa_init(numa): ...

@ctypes_function(
    [],
    llama_model_params
)
def llama_model_default_params(): ...

@ctypes_function(
    [ctypes.c_char_p, llama_model_params],
    llama_model_p
)
def llama_load_model_from_file(path_model, params): ...

@ctypes_function(
    [],
    llama_context_params
)
def llama_context_default_params(): ...

@ctypes_function(
    [llama_model_p, llama_context_params],
    llama_context_p
)
def llama_new_context_with_model(model, params): ...

@ctypes_function(
    [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32],
    llama_batch
)
def llama_batch_init(n_tokens, embd, n_seq_max): ...

@ctypes_function(
    [llama_model_p]
)
def llama_free_model(model): ...

@ctypes_function(
    [llama_batch]
)
def llama_batch_free(batch): ...

@ctypes_function()
def llama_backend_free(): ...

@ctypes_function(
    [llama_context_p]
)
def llama_free(ctx): ...

@ctypes_function(
    [
        llama_model_p,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.POINTER(llama_token),
        ctypes.c_int32,
        ctypes.c_bool,
        ctypes.c_bool
    ],
    ctypes.c_int32
)
def llama_tokenize(
    model, text, text_len, tokens, n_tokens_max, add_special, parse_special
): ...

@ctypes_function(
    [llama_context_p, llama_batch],
    ctypes.c_int32
)
def llama_decode(ctx, batch): ...

@ctypes_function(
    [llama_context_p],
    ctypes.c_int
)
def llama_pooling_type(ctx): ...

@ctypes_function(
    [llama_context_p, ctypes.c_bool],
)
def llama_set_causal_attn(ctx, causal_attn): ...

@ctypes_function(
    [llama_model_p],
    ctypes.c_int32
)
def llama_n_embd(model): ...

@ctypes_function(
    [llama_context_p],
    ctypes.POINTER(ctypes.c_float)
)
def llama_get_embeddings(ctx, seq_id): ...

@ctypes_function(
    [llama_context_p, llama_seq_id],
    ctypes.POINTER(ctypes.c_float)
)
def llama_get_embeddings_seq(ctx, seq_id): ...
