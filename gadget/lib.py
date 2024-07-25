### llama.cpp bindings for embeddings
### mostly copied from llama-cpp-python

import os
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
def load_library(lib_path):
    try:
        return ctypes.CDLL(lib_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load shared library '{lib_path}': {e}")

# get shared library path
if 'GADGET_LLAMA_LIB' in os.environ:
    llama_path = os.environ['GADGET_LLAMA_LIB']
    ggml_path = os.environ['GADGET_GGML_LIB']
else:
    module_path = os.path.dirname(os.path.abspath(__file__))
    llama_path = os.path.join(module_path, 'libllama.so')
    ggml_path = os.path.join(module_path, 'libggml.so')

# load shared library
# _ggml = DummyLib()
# _llama = DummyLib()
_ggml = load_library(ggml_path)
_llama = load_library(llama_path)

##
## utils
##

def ctypes_function(library, argtypes=None, restype=None):
    if argtypes is None:
        argtypes = []
    def decorator(func):
        name = func.__name__
        func = getattr(library, name)
        func.argtypes = argtypes
        func.restype = restype
        return func
    return decorator

########################################
## ggml defs                          ##
########################################

##
## constants
##

GGML_MAX_DIMS      = 4
GGML_MAX_PARAMS    = 2048
GGML_MAX_CONTEXTS  = 64
GGML_MAX_SRC       = 10
GGML_MAX_NAME      = 64
GGML_MAX_OP_PARAMS = 64

##
## enums
##

GGML_STATUS_ALLOC_FAILED = -2
GGML_STATUS_FAILED       = -1
GGML_STATUS_SUCCESS      = 0
GGML_STATUS_ABORTED      = 1

# ggml_object_type
GGML_OBJECT_TYPE_TENSOR      = 0
GGML_OBJECT_TYPE_GRAPH       = 1
GGML_OBJECT_TYPE_WORK_BUFFER = 2

# ggml_type
GGML_TYPE_F32     = 0
GGML_TYPE_F16     = 1
GGML_TYPE_Q4_0    = 2
GGML_TYPE_Q4_1    = 3
# GGML_TYPE_Q4_2  = 4 support has been removed
# GGML_TYPE_Q4_3  = 5 support has been removed
GGML_TYPE_Q5_0    = 6
GGML_TYPE_Q5_1    = 7
GGML_TYPE_Q8_0    = 8
GGML_TYPE_Q8_1    = 9
GGML_TYPE_Q2_K    = 10
GGML_TYPE_Q3_K    = 11
GGML_TYPE_Q4_K    = 12
GGML_TYPE_Q5_K    = 13
GGML_TYPE_Q6_K    = 14
GGML_TYPE_Q8_K    = 15
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS  = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S   = 19
GGML_TYPE_IQ4_NL  = 20
GGML_TYPE_IQ3_S   = 21
GGML_TYPE_IQ2_S   = 22
GGML_TYPE_IQ4_XS  = 23
GGML_TYPE_I8      = 24
GGML_TYPE_I16     = 25
GGML_TYPE_I32     = 26
GGML_TYPE_I64     = 27
GGML_TYPE_F64     = 28
GGML_TYPE_IQ1_M   = 29
GGML_TYPE_BF16    = 30
GGML_TYPE_COUNT   = 31

##
## structs
##

class ggml_object(ctypes.Structure): ...
ggml_object_p = ctypes.POINTER(ggml_object)
ggml_object._fields_ = [
    ("offs"   , ctypes.c_size_t  ),
    ("size"   , ctypes.c_size_t  ),
    ("next"   , ggml_object_p    ),
    ("type"   , ctypes.c_int     ),
    ("padding", ctypes.c_char * 4),
]

class ggml_scratch(ctypes.Structure):
    _fields_ = [
        ("offs", ctypes.c_size_t),
        ("size", ctypes.c_size_t),
        ("data", ctypes.c_void_p),
    ]

class ggml_init_params(ctypes.Structure):
    _fields_ = [
        ("mem_size"  , ctypes.c_size_t),
        ("mem_buffer", ctypes.c_void_p),
        ("no_alloc"  , ctypes.c_bool  ),
    ]

class ggml_context(ctypes.Structure):
    _fields_ = [
        ("mem_size"        , ctypes.c_size_t),
        ("mem_buffer"      , ctypes.c_void_p),
        ("mem_buffer_owned", ctypes.c_bool  ),
        ("no_alloc"        , ctypes.c_bool  ),
        ("no_alloc_save"   , ctypes.c_bool  ),
        ("n_objects"       , ctypes.c_int   ),
        ("objects_begin"   , ggml_object_p  ),
        ("objects_end"     , ggml_object_p  ),
        ("scratch"         , ggml_scratch   ),
        ("scratch_save"    , ggml_scratch   ),
    ]
ggml_context_p = ctypes.POINTER(ggml_context)

# interlinked tensor/buffer classes
class ggml_tensor                     (ctypes.Structure): ...
class ggml_backend_buffer_i           (ctypes.Structure): ...
class ggml_backend_buffer             (ctypes.Structure): ...
class ggml_backend_buffer_type_i      (ctypes.Structure): ...
class ggml_backend_buffer_type        (ctypes.Structure): ...
class ggml_backend_buffer_type_context(ctypes.Structure): ...

# pointer convenience vars
ggml_tensor_p                      = ctypes.POINTER(ggml_tensor)
ggml_backend_buffer_p              = ctypes.POINTER(ggml_backend_buffer)
ggml_backend_buffer_type_p         = ctypes.POINTER(ggml_backend_buffer_type)
ggml_backend_buffer_context_p      = ctypes.c_void_p
ggml_backend_buffer_type_context_p = ctypes.c_void_p

ggml_backend_buffer_i._fields_ = [
    ("get_name"   , ctypes.CFUNCTYPE(ctypes.c_char_p, ggml_backend_buffer_p                                                                  )),
    ("free_buffer", ctypes.CFUNCTYPE(None           , ggml_backend_buffer_p                                                                  )),
    ("get_base"   , ctypes.CFUNCTYPE(ctypes.c_void_p, ggml_backend_buffer_p                                                                  )),
    ("init_tensor", ctypes.CFUNCTYPE(None           , ggml_backend_buffer_p, ggml_tensor_p                                                   )),
    ("set_tensor" , ctypes.CFUNCTYPE(None           , ggml_backend_buffer_p, ggml_tensor_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t)),
    ("get_tensor" , ctypes.CFUNCTYPE(None           , ggml_backend_buffer_p, ggml_tensor_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t)),
    ("cpy_tensor" , ctypes.CFUNCTYPE(ctypes.c_bool  , ggml_backend_buffer_p, ggml_tensor_p, ggml_tensor_p                                    )),
    ("clear"      , ctypes.CFUNCTYPE(None           , ggml_backend_buffer_p, ctypes.c_uint8                                                  )),
    ("reset"      , ctypes.CFUNCTYPE(None           , ggml_backend_buffer_p                                                                  )),
]

ggml_backend_buffer._fields_ = [
    ("iface"  , ggml_backend_buffer_i        ),
    ("buft"   , ggml_backend_buffer_type_p   ),
    ("context", ggml_backend_buffer_context_p),
    ("size"   , ctypes.c_size_t              ),
    ("usage"  , ctypes.c_int                 ),
]

ggml_backend_buffer_type_i._fields_ = [
    ("get_name"      , ctypes.CFUNCTYPE(ctypes.c_char_p      , ggml_backend_buffer_type_p                 )),
    ("alloc_buffer"  , ctypes.CFUNCTYPE(ggml_backend_buffer_p, ggml_backend_buffer_type_p, ctypes.c_size_t)),
    ("get_alignment" , ctypes.CFUNCTYPE(ctypes.c_size_t      , ggml_backend_buffer_type_p                 )),
    ("get_max_size"  , ctypes.CFUNCTYPE(ctypes.c_size_t      , ggml_backend_buffer_type_p                 )),
    ("get_alloc_size", ctypes.CFUNCTYPE(ctypes.c_size_t      , ggml_backend_buffer_type_p, ggml_tensor_p  )),
    ("is_host"       , ctypes.CFUNCTYPE(ctypes.c_bool        , ggml_backend_buffer_type_p                 )),
]

ggml_backend_buffer_type._fields_ = [
    ("iface"  , ggml_backend_buffer_type_i        ),
    ("context", ggml_backend_buffer_type_context_p),
]

GGML_MAX_OP_PARAMS_INT = GGML_MAX_OP_PARAMS // ctypes.sizeof(ctypes.c_int32)
ggml_tensor._fields_ = [
    ("type"     , ctypes.c_int                           ),
    ("buffer"   , ggml_backend_buffer_p                  ),
    ("ne"       , ctypes.c_int64 * GGML_MAX_DIMS         ),
    ("nb"       , ctypes.c_size_t * GGML_MAX_DIMS        ),
    ("op"       , ctypes.c_int                           ),
    ("op_params", ctypes.c_int32 * GGML_MAX_OP_PARAMS_INT),
    ("flags"    , ctypes.c_int32                         ),
    ("grad"     , ggml_tensor_p                          ),
    ("src"      , ggml_tensor_p * GGML_MAX_SRC           ),
    ("view_src" , ggml_tensor_p                          ),
    ("view_offs", ctypes.c_size_t                        ),
    ("data"     , ctypes.c_void_p                        ),
    ("name"     , ctypes.c_char * GGML_MAX_NAME          ),
    ("extra"    , ctypes.c_void_p                        ),
]

class ggml_hash_set(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t              ),
        ("keys", ctypes.POINTER(ggml_tensor_p)),
    ]

class ggml_cgraph(ctypes.Structure):
    _fields_ = [
        ("size"              , ctypes.c_int                 ),
        ("n_nodes"           , ctypes.c_int                 ),
        ("n_leafs"           , ctypes.c_int                 ),
        ("nodes"             , ctypes.POINTER(ggml_tensor_p)),
        ("grads"             , ctypes.POINTER(ggml_tensor_p)),
        ("leafs"             , ctypes.POINTER(ggml_tensor_p)),
        ("visited_hash_table", ggml_hash_set                ),
        ("order"             , ctypes.c_int                 ),
    ]
ggml_cgraph_p = ctypes.POINTER(ggml_cgraph)

##
## functions
##

@ctypes_function(_ggml,
    [ctypes.c_int],
    ctypes.c_size_t
)
def ggml_type_size(type): ...

@ctypes_function(_ggml,
    None,
    ctypes.c_size_t,
)
def ggml_tensor_overhead(): ...

@ctypes_function(_ggml,
    None,
    ctypes.c_size_t,
)
def ggml_graph_overhead(): ...

@ctypes_function(_ggml,
    [ggml_init_params],
    ggml_context_p
)
def  ggml_init(params): ...

@ctypes_function(_ggml,
    [
        ggml_context_p,
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int64
    ],
    ggml_tensor_p
)
def ggml_new_tensor_2d(ctx, type, ne0, ne1): ...

@ctypes_function(_ggml,
    [ggml_context_p],
    ggml_cgraph_p
)
def ggml_new_graph(ctx): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_mul_mat(ctx, a, b): ...

@ctypes_function(_ggml,
    [ggml_cgraph_p, ggml_tensor_p],
)
def ggml_build_forward_expand(cgraph, tensor): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_cgraph_p, ctypes.c_int],
    ctypes.c_int
)
def ggml_graph_compute_with_ctx(ctx, cgraph, n_threads): ...

########################################
## llama defs                         ##
########################################

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
GGML_NUMA_STRATEGY_DISABLED   = 0
GGML_NUMA_STRATEGY_DISTRIBUTE = 1
GGML_NUMA_STRATEGY_ISOLATE    = 2
GGML_NUMA_STRATEGY_NUMACTL    = 3
GGML_NUMA_STRATEGY_MIRROR     = 4
GGML_NUMA_STRATEGY_COUNT      = 5

# kv_override
LLAMA_KV_OVERRIDE_TYPE_INT   = 0
LLAMA_KV_OVERRIDE_TYPE_FLOAT = 1
LLAMA_KV_OVERRIDE_TYPE_BOOL  = 2
LLAMA_KV_OVERRIDE_TYPE_STR   = 3

# pooling type
LLAMA_POOLING_TYPE_UNSPECIFIED = -1
LLAMA_POOLING_TYPE_NONE        = 0
LLAMA_POOLING_TYPE_MEAN        = 1
LLAMA_POOLING_TYPE_CLS         = 2
LLAMA_POOLING_TYPE_LAST        = 3

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
## structs
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

@ctypes_function(_llama)
def llama_backend_init(): ...

@ctypes_function(_llama,
    [ctypes.c_int]
)
def llama_numa_init(numa): ...

@ctypes_function(_llama,
    [],
    llama_model_params
)
def llama_model_default_params(): ...

@ctypes_function(_llama,
    [ctypes.c_char_p, llama_model_params],
    llama_model_p
)
def llama_load_model_from_file(path_model, params): ...

@ctypes_function(_llama,
    [],
    llama_context_params
)
def llama_context_default_params(): ...

@ctypes_function(_llama,
    [llama_model_p, llama_context_params],
    llama_context_p
)
def llama_new_context_with_model(model, params): ...

@ctypes_function(_llama,
    [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32],
    llama_batch
)
def llama_batch_init(n_tokens, embd, n_seq_max): ...

@ctypes_function(_llama,
    [llama_model_p]
)
def llama_free_model(model): ...

@ctypes_function(_llama,
    [llama_batch]
)
def llama_batch_free(batch): ...

@ctypes_function(_llama)
def llama_backend_free(): ...

@ctypes_function(_llama,
    [llama_context_p]
)
def llama_free(ctx): ...

@ctypes_function(_llama,
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

@ctypes_function(_llama,
    [llama_context_p, llama_batch],
    ctypes.c_int32
)
def llama_decode(ctx, batch): ...

@ctypes_function(_llama,
    [llama_context_p],
    ctypes.c_int
)
def llama_pooling_type(ctx): ...

@ctypes_function(_llama,
    [llama_context_p, ctypes.c_bool],
)
def llama_set_causal_attn(ctx, causal_attn): ...

@ctypes_function(_llama,
    [llama_model_p],
    ctypes.c_int32
)
def llama_n_embd(model): ...

@ctypes_function(_llama,
    [llama_context_p],
    ctypes.POINTER(ctypes.c_float)
)
def llama_get_embeddings(ctx, seq_id): ...

@ctypes_function(_llama,
    [llama_context_p, llama_seq_id],
    ctypes.POINTER(ctypes.c_float)
)
def llama_get_embeddings_seq(ctx, seq_id): ...
