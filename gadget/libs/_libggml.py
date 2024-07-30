### ggml bindings

import os
import ctypes

##
## library
##

# get shared library path
if 'GADGET_GGML_LIB' in os.environ:
    ggml_path = os.environ['GADGET_GGML_LIB']
else:
    module_path = os.path.dirname(os.path.abspath(__file__))
    ggml_path = os.path.join(module_path, 'libggml.so')

# load shared library
try:
    _ggml = ctypes.CDLL(ggml_path)
except Exception as e:
    raise RuntimeError(f"Failed to load shared library '{ggml_path}': {e}")

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

##
## constants
##

GGML_MAX_DIMS      = 4
GGML_MAX_PARAMS    = 2048
GGML_MAX_CONTEXTS  = 64
GGML_MAX_SRC       = 10
GGML_MAX_NAME      = 64
GGML_MAX_OP_PARAMS = 64

GGML_DEFAULT_GRAPH_SIZE = 2048
MAX_FREE_BLOCKS = 256

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

# graph construction
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

# backend and backend context
class ggml_backend_i    (ctypes.Structure): ...
class ggml_backend      (ctypes.Structure): ...
class ggml_backend_event(ctypes.Structure): ...

# pointer convenience variables
ggml_guid                 = ctypes.c_uint8 * 16
ggml_guid_p               = ctypes.POINTER(ggml_guid)
ggml_backend_p            = ctypes.POINTER(ggml_backend)
ggml_backend_event_p      = ctypes.POINTER(ggml_backend_event)
ggml_backend_context_p    = ctypes.c_void_p
ggml_backend_graph_plan_p = ctypes.c_void_p

ggml_backend_i._fields_ = [
    ("get_name"               , ctypes.CFUNCTYPE(ctypes.c_char_p           , ggml_backend_p                                                                  )),
    ("free"                   , ctypes.CFUNCTYPE(None                      , ggml_backend_p                                                                  )),
    ("get_default_buffer_type", ctypes.CFUNCTYPE(ggml_backend_buffer_type_p, ggml_backend_p                                                                  )),
    ("set_tensor_async"       , ctypes.CFUNCTYPE(None                      , ggml_backend_p, ggml_tensor_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t)),
    ("get_tensor_async"       , ctypes.CFUNCTYPE(None                      , ggml_backend_p, ggml_tensor_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t)),
    ("cpy_tensor_async"       , ctypes.CFUNCTYPE(ctypes.c_bool             , ggml_backend_p, ggml_backend_p, ggml_tensor_p, ggml_tensor_p                    )),
    ("synchronize"            , ctypes.CFUNCTYPE(None                      , ggml_backend_p                                                                  )),
    ("graph_plan_create"      , ctypes.CFUNCTYPE(ggml_backend_graph_plan_p , ggml_backend_p, ggml_cgraph_p                                                   )),
    ("graph_plan_free"        , ctypes.CFUNCTYPE(None                      , ggml_backend_p, ggml_backend_graph_plan_p                                       )),
    ("graph_plan_update"      , ctypes.CFUNCTYPE(None                      , ggml_backend_p, ggml_backend_graph_plan_p, ggml_cgraph_p                        )),
    ("graph_plan_compute"     , ctypes.CFUNCTYPE(ctypes.c_int              , ggml_backend_p, ggml_backend_graph_plan_p                                       )),
    ("graph_compute"          , ctypes.CFUNCTYPE(ctypes.c_int              , ggml_backend_p, ggml_cgraph_p                                                   )),
    ("supports_op"            , ctypes.CFUNCTYPE(ctypes.c_bool             , ggml_backend_p, ggml_tensor_p                                                   )),
    ("supports_buft"          , ctypes.CFUNCTYPE(ctypes.c_bool             , ggml_backend_p, ggml_backend_buffer_type_p                                      )),
    ("offload_op"             , ctypes.CFUNCTYPE(ctypes.c_bool             , ggml_backend_p, ggml_tensor_p                                                   )),
    ("event_new"              , ctypes.CFUNCTYPE(ggml_backend_event_p      , ggml_backend_p                                                                  )),
    ("event_free"             , ctypes.CFUNCTYPE(None                      , ggml_backend_event_p                                                            )),
    ("event_record"           , ctypes.CFUNCTYPE(None                      , ggml_backend_event_p                                                            )),
    ("event_wait"             , ctypes.CFUNCTYPE(None                      , ggml_backend_p, ggml_backend_event_p                                            )),
    ("event_synchronize"      , ctypes.CFUNCTYPE(None                      , ggml_backend_event_p                                                            )),
]

ggml_backend._fields_ = [
    ("guid"   , ggml_guid_p           ),
    ("iface"  , ggml_backend_i        ),
    ("context", ggml_backend_context_p),
]

ggml_backend_event._fields_ = [
    ("backend", ggml_backend_p ),
    ("context", ctypes.c_void_p),
]

class hash_node(ctypes.Structure):
    _fields_ = [
        ("n_children", ctypes.c_int   ),
        ("n_views"   , ctypes.c_int   ),
        ("buffer_id" , ctypes.c_int   ),
        ("offset"    , ctypes.c_size_t),
        ("allocated" , ctypes.c_bool  ),
    ]
hash_node_p        = ctypes.POINTER(hash_node)

class tensor_alloc(ctypes.Structure):
    _fields_ = [
        ("buffer_id", ctypes.c_int   ),
        ("offset"   , ctypes.c_size_t),
        ("size_max" , ctypes.c_size_t),
    ]
tensor_alloc_p     = ctypes.POINTER(tensor_alloc)

class leaf_alloc(ctypes.Structure):
    _fields_ = [
        ("buffer_id", ctypes.c_int),
        ("leaf"     , tensor_alloc),
    ]
leaf_alloc_p       = ctypes.POINTER(leaf_alloc)

class node_alloc(ctypes.Structure):
    _fields_ = [
        ("dst", tensor_alloc               ),
        ("src", tensor_alloc * GGML_MAX_SRC),
    ]
node_alloc_p       = ctypes.POINTER(node_alloc)

class free_block(ctypes.Structure):
    _fields_ = [
        ("offset", ctypes.c_size_t),
        ("size"  , ctypes.c_size_t),
    ]
free_block_p       = ctypes.POINTER(free_block)

class ggml_dyn_tallocr(ctypes.Structure):
    _fields_ = [
        ("alignment"    , ctypes.c_size_t             ),
        ("n_free_blocks", ctypes.c_int                ),
        ("free_blocks"  , free_block * MAX_FREE_BLOCKS),
        ("max_size"     , ctypes.c_size_t             ),
    ]
ggml_dyn_tallocr_p = ctypes.POINTER(ggml_dyn_tallocr)

class ggml_gallocr(ctypes.Structure):
    _fields_ = [
        ("bufts"      , ggml_backend_buffer_type_p        ),
        ("buffers"    , ggml_backend_buffer_p             ),
        ("buf_tallocs", ctypes.POINTER(ggml_dyn_tallocr_p)),
        ("n_buffers"  , ctypes.c_int                      ),
        ("hash_set"   , ggml_hash_set                     ),
        ("hash_values", hash_node_p                       ),
        ("node_allocs", node_alloc_p                      ),
        ("n_nodes"    , ctypes.c_int                      ),
        ("leaf_allocs", leaf_alloc_p                      ),
        ("n_leafs"    , ctypes.c_int                      ),
    ]
ggml_gallocr_p = ctypes.POINTER(ggml_gallocr)

##
## functions
##

@ctypes_function(_ggml,
    None,
    ggml_backend_p
)
def ggml_backend_cpu_init(): ...

@ctypes_function(_ggml,
    None,
    ggml_backend_p
)
def ggml_backend_free(backend): ...

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
    ctypes.c_size_t
)
def ggml_graph_overhead(): ...

@ctypes_function(_ggml,
    [ggml_init_params],
    ggml_context_p
)
def  ggml_init(params): ...

@ctypes_function(_ggml,
    [ggml_context_p],
    None
)
def ggml_free(ctx): ...

@ctypes_function(_ggml,
    [ggml_backend_p],
    ggml_backend_buffer_type_p
)
def ggml_backend_get_default_buffer_type(backend): ...

@ctypes_function(_ggml,
    [ggml_backend_p, ctypes.c_int],
    None
)
def ggml_backend_cpu_set_n_threads(backend_cpu, n_threads): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_backend_p],
    ggml_backend_buffer_p
)
def ggml_backend_alloc_ctx_tensors(): ...

@ctypes_function(_ggml,
    [ggml_backend_p, ggml_cgraph_p],
    ctypes.c_int
)
def ggml_backend_graph_compute(backend, cgraph): ...

@ctypes_function(_ggml,
    [ggml_backend_buffer_type_p],
    ggml_gallocr_p
)
def ggml_gallocr_new(buft): ...

@ctypes_function(_ggml,
    [ggml_gallocr_p, ggml_cgraph_p],
    ctypes.c_bool
)
def ggml_gallocr_reserve(galloc, graph): ...

@ctypes_function(_ggml,
    [ggml_gallocr_p, ctypes.c_int],
    ctypes.c_size_t
)
def ggml_gallocr_get_buffer_size(galloc, buffer_id): ...

@ctypes_function(_ggml,
    [ggml_gallocr_p, ggml_cgraph_p],
    ctypes.c_bool
)
def ggml_gallocr_alloc_graph(galloc, graph): ...

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
