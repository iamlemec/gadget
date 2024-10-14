### ggml bindings

import os
import ctypes
from operator import mul
from itertools import accumulate
from math import prod

from .general import load_shared_lib, ctypes_function, DummyFunction
from .constants import GGMLQuantizationType, GGML_MAX_DIMS

##
## library
##

_ggml = load_shared_lib('libggml.so', 'GADGET_GGML_LIB')

##
## function wrappers
##

ttype_size = {
    GGMLQuantizationType.BF16: 2,
    GGMLQuantizationType.F16: 2,
    GGMLQuantizationType.F32: 4,
    GGMLQuantizationType.F64: 8,
    GGMLQuantizationType.I8: 1,
    GGMLQuantizationType.I16: 2,
    GGMLQuantizationType.I32: 4,
    GGMLQuantizationType.I64: 8,
}

def cumprod(arr):
    return list(accumulate(arr, mul))

def get_tensor_type(tensor):
    value = tensor.contents
    return GGMLQuantizationType(value.type)

def get_tensor_nelem(tensor):
    value = tensor.contents
    return tuple(value.ne[:4])

def get_tensor_strides(tensor):
    value = tensor.contents
    return tuple(value.nb[:4])

def get_tensor_block(tensor):
    value = tensor.contents
    return value.blk_size

def get_arg_info(a):
    if type(a) is ggml_tensor_p:
        ty = get_tensor_type(a)
        ne = get_tensor_nelem(a)
        return f'ggml_tensor: {ty.name} × {ne}'
    else:
        return f'{type(a).__name__}: {a}'

def get_input_info(*args):
    return '\n'.join([str(get_arg_info(a)) for a in args])

# NOTE: this only handles non-quantized tensors
def is_contiguous(tensor):
    ne = get_tensor_nelem(tensor)
    nb = get_tensor_strides(tensor)
    tt = get_tensor_type(tensor)
    ts = ttype_size[tt]
    return nb == tuple(cumprod((ts,) + ne)[:-1])

def named_output(func):
    def wrapper(*args, name=None):
        c = func(*args)
        if name is not None:
            ggml_set_name(c, name.encode('utf-8'))
        return c
    return wrapper

def check_inputs(check):
    def outer(func):
        def inner(ctx, *args):
            if not check(*args):
                raise ValueError(
                    f'{func.__name__}: invalid arguments\n{get_input_info(*args)}'
                )
            return func(ctx, *args)
        return inner
    return outer

##
## pre-emptive type checking and naming
## some of these are copies of inlined functions
##

def ggml_can_mul_mat(t0, t1):
    ne0 = get_tensor_nelem(t0)
    ne1 = get_tensor_nelem(t1)
    nb0 = get_tensor_strides(t0)
    return (
        (ne0[0] == ne1[0]     ) and
        (ne1[2]  % ne0[2] == 0) and
        (ne1[3]  % ne0[3] == 0) and
        (nb0[0] <= nb0[1])
    )

def ggml_can_add(t0, t1):
    ne0 = get_tensor_nelem(t0)
    ne1 = get_tensor_nelem(t1)
    return (
        (ne0[0] % ne1[0] == 0) and
        (ne0[1] % ne1[1] == 0) and
        (ne0[2] % ne1[2] == 0) and
        (ne0[3] % ne1[3] == 0)
    )

def ggml_can_get_rows(t0, t1):
    ne0 = get_tensor_nelem(t0)
    ne1 = get_tensor_nelem(t1)
    ttype1 = get_tensor_type(t1)
    return (
        (ne0[2] == ne1[1]) and
        (ne1[3] == 1     ) and
        (ttype1 == GGMLQuantizationType.I32)
    )

def ggml_can_reshape_3d(t, ne0, ne1, ne2):
    ne = get_tensor_nelem(t)
    return prod(ne) == ne0 * ne1 * ne2

def ggml_can_view_3d(t, ne0, ne1, ne2, nb1, nb2, offset):
    ne = get_tensor_nelem(t)
    return prod(ne) == ne0 * ne1 * ne2

def ggml_can_cpy(t0, t1):
    ne0 = get_tensor_nelem(t0)
    ne1 = get_tensor_nelem(t1)
    return prod(ne0) == prod(ne1)

def ggml_can_soft_max_ext(t, mask, head_wgt, alibi):
    return is_contiguous(mask)

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

# types and quantization
ggml_to_float_p          = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)
ggml_from_float_p        = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int64)
ggml_from_float_to_mat_p = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
ggml_vec_dot_p           = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
ggml_vec_dot_p           = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
ggml_gemv_p              = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
ggml_gemm_p              = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int)

class ggml_type_traits(ctypes.Structure):
    _fields_ = [
        ("type_name"           , ctypes.c_char_p         ),
        ("blck_size"           , ctypes.c_int            ),
        ("blck_size_interleave", ctypes.c_int64          ),
        ("type_size"           , ctypes.c_size_t         ),
        ("is_quantized"        , ctypes.c_bool           ),
        ("to_float"            , ggml_to_float_p         ),
        ("from_float"          , ggml_from_float_p       ),
        ("from_float_ref"      , ggml_from_float_p       ),
        ("from_float_to_mat"   , ggml_from_float_to_mat_p),
        ("vec_dot"             , ggml_vec_dot_p          ),
        ("vec_dot_type"        , ctypes.c_int            ),
        ("nrows"               , ctypes.c_int64          ),
        ("ncols"               , ctypes.c_int64          ),
        ("gemv"                , ggml_gemv_p             ),
        ("gemm"                , ggml_gemm_p             ),
    ]
ggml_type_traits_p = ctypes.POINTER(ggml_type_traits)

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

## initialization

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

## backend

@ctypes_function(_ggml,
    None,
    ggml_backend_p
)
def ggml_backend_cpu_init(): ...

try:
    @ctypes_function(_ggml,
        [ctypes.c_int],
        ggml_backend_p
    )
    def ggml_backend_cuda_init(): ...
except:
    ggml_backend_cuda_init = DummyFunction('CUDA backend not found')

@ctypes_function(_ggml,
    None,
    ggml_backend_p
)
def ggml_backend_free(backend): ...

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
    [ggml_tensor_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t],
    None
)
def ggml_backend_tensor_set(tensor, data, offset, size): ...

@ctypes_function(_ggml,
    [ggml_tensor_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t],
    None
)
def ggml_backend_tensor_get(tensor, data, offset, size): ...

@ctypes_function(_ggml,
    [ggml_tensor_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t],
    None
)
def ggml_backend_tensor_set_async(tensor, data, offset, size): ...

@ctypes_function(_ggml,
    [ggml_tensor_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t],
    None
)
def ggml_backend_tensor_get_async(tensor, data, offset, size): ...

@ctypes_function(_ggml,
    [ggml_backend_buffer_p],
    ctypes.c_bool
)
def ggml_backend_buffer_is_host(buffer): ...

## allocation

@ctypes_function(_ggml,
    [ggml_backend_buffer_type_p],
    ggml_gallocr_p
)
def ggml_gallocr_new(buft): ...

@ctypes_function(_ggml,
    [ggml_gallocr_p],
    None
)
def ggml_gallocr_free(galloc): ...

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

## tensors

@ctypes_function(_ggml,
    [ggml_context_p, ctypes.c_int, ctypes.c_int64],
    ggml_tensor_p
)
def ggml_new_tensor_1d(ctx, type, ne0): ...

@ctypes_function(_ggml,
    [ggml_context_p, ctypes.c_int, ctypes.c_int64, ctypes.c_int64],
    ggml_tensor_p
)
def ggml_new_tensor_2d(ctx, type, ne0, ne1): ...

@ctypes_function(_ggml,
    [ggml_context_p, ctypes.c_int, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64],
    ggml_tensor_p
)
def ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2): ...

@ctypes_function(_ggml,
    [ggml_context_p, ctypes.c_int, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64],
    ggml_tensor_p
)
def ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3): ...

@ctypes_function(_ggml,
    [ggml_tensor_p, ctypes.c_char_p],
    None
)
def ggml_set_name(tensor, name): ...

@ctypes_function(_ggml,
    [ggml_tensor_p],
    ctypes.c_size_t
)
def ggml_element_size(tensor): ...

@ctypes_function(_ggml,
    [ggml_tensor_p],
    ctypes.c_int64
)
def ggml_nelements(tensor): ...

@ctypes_function(_ggml,
    [ggml_tensor_p],
    ctypes.c_int64
)
def ggml_nbytes(tensor): ...

@ctypes_function(_ggml,
    [ctypes.c_int],
    ctypes.c_bool
)
def ggml_is_quantized(tensor): ...

@ctypes_function(_ggml,
    [ggml_tensor_p],
    ctypes.c_bool
)
def ggml_is_transposed(tensor): ...

@ctypes_function(_ggml,
    [ggml_tensor_p],
    ctypes.c_bool
)
def ggml_is_contiguous(tensor): ...

@ctypes_function(_ggml,
    [ctypes.c_int],
    ggml_type_traits
)
def ggml_internal_get_type_traits(ttype): ...

## graphs

@ctypes_function(_ggml,
    [ggml_context_p],
    ggml_cgraph_p
)
def ggml_new_graph(ctx): ...

@ctypes_function(_ggml,
    [ggml_cgraph_p, ggml_tensor_p],
)
def ggml_build_forward_expand(cgraph, tensor): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_cgraph_p, ctypes.c_int],
    ctypes.c_int
)
def ggml_graph_compute_with_ctx(ctx, cgraph, n_threads): ...

## tensor ops

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_dup(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_dup_inplace(ctx, a): ...

@named_output
@check_inputs(ggml_can_add)
@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_add(ctx, a, b): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_add_inplace(ctx, a, b): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ctypes.c_int],
    ggml_tensor_p
)
def ggml_add_cast(ctx, a, b, type): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_add1(ctx, a, b): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_add1_inplace(ctx, a, b): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t],
    ggml_tensor_p
)
def ggml_acc(ctx, a, b, nb1, nb2, nb3, offset): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t],
    ggml_tensor_p
)
def ggml_acc_inplace(ctx, a, b, nb1, nb2, nb3, offset): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_sub(ctx, a, b): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_sub_inplace(ctx, a, b): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_mul(ctx, a, b): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_mul_inplace(ctx, a, b): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_div(ctx, a, b): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_div_inplace(ctx, a, b): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_sqr(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_sqr_inplace(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_sqrt(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_sqrt_inplace(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_log(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_log_inplace(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_sum(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_sum_rows(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_mean(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_argmax(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_repeat(ctx, a, b): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_repeat_back(ctx, a, b): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ctypes.c_int],
    ggml_tensor_p
)
def ggml_concat(ctx, a, b, dim): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_abs(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_abs_inplace(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_sgn(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_sgn_inplace(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_neg(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_neg_inplace(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_step(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_step_inplace(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_tanh(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_tanh_inplace(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_elu(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_elu_inplace(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_relu(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_float, ctypes.c_bool],
    ggml_tensor_p
)
def ggml_leaky_relu(ctx, a, negative_slope, inplace): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_relu_inplace(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_sigmoid(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_sigmoid_inplace(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_gelu(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_gelu_inplace(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_gelu_quick(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_gelu_quick_inplace(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_silu(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_silu_inplace(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_silu_back(ctx, a, b): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_hardswish(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_hardsigmoid(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_float],
    ggml_tensor_p
)
def ggml_norm(ctx, a, eps): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_float],
    ggml_tensor_p
)
def ggml_norm_inplace(ctx, a, eps): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_float],
    ggml_tensor_p
)
def ggml_rms_norm(ctx, a, eps): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_float],
    ggml_tensor_p
)
def ggml_rms_norm_inplace(ctx, a, eps): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int],
    ggml_tensor_p
)
def ggml_group_norm(ctx, a, n_groups): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int],
    ggml_tensor_p
)
def ggml_group_norm_inplace(ctx, a, n_groups): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ctypes.c_float],
    ggml_tensor_p
)
def ggml_rms_norm_back(ctx, a, b, eps): ...

@named_output
@check_inputs(ggml_can_mul_mat)
@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_mul_mat(ctx, a, b): ...

@ctypes_function(_ggml,
    [ggml_tensor_p, ctypes.c_int],
    None
)
def ggml_mul_mat_set_prec(a, prec): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_mul_mat_id(ctx, as_, b, ids): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_out_prod(ctx, a, b): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_float],
    ggml_tensor_p
)
def ggml_scale(ctx, a, s): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_float],
    ggml_tensor_p
)
def ggml_scale_inplace(ctx, a, s): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t],
    ggml_tensor_p
)
def ggml_set(ctx, a, b, nb1, nb2, nb3, offset): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t],
    ggml_tensor_p
)
def ggml_set_inplace(ctx, a, b, nb1, nb2, nb3, offset): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ctypes.c_size_t],
    ggml_tensor_p
)
def ggml_set_1d(ctx, a, b, offset): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ctypes.c_size_t],
    ggml_tensor_p
)
def ggml_set_1d_inplace(ctx, a, b, offset): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ctypes.c_size_t, ctypes.c_size_t],
    ggml_tensor_p
)
def ggml_set_2d(ctx, a, b, nb1, offset): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ctypes.c_size_t, ctypes.c_size_t],
    ggml_tensor_p
)
def ggml_set_2d_inplace(ctx, a, b, nb1, offset): ...

@named_output
@check_inputs(ggml_can_cpy)
@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_cpy(ctx, a, b): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int],
    ggml_tensor_p
)
def ggml_cast(ctx, a, type): ...

@named_output
@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_cont(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int64],
    ggml_tensor_p
)
def ggml_cont_1d(ctx, a, ne0): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int64, ctypes.c_int64],
    ggml_tensor_p
)
def ggml_cont_2d(ctx, a, ne0, ne1): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64],
    ggml_tensor_p
)
def ggml_cont_3d(ctx, a, ne0, ne1, ne2): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64],
    ggml_tensor_p
)
def ggml_cont_4d(ctx, a, ne0, ne1, ne2, ne3): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_reshape(ctx, a, b): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int64],
    ggml_tensor_p
)
def ggml_reshape_1d(ctx, a, ne0): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int64, ctypes.c_int64],
    ggml_tensor_p
)
def ggml_reshape_2d(ctx, a, ne0, ne1): ...

@named_output
@check_inputs(ggml_can_reshape_3d)
@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64],
    ggml_tensor_p
)
def ggml_reshape_3d(ctx, a, ne0, ne1, ne2): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64],
    ggml_tensor_p
)
def ggml_reshape_4d(ctx, a, ne0, ne1, ne2, ne3): ...

@named_output
@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int64, ctypes.c_size_t],
    ggml_tensor_p
)
def ggml_view_1d(ctx, a, ne0, offset): ...

@named_output
@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_size_t, ctypes.c_size_t],
    ggml_tensor_p
)
def ggml_view_2d(ctx, a, ne0, ne1, nb1, offset): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t],
    ggml_tensor_p
)
def ggml_view_3d(ctx, a, ne0, ne1, ne2, nb1, nb2, offset): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t],
    ggml_tensor_p
)
def ggml_view_4d(ctx, a, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ggml_tensor_p
)
def ggml_permute(ctx, a, axis0, axis1, axis2, axis3): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_transpose(ctx, a): ...

@named_output
@check_inputs(ggml_can_get_rows)
@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_get_rows(ctx, a, b): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_get_rows_back(ctx, a, b, c): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_diag(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int],
    ggml_tensor_p
)
def ggml_diag_mask_inf(ctx, a, n_past): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int],
    ggml_tensor_p
)
def ggml_diag_mask_inf_inplace(ctx, a, n_past): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int],
    ggml_tensor_p
)
def ggml_diag_mask_zero(ctx, a, n_past): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int],
    ggml_tensor_p
)
def ggml_diag_mask_zero_inplace(ctx, a, n_past): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_soft_max(ctx, a): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_soft_max_inplace(ctx, a): ...

@named_output
@check_inputs(ggml_can_soft_max_ext)
@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ctypes.c_float, ctypes.c_float],
    ggml_tensor_p
)
def ggml_soft_max_ext(ctx, a, mask, scale, max_bias): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_soft_max_back(ctx, a, b): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_soft_max_back_inplace(ctx, a, b): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ctypes.c_int, ctypes.c_int],
    ggml_tensor_p
)
def ggml_rope(ctx, a, b, n_dims, mode): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ctypes.c_int, ctypes.c_int],
    ggml_tensor_p
)
def ggml_rope_inplace(ctx, a, b, n_dims, mode): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ggml_tensor_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float],
    ggml_tensor_p
)
def ggml_rope_ext(ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ggml_tensor_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float],
    ggml_tensor_p
)
def ggml_rope_ext_inplace(ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow): ...

@ctypes_function(_ggml,
    [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_float)],
    None
)
def ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, dims): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ggml_tensor_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float],
    ggml_tensor_p
)
def ggml_rope_back(ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_float, ctypes.c_float],
    ggml_tensor_p
)
def ggml_clamp(ctx, a, min, max): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_int],
    ggml_tensor_p
)
def ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, is_2D, dst_type): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ggml_tensor_p
)
def ggml_conv_depthwise_2d(ctx, a, b, s0, s1, p0, p1, d0, d1): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ggml_tensor_p
)
def ggml_conv_1d(ctx, a, b, s0, p0, d0): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ctypes.c_int, ctypes.c_int],
    ggml_tensor_p
)
def ggml_conv_1d_ph(ctx, a, b, s, d): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ggml_tensor_p
)
def ggml_conv_transpose_1d(ctx, a, b, s0, p0, d0): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ggml_tensor_p
)
def ggml_conv_2d(ctx, a, b, s0, s1, p0, p1, d0, d1): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_conv_2d_sk_p0(ctx, a, b): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_conv_2d_s1_ph(ctx, a, b): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ctypes.c_int],
    ggml_tensor_p
)
def ggml_conv_transpose_2d_p0(ctx, a, b, stride): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ggml_tensor_p
)
def ggml_pool_1d(ctx, a, op, k0, s0, p0): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float],
    ggml_tensor_p
)
def ggml_pool_2d(ctx, a, op, k0, k1, s0, s1, p0, p1): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int],
    ggml_tensor_p
)
def ggml_upscale(ctx, a, scale_factor): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ggml_tensor_p
)
def ggml_upscale_ext(ctx, a, ne0, ne1, ne2, ne3): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ggml_tensor_p
)
def ggml_pad(ctx, a, p0, p1, p2, p3): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int, ctypes.c_int],
    ggml_tensor_p
)
def ggml_timestep_embedding(ctx, timesteps, dim, max_period): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int],
    ggml_tensor_p
)
def ggml_argsort(ctx, a, order): ...

@ctypes_function(_ggml,
    [ggml_context_p, ctypes.c_float, ctypes.c_float, ctypes.c_float],
    ggml_tensor_p
)
def ggml_arange(ctx, start, stop, step): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int],
    ggml_tensor_p
)
def ggml_top_k(ctx, a, k): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ggml_tensor_p, ggml_tensor_p, ctypes.c_float, ctypes.c_float],
    ggml_tensor_p
)
def ggml_flash_attn_ext(ctx, q, k, v, mask, scale, max_bias): ...

@ctypes_function(_ggml,
    [ggml_tensor_p, ctypes.c_int],
    None
)
def ggml_flash_attn_ext_set_prec(a, prec): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ggml_tensor_p, ggml_tensor_p, ctypes.c_bool],
    ggml_tensor_p
)
def ggml_flash_attn_back(ctx, q, k, v, d, masked): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_ssm_conv(ctx, s, x, c, sq): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ggml_tensor_p, ggml_tensor_p, ggml_tensor_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_ssm_scan(ctx, s, x, dt, A, B, C, sq): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int],
    ggml_tensor_p
)
def ggml_win_part(ctx, a, w): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ggml_tensor_p
)
def ggml_win_unpart(ctx, a, w0, h0, w): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int],
    ggml_tensor_p
)
def ggml_unary(ctx, a, op): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int],
    ggml_tensor_p
)
def ggml_unary_inplace(ctx, a, op): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ctypes.c_int, ctypes.c_int],
    ggml_tensor_p
)
def ggml_get_rel_pos(ctx, a, qh, kh): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_add_rel_pos(ctx, a, pw, ph): ...

@ctypes_function(_ggml,
    [ggml_context_p, ggml_tensor_p, ggml_tensor_p, ggml_tensor_p],
    ggml_tensor_p
)
def ggml_add_rel_pos_inplace(ctx, a, pw, ph): ...
