### general stuff

import os
import ctypes

##
## missing utils
##

class DummyFunction:
    def __init__(self, message):
        self.message = message

    def __call__(self, *args, **kwargs):
        raise Exception(self.message)

##
## ctypes helpers
##

# load a shared lib with env override
def load_shared_lib(lib_name, env_var=None):
    # get shared library path
    if env_var is not None and env_var in os.environ:
        lib_path = os.environ[env_var]
    else:
        module_path = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(module_path, lib_name)

    # load shared library
    try:
        lib_obj = ctypes.CDLL(lib_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load shared library '{lib_name}': {e}")

    # return library object
    return lib_obj

# decorate a function with ctypes information
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
## libc
##

_libc = ctypes.CDLL(None)

@ctypes_function(_libc,
    [ctypes.c_size_t],
    ctypes.c_void_p
)
def malloc(size): ...

@ctypes_function(_libc,
    [ctypes.c_void_p]
)
def free(ptr): ...
