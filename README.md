# Gadget

Simple embedding interface for `llama.cpp`.

# Install

To install with `pip` run:

```bash
pip install -e . --user
```

You can pass arguments to `cmake` using the `CMAKE_ARGS` environment varibles. For example, to add CUDA support:

```bash
CMAKE_ARGS="-DGGML_CUDA=ON"
```
