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

# Usage

*Note on GGML shape convention*: for `ggml_mul_mat` and others, the shape of inputs `x` and `y` and output `z` should be
```
x ~ (n, k, a, b)
y ~ (m, k, a, b)
z ~ (m, n, a, b)
```
In other words, the matmul is performed over the first two dimensions according to `z.T = x @ y.T` and is batched over the remaining dimensions.
