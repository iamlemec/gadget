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

# Conventions

*Matrix shape and order*: For contiguous tensors, the stride of the first dimension is one?

*Note on GGML shape convention*: for `ggml_mul_mat` and others, the shape of inputs `a` and `b` and output `c` should be
```
a ~ (k, n, i, j)
b ~ (k, m, i, j)
c ~ (n, m, i, j)
```
In other words, the matmul is performed over the first two dimensions according to `c.T = b.T @ a` (or `c = a.T @ b`) and is batched over the remaining dimensions.
