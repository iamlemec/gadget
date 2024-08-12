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

To build the shared libraries for local testing, you can use `cmake` directly
```bash
cmake -DCMAKE_BUILD_TYPE=Debug -B build .
cmake --build build -j
```

# Conventions

*Matrix shape and order*: tensors are row-major, meaning elements in a row are stored in contiguous order. However, the way in which dimensions are reported is reversed from `numpy`. The first number in the shape is the number of columns, the second is the number of rows, and so on. The logic here is that the first number denotes the number of elements in a row, the second denotes the number of columns and so on.

*Matrix multiplication*: for `ggml_mul_mat` and others, the *GGML*-style shape of inputs `a` and `b` and output `c` should be
```
a ~ (k, n, i, j)
b ~ (k, m, i, j)
c ~ (n, m, i, j)
```
In other words, the matmul is performed over the first two dimensions according to `c.T = b.T @ a` (or `c = a.T @ b`) and is batched over the remaining dimensions.

If we think about shapes `numpy`-style, then this would read
```
a ~ (n, k, i, j)
b ~ (m, k, i, j)
c ~ (m, n, i, j)
```
And we would write the matrix multiplication as `c = b @ a.T`. So basically the same thing, but everything is transposed to undo the shape reversed notation.
