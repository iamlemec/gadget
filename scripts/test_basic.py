# low level tests

import gc
import gadget.testing

# test low level interface
results = gadget.testing.test_backend()

# free memory
print(gc.collect())

# print results
graph = results[0]
print(graph.contents.n_nodes)
