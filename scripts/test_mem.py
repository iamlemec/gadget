# test memory usage

import tracemalloc
import gadget

tracemalloc.start()

print(tracemalloc.get_traced_memory())

gg = gadget.compute.test_torch()

print(tracemalloc.get_traced_memory())

del gg

print(tracemalloc.get_traced_memory())
