# test compute interface

import gadget
from gadget.ggml import GGMLQuantizationType as T

print('F32')
match = gadget.compute.test_compute(qtype=T.F32)
print('F16')
match = gadget.compute.test_compute(qtype=T.F16)
print('Q8_0')
match = gadget.compute.test_compute(qtype=T.Q8_0)
print('Q4_0')
match = gadget.compute.test_compute(qtype=T.Q4_0)
