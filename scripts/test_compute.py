# test compute interface

import gadget
from gadget.ggml import GGMLQuantizationType as T

print('CPU:F32:numpy')
match = gadget.compute.test_numpy(qtype=T.F32)

print('CPU:F32:torch')
match = gadget.compute.test_torch(qtype=T.F32)

print('CPU:F16:numpy')
match = gadget.compute.test_numpy(qtype=T.F16)

print('CPU:F16:torch')
match = gadget.compute.test_torch(qtype=T.F16)

print('CPU:Q8_0:numpy')
match = gadget.compute.test_numpy(qtype=T.Q8_0)

print('CPU:Q8_0:torch')
match = gadget.compute.test_torch(qtype=T.Q8_0)

print('CPU:Q4_0:numpy')
match = gadget.compute.test_numpy(qtype=T.Q4_0)

print('CPU:Q4_0:torch')
match = gadget.compute.test_torch(qtype=T.Q4_0)

print('CUDA:F32:torch')
match = gadget.compute.test_torch(qtype=T.F32, backend='cuda')

print('CUDA:F16:torch')
match = gadget.compute.test_torch(qtype=T.F16, backend='cuda')
