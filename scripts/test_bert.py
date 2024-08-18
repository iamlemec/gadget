# test bert model

import numpy as np
from gadget.embed import test_embed

# configure
gg_path = '/home/doug/fast/embed'
hf_model = 'TaylorAI/bge-micro-v2'

# run tests
print('bge-micro:f32')
test_embed(f'{gg_path}/bge-micro-v2-f32.gguf', hf_model)
print('bge-micro:q8_0')
test_embed(f'{gg_path}/bge-micro-v2-q8_0.gguf', hf_model)
print('bge-micro:q4_k_m')
test_embed(f'{gg_path}/bge-micro-v2-q4_k_m.gguf', hf_model)
