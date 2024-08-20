# test bert model

import numpy as np
from gadget.embed import test_embed

# configure
gg_path = '/home/doug/fast/embed'
hf_model = 'TaylorAI/bge-micro-v2'

# run tests
print('CPU:F32:bge-micro')
test_embed(f'{gg_path}/bge-micro-v2-f32.gguf', hf_model)

print('CPU:Q8_0:bge-micro')
test_embed(f'{gg_path}/bge-micro-v2-q8_0.gguf', hf_model)

print('CPU:Q4_K_M:bge-micro')
test_embed(f'{gg_path}/bge-micro-v2-q4_k_m.gguf', hf_model)

print('CUDA:F32:bge-micro')
test_embed(f'{gg_path}/bge-micro-v2-f32.gguf', hf_model, backend='cuda')

print('CUDA:Q8_0:bge-micro')
test_embed(f'{gg_path}/bge-micro-v2-q8_0.gguf', hf_model, backend='cuda')

print('CUDA:Q4_K_M:bge-micro')
test_embed(f'{gg_path}/bge-micro-v2-q4_k_m.gguf', hf_model, backend='cuda')
