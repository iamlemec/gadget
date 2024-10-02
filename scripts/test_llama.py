# test llama model

import numpy as np
from gadget.textgen import test_textgen

# configure
gg_path = '/home/doug/fast/models'
hf_model = 'meta-llama/Meta-Llama-3-8B'

# run tests
print('CPU:F32:meta-llama/Meta-Llama-3-8B')
test_textgen(f'{gg_path}/meta-llama-3-8b-f32.gguf', hf_model)
