# test llama model

import numpy as np
from gadget.textgen import test_textgen, test_huggingface

# configure
gg_path = '/home/doug/fast/models/meta-llama-3.2-1b-f32.gguf'
hf_model = 'meta-llama/Llama-3.2-1B'

# run tests
# print('CPU:F32:meta-llama/Meta-Llama-3-8B')
# test_textgen(f'{gg_path}/meta-llama-3-8b-f32.gguf', hf_model)

if __name__ == '__main__':
    gg_output = test_logits(gg_path, hf_model, batch_size=32, context_length=1024)
    print(gg_output)
    hf_output = test_logits_hf(hf_model)
    print(hf_output)
