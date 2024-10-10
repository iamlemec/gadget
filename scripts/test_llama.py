# test llama model

import numpy as np
from gadget.textgen import test_logits, test_logits_hf

# configure
gg_path = '/home/doug/fast/models/meta-llama-3.2-1b-f32.gguf'
hf_model = 'meta-llama/Llama-3.2-1B'

if __name__ == '__main__':
    gg_output = test_logits(gg_path, hf_model, batch_size=32, context_length=1024)
    print(gg_output)
    hf_output = test_logits_hf(hf_model)
    print(hf_output)
