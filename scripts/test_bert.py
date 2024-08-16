# test bert model

import numpy as np
from gadget.embed import test_bert

# configure
gguf_path = '/home/doug/fast/embed/bge-micro-v2-f32.gguf'
hf_model = 'TaylorAI/bge-micro-v2'

# run test
results = test_bert(gguf_path, hf_model, batch_size=512)

