# test bert model

import numpy as np
import gadget.bert as bert

# configure
gguf_path = '/home/doug/fast/embed/bge-micro-v2-f32.gguf'
hf_model = 'TaylorAI/bge-micro-v2'

# run test
results = bert.test_bert(gguf_path, hf_model, batch_size=512)

