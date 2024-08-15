# test bert model

import numpy as np
import gadget.bert as bert

# configure
gguf_path = '/home/doug/fast/embed/bge-micro-v2-f32.gguf'
model, embed = bert.test_bert(gguf_path)

# print results
print(embed)
