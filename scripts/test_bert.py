# test bert model

import numpy as np
from gadget.bert import BertModel

# configure
gguf_path = '/home/doug/fast/embed/bge-micro-v2-f32.gguf'
batch_size = 512

# load model
bert = BertModel.from_path(gguf_path, batch_size=batch_size)
print(bert)

# make some tokens
tokens = np.arange(batch_size, dtype=np.int32)

# embed tokens
embed = bert.embed(tokens)
