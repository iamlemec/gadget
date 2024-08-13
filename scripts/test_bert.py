# test bert model

import numpy as np
from gadget.bert import BertModel

# configure
gguf_path = '/home/doug/fast/embed/bge-micro-v2-f32.gguf'

# load model
bert = BertModel.from_path(gguf_path)
print(bert)

# make some tokens
tokens = np.arange(bert.batch_size, dtype=np.int32)

# embed tokens
embed = bert.embed(tokens)
