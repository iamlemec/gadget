# embedding interface

import torch

from . import api
from .api import (
    LLAMA_POOLING_TYPE_UNSPECIFIED,
    LLAMA_POOLING_TYPE_NONE,
    LLAMA_POOLING_TYPE_MEAN,
    LLAMA_POOLING_TYPE_CLS,
    LLAMA_POOLING_TYPE_LAST,
)
from .utils import pack_batches

class LlamaBatch:
    def __init__(self, n_tokens, n_seq_max=1):
        self.batch = api.llama_batch_init(n_tokens, embd=0, n_seq_max=n_seq_max)

    def __del__(self):
        api.llama_batch_free(self.batch)

    def clear(self):
        self.batch.n_tokens = 0

    def add_sequence(self, tokens, seq_id=0):
        for i, k in enumerate(tokens, start=self.batch.n_tokens):
            self.batch.pos[i] = i
            self.batch.token[i] = k
            self.batch.n_seq_id[i] = 1
            self.batch.seq_id[i][0] = seq_id
            self.batch.logits[i] = 1
        self.batch.n_tokens += len(tokens)

    def add_parallel(self, tokens, positions=None, sequences=None):
        n_tokens = len(tokens)
        if positions is None:
            positions = self.batch.n_tokens
        if sequences is None:
            sequences = range(n_tokens)
        if isinstance(positions, int):
            positions = [positions] * n_tokens
        for i, (k, p, s) in enumerate(zip(tokens, positions, sequences)):
            self.batch.pos[i] = p
            self.batch.token[i] = k
            self.batch.n_seq_id[i] = 1
            self.batch.seq_id[i][0] = s
            self.batch.logits[i] = 1
        self.batch.n_tokens += len(tokens)

class LlamaModel:
    def __init__(
        self, path_model, batch_size=512, n_gpu_layers=0, causal_attn=None,
        device='cpu', dtype=torch.float32, pooling_type=LLAMA_POOLING_TYPE_UNSPECIFIED
    ):
        # output params
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size

        # initialize llama backend
        api.llama_backend_init()
        api.llama_numa_init()

        # load model from file
        self.model_params = api.llama_model_default_params()
        self.model_params.n_gpu_layers = n_gpu_layers
        self.model = api.llama_load_model_from_file(path_model, self.model_params)

        # initialize context
        self.context_params = api.llama_context_default_params()
        self.context_params.embeddings = True
        self.context_params.n_ctx = batch_size
        self.context_params.n_batch = batch_size
        self.context_params.n_ubatch = batch_size
        self.context_params.pooling_type = pooling_type
        self.context = api.llama_new_context_with_model(self.model, self.context_params)

        # set causal attention
        if causal_attn is not None:
            api.llama_set_causal_attn(self.context, causal_attn)

        # static model features
        self.embed_size = api.llama_n_embd(self.model)

        # create batch for encoding
        self.batch = LlamaBatch(batch_size)

    def __del__(self):
        api.llama_free(self.context)
        api.llama_free_model(self.model)
        api.llama_backend_free()

    def tokenize(self, text, max_tokens=None):
        if max_tokens is None:
            max_tokens = self.context_params.n_batch
        return api.llama_tokenize(self.model, text, max_tokens)

    def encode_batch(self, tokens):
        # check for empty input
        if len(tokens) == 0:
            raise ValueError('No token sequence provided')

        # get token count statistics
        n_seqs = len(tokens)
        n_tokens = [len(ts) for ts in tokens]
        n_tokens_min = min(n_tokens)
        n_tokens_max = max(n_tokens)

        # validate input for all sequences
        if n_tokens_min == 0:
            raise ValueError('Token sequence is empty')
        if n_tokens_max > self.context_params.n_batch:
            raise ValueError(
                f'Token count exceeds batch size: {len(tokens)} > {self.context_params.n_batch}'
            )

        # run decode on batch
        self.batch.clear()
        for i, ts in enumerate(tokens):
            self.batch.add_sequence(ts, seq_id=i)
        api.llama_decode(self.context, self.batch.batch)

        # get embedding stats
        pooling_type = api.llama_pooling_type(self.context)

        # handle un-pooled case separately
        if pooling_type == LLAMA_POOLING_TYPE_NONE:
            n_tokens_all = sum(n_tokens)
            data = api.llama_get_embeddings(self.context, n_tokens_all, self.embed_size)
            embeds = torch.from_numpy(data).to(dtype=self.dtype, copy=True)
            return embeds.split(n_tokens, dim=0)

        # retrieve embeddings for each sequence
        embeds = torch.empty((n_seqs, self.embed_size), device=self.device, dtype=self.dtype)
        for i in range(n_seqs):
            embeds[i,:] = torch.from_numpy(
                api.llama_get_embeddings_seq(self.context, i, self.embed_size)
            )

        # return embeddings
        return embeds

    def embed(self, text, normalize=True):
        # handle single case
        if isinstance(text, str):
            text = [text]

        # get embedding stats
        pooling_type = api.llama_pooling_type(self.context)

        # tokenize text
        n_seqs = len(text)
        tokens = [self.tokenize(s) for s in text]

        # plan batch contents
        sizes = [len(toks) for toks in tokens]
        batches = pack_batches(sizes, self.batch_size)

        # handle un-pooled case separately
        if pooling_type == LLAMA_POOLING_TYPE_NONE:
            embeds = n_seqs * [None]
            for idxs in batches:
                data = self.encode_batch([tokens[i] for i in idxs])
                for i, idx in enumerate(idxs):
                    embeds[idx] = data[i]
            if normalize:
                for i, e in enumerate(embeds):
                    embeds[i] /= torch.norm(e, dim=1, keepdim=True)
            return embeds

        # compute embeddings
        embeds = torch.empty((n_seqs, self.embed_size), device=self.device, dtype=self.dtype)
        for idxs in batches:
            embeds[idxs] = self.encode_batch([tokens[i] for i in idxs])

        # normalize embeddings
        if normalize:
            embeds /= torch.norm(embeds, dim=1, keepdim=True)

        return embeds

    def __call__(self, *args, **kwargs):
        return self.embed(*args, **kwargs)
