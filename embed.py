# embedding interface

import numpy as np

import api

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
        self.batch.n_tokens += len(tokens)

class LlamaModel:
    def __init__(
        self, path_model, batch_size=512, n_gpu_layers=0, causal_attn=None,
        pooling_type=api.LLAMA_POOLING_TYPE_UNSPECIFIED
    ):
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

    def __del__(self):
        api.llama_free(self.context)
        api.llama_free_model(self.model)
        api.llama_backend_free()

    def tokenize(self, text, max_tokens=None):
        if max_tokens is None:
            max_tokens = self.context_params.n_batch
        return api.llama_tokenize(self.model, text, max_tokens)

    def encode_batch(self, tokens, normalize=True):
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
        batch = LlamaBatch(self.context_params.n_batch, n_seq_max=n_seqs)
        for i, ts in enumerate(tokens):
            batch.add_sequence(ts, seq_id=i)
        api.llama_decode(self.context, batch.batch)

        # get embedding stats
        n_embd = api.llama_n_embd(self.model)
        pooling_type = api.llama_pooling_type(self.context)

        # handle un-pooled case separately
        if pooling_type == api.LLAMA_POOLING_TYPE_NONE:
            n_tokens_all = sum(n_tokens)
            embeds = api.llama_get_embeddings(self.context, n_tokens_all, n_embd)
            # TODO: should break this up into list of ndarrays
            return embeds.copy()

        # retrieve embeddings for each sequence
        embeds = np.zeros((n_seqs, n_embd), dtype=np.float32)
        for i in range(n_seqs):
            embeds[i,:] = api.llama_get_embeddings_seq(self.context, i, n_embd)

        # normalize embeddings
        if normalize:
            embeds /= np.linalg.norm(embeds, axis=1, keepdims=True)

        # return embeddings
        return embeds
    
    def embed(self, text, normalize=True):
        if isinstance(text, str):
            text = [text]
        tokens = [self.tokenize(s) for s in text]
        return self.encode_batch(tokens, normalize=normalize)
