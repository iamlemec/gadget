// llm inspector with llama.cpp

#include "llama.h"
#include "common.h"

static std::vector<std::string> split_lines(const std::string & s) {
    std::string line;
    std::vector<std::string> lines;
    std::stringstream ss(s);
    while (std::getline(ss, line)) {
        lines.push_back(line);
    }
    return lines;
}

static void batch_add_seq(llama_batch & batch, const std::vector<int32_t> & tokens, llama_seq_id seq_id) {
    size_t n_tokens = tokens.size();
    for (size_t i = 0; i < n_tokens; i++) {
        llama_batch_add(batch, tokens[i], i, { seq_id }, true);
    }
}

static void batch_decode(llama_context * ctx, llama_batch & batch, float * output, int n_seq, int n_embd) {
    // clear previous kv_cache values (irrelevant for embeddings)
    llama_kv_cache_clear(ctx);

    // run model
    fprintf(stderr, "%s: n_tokens = %d, n_seq = %d\n", __func__, batch.n_tokens, n_seq);
    if (llama_decode(ctx, batch) < 0) {
        fprintf(stderr, "%s : failed to decode\n", __func__);
    }

    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        // try to get sequence embeddings - supported only when pooling_type is not NONE
        const float * embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
        GGML_ASSERT(embd != NULL && "failed to get sequence embeddings");

        float * out = output + batch.seq_id[i][0] * n_embd;
        //TODO: I would also add a parameter here to enable normalization or not.
        /*fprintf(stdout, "unnormalized_embedding:");
        for (int hh = 0; hh < n_embd; hh++) {
            fprintf(stdout, "%9.6f ", embd[hh]);
        }
        fprintf(stdout, "\n");*/
        llama_embd_normalize(embd, out, n_embd);
    }
}

int main(int argc, char ** argv) {
    // custom argument defaults
    bool causal_attn = false;

    // split off custom arguments
    int argc_base = 0;
    std::vector<char *> argv_base;

    // add program name
    argc_base++;
    argv_base.push_back(argv[0]);

    // parse custom arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--attention") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "%s: error: missing argument for --attn-type\n", __func__);
                return -1;
            }
            std::string value(argv[i]);
            if (value == "causal") {
                causal_attn = true;
            } else if (value == "non-causal") {
                causal_attn = false;
            } else {
                fprintf(stderr, "%s: error: invalid argument for --attn-type\n", __func__);
                return -1;
            }
        } else {
            fprintf(stderr, "%s: adding argument %s\n", __func__, argv[i]);
            argc_base++;
            argv_base.push_back(argv[i]);
        }
    }

    // parse command line arguments
    gpt_params params;
    if (!gpt_params_parse(argc_base, argv_base.data(), params)) {
        gpt_params_print_usage(argc_base, argv_base.data(), params);
        return 1;
    }

    // set to embedding mode
    params.embedding = true;
    params.n_ubatch = params.n_batch;

    // set up random seed
    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }
    std::mt19937 rng(params.seed);

    // initialize backends
    llama_backend_init();
    llama_numa_init(params.numa);

    // load the model and context
    llama_model * model;
    llama_context * ctx;
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    // set attention type
    llama_set_causal_attn(ctx, causal_attn);

    // get pooling type used
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

    // split the prompt into lines
    std::vector<std::string> prompts = split_lines(params.prompt);

    // max batch size
    const uint64_t n_batch = params.n_batch;
    GGML_ASSERT(params.n_batch >= params.n_ctx);

    // tokenize the prompts and check length
    std::vector<std::vector<int32_t>> inputs;
    for (const std::string & prompt : prompts) {
        std::vector<int32_t> inp = ::llama_tokenize(ctx, prompt, true, false);
        if (inp.size() > n_batch) {
            fprintf(stderr, "%s: error: number of tokens in input line (%lld) exceeds batch size (%lld), increase batch size and re-run\n",
                    __func__, (long long int) inp.size(), (long long int) n_batch);
            return 1;
        }
        inputs.push_back(inp);
    }

    // tokenization stats
    if (params.verbose_prompt) {
        for (int i = 0; i < (int) inputs.size(); i++) {
            fprintf(stderr, "%s: prompt %d: '%s'\n", __func__, i, prompts[i].c_str());
            fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, inputs[i].size());
            for (int j = 0; j < (int) inputs[i].size(); j++) {
                int32_t id = inputs[i][j];
                std::string token = llama_token_to_piece(ctx, id);
                fprintf(stderr, "\"%s\" %d ", token.c_str(), id);
            }
            fprintf(stderr, "\n\n");
        }
    }

    // initialize batch
    const int n_prompts = prompts.size();
    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);

    // allocate output
    const int n_embd = llama_n_embd(model);
    std::vector<float> embeddings(n_prompts * n_embd, 0);
    float * emb = embeddings.data();

    // break into batches
    int p = 0; // number of prompts processed already
    int s = 0; // number of prompts in current batch
    for (int k = 0; k < n_prompts; k++) {
        // clamp to n_batch tokens
        std::vector<int32_t> & inp = inputs[k];
        const uint64_t n_toks = inp.size();

        // encode if at capacity
        if (batch.n_tokens + n_toks > n_batch) {
            float * out = emb + p * n_embd;
            batch_decode(ctx, batch, out, s, n_embd);
            llama_batch_clear(batch);
            p += s;
            s = 0;
        }

        // add to batch
        batch_add_seq(batch, inp, s);
        s += 1;
    }

    // decode final batch
    if (s > 0) {
        float * out = emb + p * n_embd;
        batch_decode(ctx, batch, out, s, n_embd);
    }

    // print the first part of the embeddings or for a single prompt, the full embedding
    fprintf(stdout, "\n");
    for (int j = 0; j < n_prompts; j++) {
        fprintf(stdout, "embedding %d: ", j);
        for (int i = 0; i < std::min(16, n_embd); i++) {
            fprintf(stdout, "%9.6f ", emb[j * n_embd + i]);
        }
        fprintf(stdout, "\n");
    }

    // clean up
    llama_print_timings(ctx);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
