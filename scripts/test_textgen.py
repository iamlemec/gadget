# test llama textgen

from gadget.textgen import test_textgen

# configure
gg_path = '/home/doug/fast/models/meta-llama-3.2-1b-f32.gguf'
hf_model = 'meta-llama/Llama-3.2-1B'
prompt = 'The capital of France is'

if __name__ == '__main__':
    test_textgen(gg_path, hf_model, prompt=prompt, context_length=1024, max_gen=512, backend='cuda')
