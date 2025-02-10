import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer, Timer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler
from util import format_prompt

model_dir = "/lunit/home/pytholic/models_exl2/phi-3.5-mini-instruct-q8_0"
config = ExLlamaV2Config(model_dir)
config.no_flash_attn = False
config.no_flash_attn_cross = False
config.no_xformers = False
config.arch_compat_overrides()
model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, lazy=True)
model.load_autosplit(cache, progress=True)

print("Loading tokenizer...")
tokenizer = ExLlamaV2Tokenizer(config)

# Initialize the generator with all default parameters

generator = ExLlamaV2DynamicGenerator(
    model=model,
    cache=cache,
    tokenizer=tokenizer,
)

settings = ExLlamaV2Sampler.Settings()
settings.temperature = 0.0  # Set temperature (0.0 for deterministic)
settings.top_k = 1  # Set top-k
settings.top_p = 1.0  # Set top-p

max_new_tokens = 512

# Warmup generator. The function runs a small completion job to allow all the kernels to fully initialize and
# autotune before we do any timing measurements. It can be a little slow for larger models and is not needed
# to produce correct output.

generator.warmup()

# Generate one completion, using default settings
prompt_format = "llama3"  # or "llama3", "mistral", etc.
system_prompt = "You are a helpful AI assistant."
prompt = """
    Here's why you should adopt a cat:
    """
formatted_prompt = format_prompt(prompt_format, system_prompt, prompt)

with Timer() as t_single:
    output = generator.generate(
        prompt=formatted_prompt,
        max_new_tokens=max_new_tokens,
        add_bos=True,
        min_new_tokens=max_new_tokens,
    )

print("-----------------------------------------------------------------------------------")
print("- Single completion")
print("-----------------------------------------------------------------------------------")
print(output)
print()

# Do a batched generation

prompts = [
    format_prompt(prompt_format, system_prompt, "Once upon a time,"),
    format_prompt(prompt_format, system_prompt, "The secret to success is"),
    format_prompt(prompt_format, system_prompt, "There's no such thing as"),
    format_prompt(prompt_format, system_prompt, "Here's why you should adopt a cat:"),
]

with Timer() as t_batched:
    outputs = generator.generate(prompt = prompts, max_new_tokens = max_new_tokens, add_bos = True)

for idx, output in enumerate(outputs):
    print("-----------------------------------------------------------------------------------")
    print(f"- Batched completion #{idx + 1}")
    print("-----------------------------------------------------------------------------------")
    print(output)
    print()

# print("-----------------------------------------------------------------------------------")
print(f"speed, bsz 1: {max_new_tokens / t_single.interval:.2f} tokens/second")
print(f"speed, bsz {len(prompts)}: {max_new_tokens * len(prompts) / t_batched.interval:.2f} tokens/second")
