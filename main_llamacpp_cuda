import os
import time
from typing import Iterator
import gc

import numpy as np
from llama_cpp import Llama


class MetricsCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_ms": 0,
            "predicted_ms": 0,
            "predicted_per_second": 0,
            "tpot_ms": 0,
        }


def generate_text_stream(
    llm: Llama, prompt: str, metrics_collector: MetricsCollector, max_tokens: int = 512
) -> Iterator[str]:
    completion_tokens = 0
    start_time = time.perf_counter()
    first_token_generated = False

    try:
        completion = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            stream=True,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            stop=["</s>"],
        )

        for chunk in completion:
            if "content" in chunk["choices"][0]["delta"]:
                if not first_token_generated:
                    first_token_generated = True
                    prompt_end = time.perf_counter()
                    metrics_collector.metrics["prompt_ms"] = (prompt_end - start_time) * 1000
                    metrics_collector.metrics["prompt_tokens"] = llm.n_tokens

                content = chunk["choices"][0]["delta"]["content"]
                completion_tokens += 1
                yield content

        generation_end = time.perf_counter()
        generation_time = generation_end - (prompt_end if first_token_generated else start_time)

        # Update metrics
        metrics_collector.metrics["predicted_ms"] = generation_time * 1000
        metrics_collector.metrics["completion_tokens"] = completion_tokens
        metrics_collector.metrics["total_tokens"] = (
            metrics_collector.metrics["prompt_tokens"] + completion_tokens
        )
        metrics_collector.metrics["predicted_per_second"] = (
            completion_tokens / generation_time if generation_time > 0 else 0
        )
        # calculate time per output token
        metrics_collector.metrics["tpot_ms"] = (
            generation_time * 1000 / completion_tokens if completion_tokens > 0 else 0
        )

    except Exception as e:
        raise Exception(f"Error during generation: {str(e)}")


def run_inference_multiple_times(prompts: list[str]):
    metrics_collector = MetricsCollector()
    all_metrics = []

    print(f"\nRunning inference for {len(prompts)} prompts...")

    model_path = os.getenv(
        "MODEL_PATH",
        "models_gguf/Llama-3.1-8B-Instruct-GGUF/llama-3.1-8b-instruct.gguf",
    )

    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_batch=512,
        n_threads=18,
        n_gpu_layers=-1,
        seed=42,
        cuda_backend="cublas",
        main_gpu=0,
        cache_prompt=False,
        flash_attn=True,
    )

    try:
        for i, prompt in enumerate(prompts):
            print(f"\nPrompt {i+1}/{len(prompts)}")
            print(f"Prompt: {prompt[:100]}...")

            metrics_collector.reset()

            try:
                for text_chunk in generate_text_stream(llm, prompt, metrics_collector):
                    print(text_chunk, end="", flush=True)

                all_metrics.append(metrics_collector.metrics.copy())

            except Exception as e:
                print(f"Error in prompt {i+1}: {e}")
                continue

            llm.reset()
            gc.collect()  # Force garbage collection between runs
            time.sleep(1)  # Cool-down period between runs

    finally:
        del llm
        gc.collect()

    print("\n" + "=" * 50)
    print("Final Statistics (averaged across all prompts):")
    print("=" * 50)

    # Token Usage (from last run)
    last_metrics = all_metrics[-1]
    print("\nToken Usage (last prompt):")
    print(f"  Prompt tokens:     {last_metrics['prompt_tokens']}")
    print(f"  Completion tokens: {last_metrics['completion_tokens']}")
    print(f"  Total tokens:      {last_metrics['total_tokens']}")

    # Time to first token
    prompt_times = [m["prompt_ms"] / 1000 for m in all_metrics]  # Convert to seconds
    print("\nTime to first token:")
    print(f"  {np.mean(prompt_times):.3f} ± {np.std(prompt_times):.3f} s")
    print(f"  Min:  {np.min(prompt_times):.3f} s")
    print(f"  Max:  {np.max(prompt_times):.3f} s")

    # Generation Time
    generation_times = [m["predicted_ms"] / 1000 for m in all_metrics]  # Convert to seconds
    print("\nGeneration Time:")
    print(f"  {np.mean(generation_times):.3f} ± {np.std(generation_times):.3f} s")
    print(f"  Min:  {np.min(generation_times):.3f} s")
    print(f"  Max:  {np.max(generation_times):.3f} s")

    # Tokens per Second
    tokens_per_second = [m["predicted_per_second"] for m in all_metrics]
    print("\nTokens per Second:")
    print(f"  {np.mean(tokens_per_second):.2f} ± {np.std(tokens_per_second):.2f}")
    print(f"  Min:  {np.min(tokens_per_second):.2f}")
    print(f"  Max:  {np.max(tokens_per_second):.2f}")

    # Time per output token
    tpot = [m["tpot_ms"] for m in all_metrics]
    print("\nTime per output token:")
    print(f"  {np.mean(tpot):.3f} ± {np.std(tpot):.3f} ms")
    print(f"  Min:  {np.min(tpot):.3f} ms")
    print(f"  Max:  {np.max(tpot):.3f} ms")


if __name__ == "__main__":
    prompt = """
    Write a short poem about a cat.
    """

    prompts = [prompt] * 3

    run_inference_multiple_times(prompts)
