import random
import string
import time
from pathlib import Path
from statistics import mean, stdev

import tiktoken

from tokenizer import BPENaive, BPEImproved


def generate_random_text(length):
    """Generate random text of specified length."""
    return "".join(
        random.choices(
            string.ascii_letters + string.digits + string.punctuation + " ", k=length
        )
    )


def benchmark_tokenizers(your_tokenizer, text_samples, num_runs=5):
    """
    Benchmark tokenizers against each other.

    Args:
        your_tokenizer: Your BPE tokenizer instance
        text_samples: List of strings to tokenize
        num_runs: Number of times to run each benchmark
    """
    # Initialize tiktoken
    enc = tiktoken.get_encoding("cl100k_base")  # Using OpenAI's common encoding

    results = {
        "your_tokenizer": {"times": [], "tokens": []},
        "tiktoken": {"times": [], "tokens": []},
    }

    # Run benchmarks multiple times
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")

        # Benchmark your tokenizer
        start_time = time.perf_counter()
        your_tokens = [your_tokenizer.encode(text) for text in text_samples]
        your_time = time.perf_counter() - start_time

        # Benchmark tiktoken
        start_time = time.perf_counter()
        tiktoken_tokens = [enc.encode(text) for text in text_samples]
        tiktoken_time = time.perf_counter() - start_time

        # Store results
        results["your_tokenizer"]["times"].append(your_time)
        results["tiktoken"]["times"].append(tiktoken_time)
        results["your_tokenizer"]["tokens"].append(sum(len(t) for t in your_tokens))
        results["tiktoken"]["tokens"].append(sum(len(t) for t in tiktoken_tokens))

    # Calculate and print statistics
    print("\nBenchmark Results:")
    print("-" * 50)

    for tokenizer_name, data in results.items():
        times = data["times"]
        tokens = data["tokens"]

        print(f"\n{tokenizer_name}:")
        print(f"Average time: {mean(times):.4f} seconds")
        print(f"Standard deviation: {stdev(times):.4f} seconds")
        print(f"Average tokens per sample: {mean(tokens) / len(text_samples):.2f}")
        print(f"Tokens/second: {mean(tokens) / mean(times):,.2f}")


if __name__ == "__main__":
    # Generate test data of varying lengths
    test_samples = [
        generate_random_text(100),  # Short text
        generate_random_text(1000),  # Medium text
        generate_random_text(10000),  # Long text
    ]

    # Replace MyBPETokenizer with your actual tokenizer class
    class MyBPETokenizer:
        def encode(self, text):
            # Replace this with your tokenizer's encode method
            pass

    corpus_path = Path("./test_data/test.txt")
    vocab_size = 512  # 'initial' size is 256 (bytes)
    my_tokenizer = BPENaive(corpus_path, vocab_size)
    benchmark_tokenizers(my_tokenizer, test_samples)
