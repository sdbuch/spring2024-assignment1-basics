import cProfile
import io
import pstats
import time
from pathlib import Path

from line_profiler import LineProfiler

from tokenizer import BPENaive


def profile_tokenizer(your_tokenizer, sample_text, num_runs=100):
    """
    Comprehensive profiling of a tokenizer using both cProfile and line_profiler.
    Added support for deeper call stack analysis.
    """
    print("Running detailed cProfile analysis...")
    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(num_runs):
        your_tokenizer.encode(sample_text)

    profiler.disable()

    # Create string buffer for output
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)

    # Method 1: Print detailed recursive call paths
    print("\nDetailed call paths (including subcalls):")
    stats.sort_stats("cumulative").print_callees()

    # # Method 2: Print full call stack with custom depth
    # print("\nFull call stack analysis:")
    # stats.sort_stats("cumulative").print_stats(
    #     ".*", maxdepth=10
    # )  # Increase maxdepth for deeper analysis

    # Method 3: Print specific function and its callees
    print("\nDetailed analysis of specific function and its subcalls:")
    stats.print_callees("BPENaive._merge")  # Replace with your specific method name

    # You can also see who called your function
    print("\nCalls to your function came from:")
    stats.print_callers("BPENaive._merge")  # Replace with your specific method name

    # Original line-by-line profiling
    print("\nRunning line-by-line profiling...")
    lp = LineProfiler()
    lp.add_function(your_tokenizer.encode)

    # Add specific methods you want to profile
    if hasattr(your_tokenizer, "_merge"):
        lp.add_function(your_tokenizer._merge)

    wrapped_encode = lp(your_tokenizer.encode)
    wrapped_encode(sample_text)
    lp.print_stats()


if __name__ == "__main__":
    sample_text = (
        """
    This is a sample text that will be used for profiling.
    Make sure it's long enough to get meaningful results.
    """
        * 100
    )

    corpus_path = Path("./test_data/test.txt")
    vocab_size = 512  # 'initial' size is 256 (bytes)
    my_tokenizer = BPENaive(corpus_path, vocab_size)
    profile_tokenizer(my_tokenizer, sample_text)
