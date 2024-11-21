import re
from dataclasses import dataclass
from typing import Iterator, List


@dataclass
class TokenizerMatch:
    """Represents a match in the text, either a regular token or special token"""

    text: str
    is_special: bool
    start: int
    end: int


def find_partial_special_token_match(text: str, special_tokens: List[str]) -> int:
    """
    Check if the end of text partially matches any special token.
    Returns the number of additional characters needed to complete the potential match.
    """
    if not special_tokens:
        return 0

    # Sort special tokens by length in descending order
    sorted_tokens = sorted(special_tokens, key=len, reverse=True)
    max_token_len = len(sorted_tokens[0])

    # Check for partial matches at the end of the text
    for i in range(min(max_token_len, len(text))):
        suffix = text[-(i + 1) :]
        for token in sorted_tokens:
            if token.startswith(suffix):
                # Return how many more characters we need
                return len(token) - len(suffix)
    return 0


def stream_tokenize(
    file_path: str, special_tokens: List[str] = None, chunk_size: int = 1024 * 1024
) -> Iterator[str]:
    """
    Stream-process a file using GPT-2 pretokenization and special token splitting.

    Args:
        file_path: Path to the file to process
        special_tokens: List of special tokens to split on (these will be removed from output)
        chunk_size: Size of chunks to read at once
    """
    special_tokens = special_tokens or []

    # Base pattern for GPT-2 tokenization
    base_pattern = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[\w]+| ?[\d]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

    # Create special token pattern
    if special_tokens:
        special_token_pattern = "|".join(
            map(re.escape, sorted(special_tokens, key=len, reverse=True))
        )

    buffer = ""

    with open(file_path, "r") as file:
        while True:
            # Read next chunk and combine with buffer
            chunk = file.read(chunk_size)
            if not chunk and not buffer:
                break

            text = buffer + chunk

            if chunk:  # Only if not at EOF
                # Handle potential contractions at the end
                if text.endswith("'"):
                    extra = file.read(2)
                    text += extra
                elif text.endswith("'", 0, -1):
                    extra = file.read(1)
                    text += extra

                # Handle potential special tokens at the end
                chars_needed = find_partial_special_token_match(text, special_tokens)
                while chars_needed:
                    extra = file.read(1)
                    text += extra
                    chars_needed = find_partial_special_token_match(
                        text, special_tokens
                    )

            # First split on special tokens if any exist
            if special_tokens:
                segments = re.split(f"({special_token_pattern})", text)
            else:
                segments = [text]

            # Process each segment
            current_pos = 0
            matches = []

            for i, segment in enumerate(segments):
                if not segment:  # Skip empty segments
                    continue

                if special_tokens and i % 2 == 1:  # This is a special token
                    # We skip special tokens as they should be removed
                    current_pos += len(segment)
                    continue

                # Find regular tokens in this segment
                for match in re.finditer(base_pattern, segment):
                    matches.append(
                        TokenizerMatch(
                            text=match.group(0),
                            is_special=False,
                            start=current_pos + match.start(),
                            end=current_pos + match.end(),
                        )
                    )
                current_pos += len(segment)

            if not matches:
                if chunk:
                    buffer = text
                    continue
                break

            if len(matches) == 1:
                if chunk:
                    buffer = text
                    continue
                yield matches[0].text
                break

            # Yield all matches except the last one
            for match in matches[:-1]:
                yield match.text

            # Buffer everything after the second-to-last match
            if len(matches) >= 2:
                last_complete_match = matches[-2]
                buffer = text[last_complete_match.end :]

            if not chunk:
                if len(matches) > 1:
                    yield matches[-1].text
                if buffer:
                    # Process any remaining buffer
                    if special_tokens:
                        segments = re.split(f"({special_token_pattern})", buffer)
                    else:
                        segments = [buffer]

                    for i, segment in enumerate(segments):
                        if not segment or (special_tokens and i % 2 == 1):
                            continue
                        yield from re.findall(base_pattern, segment)
                break


def test_special_tokenization(
    text: str, special_tokens: List[str], chunk_size: int = 20
) -> bool:
    """Test if streaming tokenization with special tokens gives identical results to non-streaming."""
    import tempfile

    # Write test text to temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp:
        temp.write(text)
        temp_path = temp.name

    try:
        # Get streaming results
        streaming_tokens = list(stream_tokenize(temp_path, special_tokens, chunk_size))

        # Get non-streaming results
        # First split on special tokens
        if special_tokens:
            pattern = "|".join(
                map(re.escape, sorted(special_tokens, key=len, reverse=True))
            )
            segments = re.split(pattern, text)
        else:
            segments = [text]

        # Then apply regular tokenization to non-special segments
        base_pattern = (
            r"""'(?:[sdmt]|ll|ve|re)| ?[\w]+| ?[\d]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
        )
        non_streaming_tokens = []
        for segment in segments:
            if segment:  # Skip empty segments
                non_streaming_tokens.extend(re.findall(base_pattern, segment))

        # Compare results
        if streaming_tokens != non_streaming_tokens:
            print("Mismatch found!")
            print("Streaming:", streaming_tokens)
            print("Non-streaming:", non_streaming_tokens)
            return False
        return True
    finally:
        import os

        os.unlink(temp_path)


def benchmark_tokenizers():
    """Benchmark chunked vs non-chunked tokenization with various parameters."""
    import random
    import string
    import tempfile
    import time

    def generate_text(size_kb: int, special_token_freq: float = 0.05) -> str:
        """Generate random text with occasional special tokens."""
        words = [
            "the",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
            "lazy",
            "dog",
            "they've",
            "don't",
            "it's",
            "I'll",
            "we're",
        ]
        special_tokens = ["<START>", "<MID>", "<END>"]

        text = []
        current_size = 0
        target_size = size_kb * 1024

        while current_size < target_size:
            if random.random() < special_token_freq:
                text.append(random.choice(special_tokens))
            else:
                text.append(random.choice(words))
            current_size += len(text[-1]) + 1  # +1 for space

        return " ".join(text)

    # Test parameters
    sizes_kb = [100, 1000, 10000, 100000]  # Different input sizes in KB
    chunk_sizes = [128, 512, 1024, 2048]
    special_tokens = ["<START>", "<MID>", "<END>"]
    iterations = 3  # Number of times to run each test

    print("\nTokenizer Benchmarking")
    print("=" * 80)
    print(f"Running {iterations} iterations for each configuration")
    print("-" * 80)
    print(f"{'Size (KB)':>10} {'Chunk Size':>12} {'Method':>15} {'Time (s)':>10}")
    print("-" * 80)

    for size_kb in sizes_kb:
        # Generate test text
        text = generate_text(size_kb)

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp:
            temp.write(text)
            temp_path = temp.name

        try:
            # Test non-streaming version
            times = []
            for _ in range(iterations):
                start = time.time()
                if special_tokens:
                    pattern = "|".join(
                        map(re.escape, sorted(special_tokens, key=len, reverse=True))
                    )
                    segments = re.split(pattern, text)
                else:
                    segments = [text]

                base_pattern = (
                    r"""'(?:[sdmt]|ll|ve|re)| ?[\w]+| ?[\d]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
                )
                tokens = []
                for segment in segments:
                    if segment:
                        tokens.extend(re.findall(base_pattern, segment))
                end = time.time()
                times.append(end - start)
            avg_time = sum(times) / len(times)
            print(f"{size_kb:>10} {'N/A':>12} {'non-streaming':>15} {avg_time:>10.4f}")

            # Test streaming version with different chunk sizes
            for chunk_size in chunk_sizes:
                times = []
                for _ in range(iterations):
                    start = time.time()
                    tokens = list(
                        stream_tokenize(temp_path, special_tokens, chunk_size)
                    )
                    end = time.time()
                    times.append(end - start)
                avg_time = sum(times) / len(times)
                print(
                    f"{size_kb:>10} {chunk_size:>12} {'streaming':>15} {avg_time:>10.4f}"
                )

        finally:
            import os

            os.unlink(temp_path)

        print("-" * 80)


if __name__ == "__main__":
    # Run the original tests
    print("Running correctness tests...")
    test_cases = [
        ("Hello <START> world <END> test", ["<START>", "<END>"]),
        ("They've <START> done it <END> here", ["<START>", "<END>"]),
        ("<START><MID><END>test", ["<START>", "<MID>", "<END>"]),
    ]

    for text, special_tokens in test_cases:
        print(f"\nTesting: {text}")
        for chunk_size in [1, 2, 3, 4, 5, 10, 20]:
            result = test_special_tokenization(text, special_tokens, chunk_size)
            print(f"Chunk size {chunk_size}: {'PASS' if result else 'FAIL'}")

    # Run benchmarks
    print("\nRunning benchmarks...")
    benchmark_tokenizers()


# if __name__ == "__main__":
#     # Test cases focusing on special tokens and their interactions with regular tokens
#     test_cases = [
#         # Basic special token tests
#         ("Hello <START> world <END> test", ["<START>", "<END>"]),
#         # Nested special tokens
#         ("Hello <START> <INNER> world <END>", ["<START>", "<INNER>", "<END>"]),
#         # Special tokens with contractions
#         ("They've <START> done it <END> here", ["<START>", "<END>"]),
#         # Special tokens at chunk boundaries
#         ("abc<START>def", ["<START>"]),
#         ("abc<START>", ["<START>"]),
#         ("<START>def", ["<START>"]),
#         # Multiple special tokens together
#         ("<START><MID><END>test", ["<START>", "<MID>", "<END>"]),
#         ("<<<START>>>", ["<START>", "<MID>", "<END>"]),
#         # Long text with multiple special tokens
#         (
#             """This is a longer text with <START> multiple special tokens
#         and they've done it <MID> right here. The system's working
#         <END> as expected.""",
#             ["<START>", "<MID>", "<END>"],
#         ),
#     ]
#
#     for text, special_tokens in test_cases:
#         print(f"\nTesting with text: {text}")
#         print(f"Special tokens: {special_tokens}")
#         for chunk_size in [1, 2, 3, 4, 5, 10, 20]:
#             result = test_special_tokenization(text, special_tokens, chunk_size)
#             print(f"Chunk size {chunk_size}: {'PASS' if result else 'FAIL'}")
