import cProfile
import inspect
import io
import os
import pstats
import re
from io import StringIO
from pathlib import Path
from time import time
from typing import Dict, List, Tuple

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


# Method 1: Create a profiled subclass
class ProfiledBPENaive(BPENaive):
    def __init__(self, *args, **kwargs):
        self._init_start = time()
        # Create profiler with subcalls enabled
        self._profiler = cProfile.Profile(subcalls=True, builtins=True)
        self._profiler.enable()
        super().__init__(*args, **kwargs)
        self._profiler.disable()
        self._init_end = time()
        self._class_methods = self._get_class_methods()

    @staticmethod
    def _merge(old_seq, merge_pair, count_dict, delta):
        # We'll wrap the original static method to ensure it's profiled
        try:
            frame = inspect.currentframe()
            if frame and frame.f_back:
                caller = frame.f_back.f_code.co_name
            else:
                caller = "unknown"
        except Exception:
            caller = "unknown"
        finally:
            if frame:
                del frame  # Avoid reference cycles

        result = BPENaive._merge(old_seq, merge_pair, count_dict, delta)
        return result

    def explore_function(self, func_pattern: str):
        """
        Explore a function's callers and callees with better static method handling.
        """
        s = StringIO()
        stats = pstats.Stats(self._profiler, stream=s)

        # First find all possible function names that might match
        stats.sort_stats("cumulative")
        stats.print_stats()
        all_stats = s.getvalue()

        matching_funcs = []
        static_method_names = []

        # Look for both direct matches and class method matches
        for line in all_stats.split("\n"):
            if (
                func_pattern in line
                or f"{self.__class__.__name__}.{func_pattern}" in line
            ):
                parts = line.split()
                if len(parts) >= 6:
                    func_desc = " ".join(parts[5:])
                    matching_funcs.append(func_desc)
                    if "static" in func_desc:
                        static_method_names.append(func_desc)

        if not matching_funcs:
            print(f"No functions found matching pattern '{func_pattern}'")
            return

        print(f"\nFound {len(matching_funcs)} matching functions:")
        for i, func in enumerate(matching_funcs, 1):
            print(f"{i}. {func}")

        # For each matching function, show detailed stats
        for func in matching_funcs:
            print(f"\n=== Function: {func} ===")

            # Get callers
            print("\nCallers:")
            print("-" * 80)
            s = StringIO()
            stats.print_callers(func)
            callers = s.getvalue()
            if callers.strip():
                print(callers)
            else:
                print("No direct callers found")

            # Get callees
            print("\nCallees:")
            print("-" * 80)
            s = StringIO()
            stats.print_callees(func)
            callees = s.getvalue()
            if callees.strip():
                print(callees)
            else:
                print("No callees found")

            # Get function-specific stats
            print("\nDetailed statistics:")
            print("-" * 80)
            s = StringIO()
            specific_stats = pstats.Stats(self._profiler, stream=s)
            specific_stats.sort_stats("cumulative")
            # Use a filter to only show stats for this function
            specific_stats.print_stats(f".*{re.escape(func)}.*")
            filtered_stats = s.getvalue()
            if filtered_stats.strip():
                print(filtered_stats)
            else:
                print("No detailed statistics available")

    def _get_class_methods(self) -> Dict[str, str]:
        method_types = {}
        for name, value in inspect.getmembers(self.__class__):
            if name.startswith("_") or name in ["print_profile"]:
                continue
            if inspect.ismethod(value):
                method_types[name] = "classmethod"
            elif inspect.isfunction(value):
                if isinstance(self.__class__.__dict__.get(name), staticmethod):
                    method_types[name] = "staticmethod"
                else:
                    method_types[name] = "method"
        return method_types

    def _clean_function_name(self, func_desc: str) -> Tuple[str, str]:
        """
        Clean and categorize a function description from the profiler output.
        Returns (cleaned_name, category)
        """
        # Handle built-in methods
        if func_desc.startswith("{method"):
            match = re.search(r"'([^']+)'", func_desc)
            if match:
                return f"builtin.{match.group(1)}", "builtin"

        # Handle built-in functions
        if func_desc.startswith("{built-in method"):
            match = re.search(r"method ([^\}]+)", func_desc)
            if match:
                return f"builtin.{match.group(1)}", "builtin"

        # Handle class methods
        class_name = self.__class__.__name__.replace("Profiled", "")
        for method_name, method_type in self._class_methods.items():
            if method_name in func_desc:
                return f"{class_name}.{method_name}", method_type

        # Handle file paths
        if "/" in func_desc:
            # Extract filename and function name
            parts = func_desc.split("/")
            file_func = parts[-1].split(":")
            if len(file_func) > 1:
                # Handle line numbers and function names
                match = re.search(r"(\w+)\(([^)]+)\)", file_func[1])
                if match:
                    return f"{file_func[0]}::{match.group(2)}", "other"
            return os.path.basename(func_desc), "other"

        return func_desc, "other"

    def _format_stats(
        self, min_percentage: float = 1.0
    ) -> List[Tuple[str, float, float, str]]:
        s = StringIO()
        stats = pstats.Stats(self._profiler, stream=s)
        stats.sort_stats("cumulative")
        stats.print_stats()
        raw_stats = s.getvalue()

        total_time = self._init_end - self._init_start
        formatted_stats = []
        in_stats = False

        seen_times = set()  # To avoid duplicate entries

        for line in (lines := raw_stats.split("\n")):
            if "ncalls" in line and "tottime" in line:
                in_stats = True
                continue
            if in_stats and line.strip():
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        cumtime = float(parts[3])
                        percentage = (cumtime / total_time) * 100

                        if percentage >= min_percentage:
                            # Get the function description
                            func_desc = " ".join(parts[5:])
                            clean_name, category = self._clean_function_name(func_desc)

                            # Avoid duplicate entries (same name and time)
                            entry_key = (clean_name, cumtime)
                            if entry_key not in seen_times:
                                seen_times.add(entry_key)
                                formatted_stats.append(
                                    (clean_name, cumtime, percentage, category)
                                )
                    except (ValueError, IndexError):
                        continue

        return sorted(formatted_stats, key=lambda x: x[1], reverse=True)

    def print_profile(
        self,
        min_percentage: float = 1.0,
        show_total: bool = True,
        group_by_type: bool = True,
    ):
        total_time = self._init_end - self._init_start
        stats = self._format_stats(min_percentage)

        print("\n=== Profiling Results ===")
        if show_total:
            print(f"Total runtime: {total_time:.4f} seconds\n")

        if group_by_type:
            # Define groups and their display order
            groups = {
                "Instance Methods": [],
                "Static Methods": [],
                "Class Methods": [],
                "Built-in Functions": [],
                "Other": [],
            }

            # Categorize stats
            for stat in stats:
                func_name, func_time, percentage, category = stat
                if category == "method":
                    groups["Instance Methods"].append(stat)
                elif category == "staticmethod":
                    groups["Static Methods"].append(stat)
                elif category == "classmethod":
                    groups["Class Methods"].append(stat)
                elif category == "builtin":
                    groups["Built-in Functions"].append(stat)
                else:
                    groups["Other"].append(stat)

            # Print each non-empty group
            for group_name, group_stats in groups.items():
                if group_stats:
                    print(f"\n{group_name}:")
                    print(f"{'Function':<50} {'Time (s)':<12} {'% of Total':<10}")
                    print("-" * 72)
                    for func_name, func_time, percentage, _ in group_stats:
                        print(
                            f"{func_name[:50]:<50} {func_time:>10.4f}s {percentage:>8.1f}%"
                        )
        else:
            print(f"{'Function':<50} {'Time (s)':<12} {'% of Total':<10}")
            print("-" * 72)
            for func_name, func_time, percentage, _ in stats:
                print(f"{func_name[:50]:<50} {func_time:>10.4f}s {percentage:>8.1f}%")

        # Print filtered functions note
        all_stats = self._format_stats(0)
        if len(all_stats) > len(stats):
            filtered_count = len(all_stats) - len(stats)
            filtered_time = sum(stat[1] for stat in all_stats[len(stats) :])
            filtered_percentage = (filtered_time / total_time) * 100
            print(
                f"\nNote: {filtered_count} functions contributing {filtered_percentage:.1f}% of runtime were filtered out"
            )

        # print("\nDetailed analysis of specific function and its subcalls:")
        # stats = pstats.Stats(self._profiler)
        # # stats.print_callees("tokenizer.py::_merge")  # Replace with your specific method name
        # stats.print_callees(
        #     "/home/sam/github/cs336-hw1/cs336_basics/tokenizer/tokenizer.py:209(_merge)"
        # )
        # stats.print_stats()


if __name__ == "__main__":
    sample_text = (
        """
    This is a sample text that will be used for profiling.
    Make sure it's long enough to get meaningful results.
    """
        * 100
    )

    # corpus_path = Path("./test_data/test.txt")
    # corpus_path = Path("../../data/TinyStoriesV2-GPT4-train.txt")
    corpus_path = Path("../../data/owt_train.txt")
    vocab_size = 32000  # 'initial' size is 256 (bytes)
    # my_tokenizer = BPENaive(corpus_path, vocab_size)
    # profile_tokenizer(my_tokenizer, sample_text)
    my_tokenizer = ProfiledBPENaive(
        corpus_path=corpus_path,
        max_vocab_size=vocab_size,
        serialize=True,
        special_tokens=["<|endoftext|>"],
    )
    my_tokenizer.print_profile(min_percentage=0.1, group_by_type=False)
    # my_tokenizer.explore_function("_merge")
