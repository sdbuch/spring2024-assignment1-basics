#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import heapq
import linecache
import operator
import os
import pprint
import sys
import threading
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from functools import reduce, total_ordering
from pathlib import Path
from types import NoneType
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import psutil
import pytest
import regex as re
from sortedcontainers import SortedDict, SortedSet

# from memory_monitor import memory_tracker, track_memory

# Tokenizer class and tests


@dataclass
class Node:
    value: bytes
    rank: int
    next: Optional["Node"] = None
    prev: Optional["Node"] = None

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"Node(value={self.value})"

    def __hash__(self) -> int:
        # Use object's id as hash - nodes are only equal if they're the same object
        return id(self)

    def __eq__(self, other: Any) -> bool:
        # Nodes are equal only if they're the same object
        if not isinstance(other, Node):
            return NotImplemented
        return id(self) == id(other)


@dataclass
class DLinkedList:
    head: Optional[Node] = None
    init_rank: int = 0

    def append(self, value: bytes) -> None:
        if not self.head:
            self.head = Node(value, self.init_rank)
            return

        current = self.head
        while current.next:
            current = current.next
        current.next = Node(value, current.rank + 1)
        current.next.prev = current

    def as_byte_sequence(self) -> Tuple[bytes, ...]:
        return tuple(node.value for node in self)

    def __str__(self) -> str:
        return " <-> ".join(str(node.value) for node in self)

    def __iter__(self) -> Iterator[Node]:
        current = self.head
        while current:
            yield current
            current = current.next


def pretokenizer_gpt2(text: str) -> list[str]:
    # GPT2 pretokenizer
    # https://github.com/openai/tiktoken/pull/234/files
    pat_str = (
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    return re.findall(pat_str, text)


@dataclass
class BPENaive:
    """BPE-based utf8 tokenizer."""

    corpus_path: Path = Path(".")
    max_vocab_size: int = 1024
    pretokenizer: Callable = pretokenizer_gpt2
    special_tokens: List[str] = field(default_factory=list)
    vocab: Dict[int, bytes] = field(default_factory=dict)
    merges: list[tuple[bytes, bytes]] = field(default_factory=list)
    vocab_inverse: Dict[bytes, int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if len(self.vocab) == 0:
            # We allow for vocab/merges to be passed in from a pretrained tokenizer.
            self._map_special_tokens()
            self._train(self.corpus_path)
        # BUG: could be a bug, not being careful...
        self.vocab_inverse = {v: k for k, v in self.vocab.items()}

    def _map_special_tokens(self):
        # Scheme: split the string on special tokens. Tokenize these manually.
        # Then operate normally on the rest. (no cou nting on special tokens)
        # NOTE: Assume "valid" special tokens (no substring issues)
        for i, token in enumerate(self.special_tokens):
            # self.special_tokens_dict[token] = i
            self.vocab[i] = token.encode("utf8")

    def _split_on_special_tokens(self, text: str, training=False) -> List[str]:
        """training=True strips the special tokens. False preserves them (for encoding)."""

        if len(self.special_tokens) == 0:
            # short circuit: empty regex breaks everything up
            return [text]
        patterns = sorted(self.special_tokens, key=len, reverse=True)
        special_token_pat = "|".join(map(re.escape, patterns))
        if training:
            return re.split(special_token_pat, text)
        else:
            return re.split("(" + special_token_pat + ")", text)

    def _train(self, corpus_path: Path):
        # Initialize with bytes.
        for i in range(256):
            self.vocab[i + len(self.special_tokens)] = bytes([i])
        # Get corpus content
        with open(corpus_path, "r") as file:
            corpus = file.read()
        # Strip special tokens.
        corpus = "".join(self._split_on_special_tokens(corpus, training=True))
        # Pre-tokenize and create count table
        text = self.pretokenizer(corpus)
        # BUG: we count overlapping tokens, but should probably disregard (sentencepiece...)
        # count_dict_pretoken: Dict[Tuple[int, ...], int] = {}
        count_dict_pretoken: Dict[Tuple[bytes, ...], int] = {}
        # count_dict_pair = Dict[tuple[int, int], int] = {}
        for pretoken in text:
            token = tuple(bytes([b]) for b in pretoken.encode("utf8"))
            if token not in count_dict_pretoken:
                count_dict_pretoken[token] = 1
            else:
                count_dict_pretoken[token] += 1
        while len(self.vocab) < self.max_vocab_size:
            # count frequencies
            count_dict: Dict[tuple[bytes, bytes], int] = {}
            for pretoken in count_dict_pretoken.keys():
                for pair in zip(pretoken, pretoken[1:]):
                    if pair not in count_dict:
                        count_dict[pair] = count_dict_pretoken[pretoken]
                    else:
                        count_dict[pair] += count_dict_pretoken[pretoken]
            # Most frequent scan
            most_frequent_pair = None
            max_count = -1
            for pair in sorted(
                count_dict.keys()
            ):  # Guarantees lexicographic merge ordering
                count = count_dict[pair]
                if count >= max_count:
                    most_frequent_pair = pair
                    max_count = count
            # Log merge and update vocab
            if most_frequent_pair is None:
                # Nothing left to merge.
                break
            self.merges.append(most_frequent_pair)
            with pytest.raises(KeyError):
                print(self.vocab[len(self.vocab)])
            self.vocab[len(self.vocab)] = reduce(operator.add, most_frequent_pair)
            # Update the pretoken count dict (manual merge)
            # PERF: conversion to/from tuple is wasteful (array.array not hashable so need to...)
            for pretoken in list(count_dict_pretoken):
                new_key = list(pretoken)  # can't avoid this?
                new_key = BPENaive._merge(new_key, most_frequent_pair)
                new_key = tuple(new_key)  # keys need to be mutable...
                if new_key != pretoken:
                    count_dict_pretoken[new_key] = count_dict_pretoken[pretoken]
                    del count_dict_pretoken[pretoken]

    @staticmethod
    def _merge(token_seq: List[bytes], merge_pair: tuple[bytes, bytes]) -> List[bytes]:
        ptr = 0
        while ptr < len(token_seq) - 1:
            byte_seq = token_seq[ptr]
            next_byte_seq = token_seq[ptr + 1]
            if (byte_seq, next_byte_seq) == merge_pair:
                token_seq[ptr] = byte_seq + next_byte_seq
                token_seq[ptr + 1 :] = token_seq[ptr + 2 :]
                # token_seq = token_seq[:-1]
            else:
                ptr += 1
        return token_seq

    def encode(self, text: str) -> list[int]:
        # The encoding process involves converting to utf8, then applying the merges.
        # We also need to start by treating special characters in a special way.
        text_split_special = self._split_on_special_tokens(text, training=False)
        text_encoded = []
        for segment in text_split_special:
            if segment in self.special_tokens:
                segment_encoded = segment.encode("utf8")
                token = self.vocab_inverse[segment_encoded]
                text_encoded.append(token)
            else:
                segment_pretokenized = self.pretokenizer(segment)
                for segment in segment_pretokenized:
                    segment_encoded = list(bytes([b]) for b in segment.encode("utf8"))
                    for merge in self.merges:
                        segment_encoded = BPENaive._merge(segment_encoded, merge)
                    tokens = [
                        self.vocab_inverse[byte_seq] for byte_seq in segment_encoded
                    ]
                    text_encoded += tokens
        return text_encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for string in iterable:
            yield from self.encode(string)

    def decode(self, tokens: list[int]) -> str:
        # The decoding process simply involves the vocab lookup.
        output = bytes([])
        for token in tokens:
            output += self.vocab[token]
        return output.decode("utf8", errors="replace")


# TODO: Fast encoding implementation: support chunking, possible merge optimizations.
# Chunking notes.
# Chunk with (forward) overlaps (one-side).
# Pick the size of the overlap region to be 127 bytes: tiktoken cl100k-base has
#   max token size 128byte. (remember that special unicode chars are 4byte...)
# This choice gives us a guarantee as long as the max vocab el. length does not
#   exceed 128 bytes: if a merge crosses the overlap boundary in one of two
#   overlapped chunks, it does NOT cross the overlap boundary in the other
#   overlapped chunk!
# So: for each chunk, we track if a merge occurs across its overlap boundary.
#   If it does, we mark that chunk (eg add a flag to the chunk data structure:
#   could just be a list of 2-el lists, 1 per chunk, which marks start/end
#   boundary crossing).
# After processing chunks, we de-overlap using this list. De-overlapping is easy if
#   only one of two elements is marked, or neither (we just preserve the overlapped region
#   from one chunk, or the first chunk in the latter case). If both elements are marked,
#   it means we merged across both overlap boundaries independently (because we set the
#   overlap region appropriately). We need to carefully stitch in this case.
# The above discussion applies to encoding. For training, we don't need to
#   worry about resolving across overlap boundaries, but we do need to avoid
#   double-counting...
@dataclass
class BPEImproved:
    """BPE-based utf8 tokenizer. With some optimizations"""

    corpus_path: Path = Path(".")
    max_vocab_size: int = 1024
    pretokenizer: Callable = pretokenizer_gpt2
    special_tokens: List[str] = field(default_factory=list)
    vocab: Dict[int, bytes] = field(default_factory=dict)
    merges: list[tuple[bytes, bytes]] = field(default_factory=list)
    vocab_inverse: Dict[bytes, int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if len(self.vocab) == 0:
            # We allow for vocab/merges to be passed in from a pretrained tokenizer.
            self._map_special_tokens()
            self._train(self.corpus_path)
        # BUG: could be a bug, not being careful...
        self.vocab_inverse = {v: k for k, v in self.vocab.items()}

    def _map_special_tokens(self):
        # Scheme: split the string on special tokens. Tokenize these manually.
        # Then operate normally on the rest. (no cou nting on special tokens)
        # NOTE: Assume "valid" special tokens (no substring issues)
        for i, token in enumerate(self.special_tokens):
            self.vocab[i] = token.encode("utf8")

    def _split_on_special_tokens(self, text: str, training=False) -> List[str]:
        """training=True strips the special tokens. False preserves them (for encoding)."""

        if len(self.special_tokens) == 0:
            # short circuit: empty regex breaks everything up
            return [text]
        patterns = sorted(self.special_tokens, key=len, reverse=True)
        special_token_pat = "|".join(map(re.escape, patterns))
        if training:
            return re.split(special_token_pat, text)
        else:
            return re.split("(" + special_token_pat + ")", text)

    # @track_memory('training')
    def _train(self, corpus_path: Path):
        # Initialize with bytes.
        for i in range(256):
            self.vocab[i + len(self.special_tokens)] = bytes([i])
        # Get corpus content
        # TODO: Implement chunked mode...
        with open(corpus_path, "r") as file:
            corpus = file.read()
        # Strip special tokens.
        corpus = "".join(self._split_on_special_tokens(corpus, training=True))
        # Pre-tokenize and create count table
        # TODO: should be simplifying by counting repetitions of pretokens
        text = self.pretokenizer(corpus)
        merge_helper = BPEHelper(text)
        while len(self.vocab) < self.max_vocab_size:
            new_merge = merge_helper.merge_most_frequent()
            if new_merge is None:
                break
            self.merges.append(new_merge)
            self.vocab[len(self.vocab)] = reduce(operator.add, new_merge)

    # @track_memory('encoding')
    def encode(self, text: str) -> list[int]:
        # The encoding process involves converting to utf8, then applying the merges.
        # We also need to start by treating special characters in a special way.
        text_split_special = self._split_on_special_tokens(text, training=False)
        segment_list = []
        for segment in text_split_special:
            if segment in self.special_tokens:
                segment_list.append(segment)
            else:
                segment_pretokenized = self.pretokenizer(segment)
                for string in segment_pretokenized:
                    segment_list.append(string)
        merge_helper = BPEHelper(segment_list)
        merge_helper.merge_from_list(self.merges)
        text_encoded = []
        for idx, string in enumerate(segment_list):
            if string in self.special_tokens:
                string_encoded = string.encode("utf8")
                token = self.vocab_inverse[string_encoded]
                text_encoded.append(token)
            else:
                string_encoded = merge_helper.string_list[idx].as_byte_sequence()
                text_encoded += [
                    self.vocab_inverse[byte_seq] for byte_seq in string_encoded
                ]
        return text_encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for string in iterable:
            yield from self.encode(string)

    def decode(self, tokens: list[int]) -> str:
        # The decoding process simply involves the vocab lookup.
        output = bytes([])
        for token in tokens:
            output += self.vocab[token]
        return output.decode("utf8", errors="replace")


class BPEHelper:
    def __init__(self, texts: List[str]):
        self.priority_queue: SortedDict[
            tuple[int, tuple[bytes, bytes]], SortedSet[Node]
        ] = SortedDict()
        self.pair_counts: Dict[tuple[bytes, bytes], tuple[int, SortedSet[Node]]] = {}
        self.string_list: List[DLinkedList] = []
        self._initialize_from_strings(texts)

    def _initialize_from_strings(self, texts: List[str]) -> None:
        dlls = []
        last_len = 0
        for text in texts:
            string = DLinkedList(init_rank=last_len)
            for char in text:
                for i in char.encode("utf8"):
                    b = bytes([i])
                    string.append(b)
                    last_len += len(b)

            dlls.append(string)
        self._initialize_counts(dlls)
        self.string_list = dlls

    def _initialize_counts(self, strings: List[DLinkedList]) -> None:
        for string in strings:
            current = string.head
            while current and current.next:
                pair = (current.value, current.next.value)
                count, ptrs = self.pair_counts.get(
                    pair, (0, SortedSet(key=lambda x: x.rank))
                )
                ptrs.add(current)
                self.pair_counts[pair] = (count + 1, ptrs)
                # TODO: not counting overlap here (hard to get right...)
                current = current.next

        for pair, (count, ptrs) in self.pair_counts.items():
            self.priority_queue[(count, pair)] = ptrs

    def _update_priority_queue(
        self,
        pair: tuple[bytes, bytes],
        node: Node,
        count_delta: int,
        tabulate_counts: bool = True,
    ) -> None:
        """Helper to update counts and pointers for a pair in both data structures"""
        if pair not in self.pair_counts:
            if count_delta > 0:
                self.pair_counts[pair] = (
                    count_delta,
                    SortedSet([node], key=lambda x: x.rank),
                )
                if tabulate_counts:
                    self.priority_queue[(count_delta, pair)] = SortedSet(
                        [node], key=lambda x: x.rank
                    )
            return

        old_count, ptrs = self.pair_counts[pair]
        new_count = old_count + count_delta
        if count_delta > 0:
            ptrs.add(node)
        else:
            ptrs.remove(node)

        if tabulate_counts:
            del self.priority_queue[(old_count, pair)]

        if new_count > 0:
            self.pair_counts[pair] = (new_count, ptrs)
            if tabulate_counts:
                self.priority_queue[(new_count, pair)] = ptrs
        else:
            del self.pair_counts[pair]

    def _process_merges(
        self, pair: tuple[bytes, bytes], nodes: SortedSet[Node], training: bool = True
    ) -> None:
        if pair[0] == pair[1]:
            # This case can have overlap issues. We need to be a bit careful.
            # Our approach is to ensure the merges occur safely.
            # BUG: (possible) this is a pretty brittle implementation. if node.rank overflows it breaks, + not multiproc-able (needs to operate sequentially)
            node_list = nodes.copy()
            overlap_toggle = 1
            for node in nodes:
                if not node.next or not node.next.next:
                    continue
                if not node.prev or not node.prev.value == node.value:
                    # Reset the toggle if we're at the start of a new run
                    overlap_toggle = 1
                if node.next.value == node.value and node.next.next.value == node.value:
                    # Run of three. We should skip the previous (if toggle).
                    if overlap_toggle:
                        node_list.remove(node.next)
                    overlap_toggle ^= 1
        else:
            node_list = nodes

        for node in node_list:
            if not node.next:  # Skip if no next node to merge with
                continue

            # Update counts for affected pairs
            if node.prev:
                before_pair = (node.prev.value, node.value)
                update_pq = before_pair != pair  # This only happens with overlap
                self._update_priority_queue(
                    before_pair, node.prev, -1, update_pq & training
                )

            # If there's a pair after the one we're merging, update its count
            if node.next.next:
                after_pair = (node.next.value, node.next.next.value)
                update_pq = after_pair != pair  # This only happens with overlap
                self._update_priority_queue(
                    after_pair, node.next, -1, update_pq & training
                )

            # Perform merge
            node.value += node.next.value
            old_next = node.next
            new_next = old_next.next
            node.next = new_next
            if new_next:
                new_next.prev = node
            del old_next

            # Add new pairs
            if node.prev:
                self._update_priority_queue(
                    (node.prev.value, node.value), node.prev, 1, training
                )
            if node.next:
                self._update_priority_queue(
                    (node.value, node.next.value), node, 1, training
                )

        del self.pair_counts[pair]

    def merge_most_frequent(self) -> tuple[bytes, bytes] | None:
        """Merge the most frequent pair and update data structures"""
        if not self.priority_queue:
            return None

        (count, pair), nodes = self.priority_queue.popitem()
        self._process_merges(pair, nodes, training=True)
        return pair

    def merge_from_list(self, merge_list: List[tuple[bytes, bytes]]) -> None:
        for pair in merge_list:
            if pair not in self.pair_counts:
                continue
            count, nodes = self.pair_counts[pair]
            self._process_merges(pair, nodes, training=False)


def measure_runtime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Function '{func.__name__}' took {runtime:.4f} seconds to run")
        return result

    return wrapper


def test_BPE_naive():
    corpus_path = Path("./test_data/test.txt")
    vocab_size = 512  # 'initial' size is 256 (bytes)
    tokenizer = BPENaive(corpus_path, vocab_size, special_tokens=["<|STOP|>"])

    test_str = (
        "Hello, world! This is a test.<|STOP|>여러분들, 안녕하세요? 12,34 1 -- 3 #$@$)@"
    ) * 1

    encoded = tokenizer.encode(test_str)
    decoded = tokenizer.decode(encoded)
    assert decoded == test_str


def test_BPE_improved():
    corpus_path = Path("../../data/TinyStoriesV2-GPT4-train.txt")
    vocab_size = 10000  # 'initial' size is 256 (bytes)

    @measure_runtime
    def create_tokenizer():
        return BPEImproved(
            corpus_path,
            vocab_size,
            special_tokens=["<|endoftext|>"],
        )
    tokenizer = create_tokenizer()

    # test_str = (
    #     "Hello, world! This is a test.<|STOP|>여러분들, 안녕하세요? 12,34 1 -- 3 #$@$)@"
    # ) * 1
    test_str = "10000000000000000000 00000000000"

    encoded = tokenizer.encode(test_str)
    decoded = tokenizer.decode(encoded)
    assert decoded == test_str


if __name__ == "__main__":
    # Some benchmarking
    test_BPE_improved()
