"""Vocabulary output-layer construction and softmax probabilities."""

import math

from neuron import create_neuron


def create_final_layer(vocab, num_inputs):
    """Create one scoring neuron for each possible output word."""
    return {word: create_neuron(num_inputs) for word in vocab}


def softmax(scores: dict):
    """Convert raw word scores into probabilities that sum to one."""
    values = list(scores.values())
    max_val = max(values)
    exps = {k: math.exp(v - max_val) for k, v in scores.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}
