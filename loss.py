"""Prediction loss calculation."""

import math


def negative_log_likelihood(probabilities, correct_word):
    """Penalize the model when it assigns low probability to the correct word."""
    return -math.log(max(probabilities[correct_word], 1e-9))
