"""Single-neuron creation and activation functions."""

import random


def leaky_relu(x):
    """Activate a raw neuron value while retaining 1% of negative signals."""
    return x if x > 0 else 0.01 * x


def leaky_relu_deriv(x):
    """Return the Leaky ReLU slope needed by the chain rule."""
    return 1.0 if x > 0 else 0.01


def create_neuron(num_inputs):
    """Create one neuron with a random weight per input and a random bias."""
    return {
        "weights": [random.uniform(-1, 1) for _ in range(num_inputs)],
        "bias": random.uniform(-1, 1),
    }
