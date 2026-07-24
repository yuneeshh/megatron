"""Hidden-layer construction."""

from neuron import create_neuron


def create_layer(num_neurons, num_inputs):
    """Create neurons that each receive all inputs to this layer."""
    return [create_neuron(num_inputs) for _ in range(num_neurons)]
