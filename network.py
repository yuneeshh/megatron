"""Network creation and forward propagation through all layers."""

import logging

from final_layer import softmax
from layer import create_layer
from neuron import leaky_relu

logger = logging.getLogger(__name__)


def create_network(num_layers, neurons_per_layer, num_inputs):
    """Build fully connected hidden layers in their forward-pass order."""
    layers = []
    n_in = num_inputs
    for _ in range(num_layers):
        layers.append(create_layer(neurons_per_layer, n_in))
        n_in = neurons_per_layer
    return layers


def forward_full(inputs, hidden_network, final_layer, vocab, trace=False):
    """Run inference and retain activations/raw z values for backpropagation."""
    activations = [inputs]
    zs = []

    if trace:
        logger.debug("\n  [FORWARD PASS]")
        logger.debug("  Network input: %s", inputs)

    current = inputs
    for layer_index, layer in enumerate(hidden_network):
        z_layer = []
        h_layer = []
        if trace:
            logger.debug("\n  Hidden layer %d", layer_index + 1)
        for neuron_index, neuron in enumerate(layer):
            # Products show how much each incoming connection contributes.
            products = [
                i * w for i, w in zip(current, neuron["weights"])
            ]
            z = sum(products) + neuron["bias"]
            h = leaky_relu(z)
            z_layer.append(z)
            h_layer.append(h)
            if trace:
                logger.debug("    neuron %d:", neuron_index)
                logger.debug(
                    "      inputs  = %s", [round(v, 6) for v in current]
                )
                logger.debug(
                    "      weights = %s",
                    [round(v, 6) for v in neuron["weights"]],
                )
                logger.debug(
                    "      products(input * weight) = %s",
                    [round(v, 6) for v in products],
                )
                logger.debug(
                    "      z = sum(products) + bias(%.6f) = %.6f",
                    neuron["bias"],
                    z,
                )
                logger.debug("      activation = LeakyReLU(z) = %.6f", h)
        zs.append(z_layer)
        activations.append(h_layer)
        current = h_layer

    # Output neurons produce raw scores; softmax supplies the probabilities.
    final_z = {}
    for word in vocab:
        neuron = final_layer[word]
        products = [
            i * w for i, w in zip(current, neuron["weights"])
        ]
        final_z[word] = sum(products) + neuron["bias"]
        if trace:
            logger.debug("\n  Final neuron '%s':", word)
            logger.debug(
                "    inputs  = %s", [round(v, 6) for v in current]
            )
            logger.debug(
                "    weights = %s",
                [round(v, 6) for v in neuron["weights"]],
            )
            logger.debug(
                "    products = %s", [round(v, 6) for v in products]
            )
            logger.debug(
                "    raw score = sum(products) + bias(%.6f) = %.6f",
                neuron["bias"],
                final_z[word],
            )

    probs = softmax(final_z)
    if trace:
        logger.debug("\n  Softmax converts raw scores into probabilities:")
        logger.debug(
            "    raw scores    = %s",
            {k: round(v, 6) for k, v in final_z.items()},
        )
        logger.debug(
            "    probabilities = %s",
            {k: round(v, 6) for k, v in probs.items()},
        )
    return activations, zs, probs


def forward(inputs, hidden_network, final_layer, vocab):
    """Run inference when only the final probabilities are needed."""
    _, _, probs = forward_full(
        inputs, hidden_network, final_layer, vocab
    )
    return probs
