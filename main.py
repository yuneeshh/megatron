"""Entry point for the complete from-scratch neural-network example."""

import logging

import embed  # Integration point; fixed vectors remain unchanged below.
import log_config  # Configure logging before executing network operations.
from final_layer import create_final_layer
from network import create_network, forward
from train import train

logger = logging.getLogger(__name__)


def log_predictions(title, training_data, hidden_network, final_layer, vocab):
    """Display predictions for each labelled example without updating weights."""
    logger.info(title)
    for inputs, correct in training_data:
        probs = forward(inputs, hidden_network, final_layer, vocab)
        predicted = max(probs, key=probs.get)
        logger.info(
            f"  target={correct:8s} predicted={predicted:8s} "
            f"probs={ {k: round(v, 2) for k, v in probs.items()} }"
        )


def main():
    """Create the model, train it, and report before/after predictions."""
    vocab = ["blue", "green", "cat", "running", "happy"]

    # These fake vectors are intentionally unchanged. Merely importing embed.py
    # provides an integration point without changing random-number consumption.
    training_data = [
        ([0.5, -0.3, 0.8, 0.1, -0.6], "blue"),
        ([0.2, 0.4, -0.1, 0.6, 0.3], "green"),
        ([-0.4, 0.1, 0.5, -0.2, 0.7], "happy"),
    ]

    hidden_network = create_network(
        num_layers=2, neurons_per_layer=5, num_inputs=5
    )
    final_layer = create_final_layer(vocab, num_inputs=5)

    log_predictions(
        "--- Before training ---",
        training_data,
        hidden_network,
        final_layer,
        vocab,
    )

    train(
        training_data,
        hidden_network,
        final_layer,
        vocab,
        epochs=300,
        learning_rate=0.3,
        detailed_trace=True,
    )

    log_predictions(
        "\n--- After training ---",
        training_data,
        hidden_network,
        final_layer,
        vocab,
    )

    logger.info("\n--- Custom Run ---")
    custom_inputs = [-0.4, 0.1, 0.5, -0.2, 0.7]
    probs = forward(custom_inputs, hidden_network, final_layer, vocab)
    logger.info(f"  custom inputs={custom_inputs}")
    logger.info(
        f"  probs={ {k: round(v, 2) for k, v in probs.items()} }"
    )


if __name__ == "__main__":
    main()
