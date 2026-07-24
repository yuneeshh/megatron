"""Training-loop orchestration."""

import logging

from backprop import backward

logger = logging.getLogger(__name__)


def train(
    training_data,
    hidden_network,
    final_layer,
    vocab,
    epochs,
    learning_rate,
    detailed_trace,
):
    """Train once per example per epoch, preserving the original order."""
    for epoch in range(epochs):
        total_loss = 0
        for example_index, (inputs, correct) in enumerate(training_data):
            # Preserve the original detailed trace for examples 0 and 1.
            trace_this_step = (
                detailed_trace
                and epoch == 0
                and example_index in [0, 1]
            )
            total_loss += backward(
                inputs,
                correct,
                hidden_network,
                final_layer,
                vocab,
                lr=learning_rate,
                trace=trace_this_step,
            )
        if epoch % 30 == 0:
            logger.info(
                f"Epoch {epoch:3d}  "
                f"avg loss={total_loss / len(training_data):.4f}"
            )
