"""Analytic backpropagation and in-place gradient-descent updates."""

import logging

from loss import negative_log_likelihood
from network import forward_full
from neuron import leaky_relu_deriv

logger = logging.getLogger(__name__)


def backward(
    inputs, correct_word, hidden_network, final_layer, vocab, lr, trace=False
):
    """Run one forward/backward training step and return its loss."""
    activations, zs, probs = forward_full(
        inputs, hidden_network, final_layer, vocab, trace=trace
    )
    loss = negative_log_likelihood(probs, correct_word)

    if trace:
        logger.debug("\n  [LOSS]")
        logger.debug("  Correct word = '%s'", correct_word)
        logger.debug(
            "  loss = -log(P(correct word)) = -log(%.6f) = %.6f",
            probs[correct_word], loss,
        )
        logger.debug("  Learning rate (chosen by us) = %s", lr)

    # Softmax + negative log likelihood simplifies to probability - target.
    d_scores = {
        w: probs[w] - (1.0 if w == correct_word else 0.0)
        for w in vocab
    }
    if trace:
        logger.debug("\n  [BACKWARD PASS: FINAL LAYER]")
        logger.debug("  score gradient = probability - target")
        logger.debug(
            "  score gradients = %s",
            {k: round(v, 6) for k, v in d_scores.items()},
        )

    last_hidden = activations[-1]
    d_last_hidden = [0.0] * len(last_hidden)

    for word in vocab:
        neuron = final_layer[word]
        d_score = d_scores[word]
        if trace:
            logger.debug(
                "\n  Final neuron '%s': d_score = %.6f", word, d_score
            )
        for i in range(len(neuron["weights"])):
            old_weight = neuron["weights"][i]
            grad_w = d_score * last_hidden[i]
            contribution_back = d_score * old_weight
            d_last_hidden[i] += contribution_back
            neuron["weights"][i] -= lr * grad_w
            if trace:
                logger.debug(
                    f"    weight[{i}]: gradient = d_score({d_score:.6f}) "
                    f"* input({last_hidden[i]:.6f}) = {grad_w:.6f}"
                )
                logger.debug(
                    f"      update: {old_weight:.6f} - {lr} * "
                    f"{grad_w:.6f} = {neuron['weights'][i]:.6f}"
                )
                logger.debug(
                    f"      error sent backward += d_score * old_weight "
                    f"      d_score : {d_score:.6f} "
                    f"      old_weight : {old_weight:.6f}"
                    f"= {contribution_back:.6f}"
                )
                logger.debug(
                    f"      d_last_hidden[i] += contribution_back : "
                    f"{d_last_hidden[i]:.6f} "
                )
                logger.debug(
                    f"      d_last_hidden[i] {i}= "
                    f"{d_last_hidden[i]:.6f} "
                )
        old_bias = neuron["bias"]
        neuron["bias"] -= lr * d_score
        if trace:
            logger.debug(
                f"    bias update: {old_bias:.6f} - {lr} * "
                f"{d_score:.6f} = {neuron['bias']:.6f}"
            )

    # Send each layer its share of the output error, walking right-to-left.
    d_h = d_last_hidden
    for l in range(len(hidden_network) - 1, -1, -1):
        layer = hidden_network[l]
        z_layer = zs[l]
        prev_activation = activations[l]
        if trace:
            logger.debug(
                f"\n  [BACKWARD PASS: HIDDEN LAYER {l + 1}]"
            )

        d_prev = [0.0] * len(prev_activation)
        for i, neuron in enumerate(layer):
            activation_slope = leaky_relu_deriv(z_layer[i])
            d_z = d_h[i] * activation_slope
            if trace:
                logger.debug(f"\n    neuron {i}:")
                logger.debug(f"\n    z_layer {i} = {z_layer[i]}:")
                logger.debug(
                    f"\n    activation_slope = {activation_slope:.2f}:"
                )
                logger.debug(f"\n    d_h {i} = {d_h[i]}:")
                logger.debug(
                    f"      d_z = incoming error({d_h[i]:.6f}) "
                    f"* LeakyReLU derivative at z={z_layer[i]:.6f} "
                    f"({activation_slope:.2f}) = {d_z:.6f}"
                )
            for k in range(len(neuron["weights"])):
                old_weight = neuron["weights"][k]
                grad_w = d_z * prev_activation[k]
                contribution_back = d_z * old_weight
                d_prev[k] += contribution_back
                neuron["weights"][k] -= lr * grad_w
                if trace:
                    logger.debug(
                        f"      weight[{k}] gradient = d_z({d_z:.6f}) "
                        f"* input({prev_activation[k]:.6f}) = {grad_w:.6f}"
                    )
                    logger.debug(
                        f"        update: {old_weight:.6f} - {lr} * "
                        f"{grad_w:.6f} = {neuron['weights'][k]:.6f}"
                    )
            old_bias = neuron["bias"]
            neuron["bias"] -= lr * d_z
            if trace:
                logger.debug(
                    f"      bias update: {old_bias:.6f} - {lr} * "
                    f"{d_z:.6f} = {neuron['bias']:.6f}"
                )
        d_h = d_prev

    if trace:
        logger.debug("\n  [END OF TRACED TRAINING STEP]")
        logger.debug(
            "  All weights and biases have now moved slightly toward a "
            "lower loss.\n"
        )
    return loss
