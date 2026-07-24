import random
import math
import logging

# DEBUG shows the detailed forward/backward calculations.
# INFO shows only the normal training summaries.
LOG_LEVEL = logging.DEBUG

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(levelname)-5s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------- leaky relu + derivative ----------

def leaky_relu(x):
    return x if x > 0 else 0.01 * x

def leaky_relu_deriv(x):
    return 1.0 if x > 0 else 0.01

# ---------- network as plain lists of {weights, bias} dicts ----------

def create_neuron(num_inputs):
    return {
        "weights": [random.uniform(-1, 1) for _ in range(num_inputs)],
        "bias": random.uniform(-1, 1),
    }

def create_layer(num_neurons, num_inputs):
    return [create_neuron(num_inputs) for _ in range(num_neurons)]

def create_network(num_layers, neurons_per_layer, num_inputs):
    layers = []
    n_in = num_inputs
    for _ in range(num_layers):
        layers.append(create_layer(neurons_per_layer, n_in))
        n_in = neurons_per_layer
    return layers

def create_final_layer(vocab, num_inputs):
    return {word: create_neuron(num_inputs) for word in vocab}

def softmax(scores: dict):
    values = list(scores.values())
    max_val = max(values)
    exps = {k: math.exp(v - max_val) for k, v in scores.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}

# ---------- forward pass, but SAVE everything backprop will need ----------
# For every layer we keep: the inputs it received, the raw z (pre-activation)
# per neuron, and the activated output per neuron.

def forward_full(inputs, hidden_network, final_layer, vocab, trace=False):
    activations = [inputs]   # activations[0] = network input
    zs = []                  # zs[l] = pre-activation values of layer l

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
            # Each product says how much one input contributes to this neuron.
            products = [i * w for i, w in zip(current, neuron["weights"])]
            # z is the raw value before Leaky ReLU changes it.
            z = sum(products) + neuron["bias"]
            # h is the value that this neuron sends to the next layer.
            h = leaky_relu(z)
            z_layer.append(z)
            h_layer.append(h)
            if trace:
                logger.debug("    neuron %d:", neuron_index)
                logger.debug("      inputs  = %s", [round(v, 6) for v in current])
                logger.debug("      weights = %s", [round(v, 6) for v in neuron["weights"]])
                logger.debug("      products(input * weight) = %s", [round(v, 6) for v in products])
                logger.debug("      z = sum(products) + bias(%.6f) = %.6f", neuron["bias"], z)
                logger.debug("      activation = LeakyReLU(z) = %.6f", h)
        zs.append(z_layer)
        activations.append(h_layer)
        current = h_layer

    # final layer -- no activation, raw scores straight into softmax
    final_z = {}
    for word in vocab:
        neuron = final_layer[word]
        products = [i * w for i, w in zip(current, neuron["weights"])]
        final_z[word] = sum(products) + neuron["bias"]
        if trace:
            logger.debug("\n  Final neuron '%s':", word)
            logger.debug("    inputs  = %s", [round(v, 6) for v in current])
            logger.debug("    weights = %s", [round(v, 6) for v in neuron["weights"]])
            logger.debug("    products = %s", [round(v, 6) for v in products])
            logger.debug(
                "    raw score = sum(products) + bias(%.6f) = %.6f",
                neuron["bias"], final_z[word],
            )

    probs = softmax(final_z)
    if trace:
        logger.debug("\n  Softmax converts raw scores into probabilities:")
        logger.debug("    raw scores    = %s", {k: round(v, 6) for k, v in final_z.items()})
        logger.debug("    probabilities = %s", {k: round(v, 6) for k, v in probs.items()})
    return activations, zs, probs

# ---------- backward pass: real analytic backprop ----------
# Key formula (this is the one bit of calculus, already worked out for you):
# for softmax + "negative log likelihood of the correct word" loss combined,
# the gradient w.r.t. the final raw scores is simply:
#     dL/dscore[word] = prob[word] - 1_if_word_is_correct_else_0
# This is a well-known simplification -- softmax and this loss were basically
# designed to cancel out into that clean form.

def backward(inputs, correct_word, hidden_network, final_layer, vocab, lr, trace=False):
    activations, zs, probs = forward_full(
        inputs, hidden_network, final_layer, vocab, trace=trace
    )
    loss = -math.log(max(probs[correct_word], 1e-9))

    if trace:
        logger.debug("\n  [LOSS]")
        logger.debug("  Correct word = '%s'", correct_word)
        logger.debug(
            "  loss = -log(P(correct word)) = -log(%.6f) = %.6f",
            probs[correct_word], loss,
        )
        logger.debug("  Learning rate (chosen by us) = %s", lr)

    # --- gradient at the final layer ---
    # For the correct word, subtract 1. For every other word, subtract 0.
    # This gradient tells us how each raw score should change to reduce loss.
    d_scores = {w: probs[w] - (1.0 if w == correct_word else 0.0) for w in vocab}

    if trace:
        logger.debug("\n  [BACKWARD PASS: FINAL LAYER]")
        logger.debug("  score gradient = probability - target")
        logger.debug("  score gradients = %s", {k: round(v, 6) for k, v in d_scores.items()})

    last_hidden = activations[-1]  # output of the last hidden layer

    # d_loss/d_(last hidden output), accumulated across all final-layer neurons
    d_last_hidden = [0.0] * len(last_hidden)

    for word in vocab:
        neuron = final_layer[word]
        d_score = d_scores[word]
        if trace:
            logger.debug("\n  Final neuron '%s': d_score = %.6f", word, d_score)
        for i in range(len(neuron["weights"])):
            old_weight = neuron["weights"][i]
            # A weight's gradient is the error signal multiplied by the value
            # that travelled through that weight during the forward pass.
            grad_w = d_score * last_hidden[i]
            contribution_back = d_score * old_weight
            d_last_hidden[i] += contribution_back  # use old weight before changing it
            neuron["weights"][i] -= lr * grad_w
            if trace:
                logger.debug(
                    f"    weight[{i}]: gradient = d_score({d_score:.6f}) "
                    f"* input({last_hidden[i]:.6f}) = {grad_w:.6f}"
                )
                logger.debug(
                    f"      update: {old_weight:.6f} - {lr} * {grad_w:.6f} "
                    f"= {neuron['weights'][i]:.6f}"
                )
                logger.debug(
                    f"      error sent backward += d_score * old_weight "
                    f"      d_score : {d_score:.6f} " 
                    f"      old_weight : {old_weight:.6f}" 
                    f"= {contribution_back:.6f}"
                )
                logger.debug(
                    f"      d_last_hidden[i] += contribution_back : {d_last_hidden[i]:.6f} "
                )
                logger.debug(
                    f"      d_last_hidden[i] {i}= {d_last_hidden[i]:.6f} "
                )
        old_bias = neuron["bias"]
        # Bias acts like a weight whose input is always 1, so grad_bias=d_score.
        neuron["bias"] -= lr * d_score
        if trace:
            logger.debug(
                f"    bias update: {old_bias:.6f} - {lr} * {d_score:.6f} "
                f"= {neuron['bias']:.6f}"
            )

    # --- gradient through hidden layers, walking backward ---
    d_h = d_last_hidden  # gradient w.r.t. current layer's OUTPUT

    for l in range(len(hidden_network) - 1, -1, -1):
        layer = hidden_network[l]
        z_layer = zs[l]
        prev_activation = activations[l]  # this layer's INPUT

        if trace:
            logger.debug(f"\n  [BACKWARD PASS: HIDDEN LAYER {l + 1}]")

        d_prev = [0.0] * len(prev_activation)

        for i, neuron in enumerate(layer):
            activation_slope = leaky_relu_deriv(z_layer[i])
            # Chain rule: combine the error arriving from the right with the
            # slope of this neuron's activation at its saved raw z value.
            d_z = d_h[i] * activation_slope
            if trace:
                logger.debug(f"\n    neuron {i}:")
                logger.debug(f"\n    z_layer {i} = {z_layer[i]}:")
                logger.debug(f"\n    activation_slope = {activation_slope:.2f}:")
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
                        f"        update: {old_weight:.6f} - {lr} * {grad_w:.6f} "
                        f"= {neuron['weights'][k]:.6f}"
                    )
            old_bias = neuron["bias"]
            neuron["bias"] -= lr * d_z
            if trace:
                logger.debug(
                    f"      bias update: {old_bias:.6f} - {lr} * {d_z:.6f} "
                    f"= {neuron['bias']:.6f}"
                )

        d_h = d_prev  # this becomes the gradient signal for the layer before it

    if trace:
        logger.debug("\n  [END OF TRACED TRAINING STEP]")
        logger.debug("  All weights and biases have now moved slightly toward a lower loss.\n")

    return loss

def forward(inputs, hidden_network, final_layer, vocab):
    _, _, probs = forward_full(inputs, hidden_network, final_layer, vocab)
    return probs

# ---------- usage ----------

if __name__ == "__main__":
    VOCAB = ["blue", "green", "cat", "running", "happy"]

    training_data = [
        ([0.5, -0.3, 0.8, 0.1, -0.6], "blue"),
        ([0.2, 0.4, -0.1, 0.6, 0.3], "green"),
        ([-0.4, 0.1, 0.5, -0.2, 0.7], "happy"),
    ]

    hidden_network = create_network(num_layers=2, neurons_per_layer=5, num_inputs=5)
    final_layer = create_final_layer(VOCAB, num_inputs=5)

    logger.info("--- Before training ---")
    for inputs, correct in training_data:
        probs = forward(inputs, hidden_network, final_layer, VOCAB)
        predicted = max(probs, key=probs.get)
        logger.info(f"  target={correct:8s} predicted={predicted:8s} probs={ {k: round(v,2) for k,v in probs.items()} }")

    EPOCHS = 300
    # Detailed tracing produces many lines. By default, trace only the first
    # example of the first epoch. Set this to False to disable the teaching log.
    DETAILED_TRACE = True

    for epoch in range(EPOCHS):
        total_loss = 0
        for example_index, (inputs, correct) in enumerate(training_data):
            # Trace one step so we can inspect every calculation without
            # printing the same details thousands of times.
            trace_this_step = DETAILED_TRACE and epoch == 0 and example_index in  [0,1]
            total_loss += backward(
                inputs,
                correct,
                hidden_network,
                final_layer,
                VOCAB,
                lr=0.3,
                trace=trace_this_step,
            )
        if epoch % 30 == 0:
            logger.info(f"Epoch {epoch:3d}  avg loss={total_loss/len(training_data):.4f}")

    logger.info("\n--- After training ---")
    for inputs, correct in training_data:
        probs = forward(inputs, hidden_network, final_layer, VOCAB)
        predicted = max(probs, key=probs.get)
        logger.info(f"  target={correct:8s} predicted={predicted:8s} probs={ {k: round(v,2) for k,v in probs.items()} }")

    logger.info("\n--- Custom Run ---")
    custom_inputs = [-0.4, 0.1, 0.5, -0.2, 0.7]
    probs = forward(custom_inputs, hidden_network, final_layer, VOCAB)
    logger.info(f"  custom inputs={custom_inputs}")
    logger.info(f"  probs={ {k: round(v,2) for k,v in probs.items()} }")
