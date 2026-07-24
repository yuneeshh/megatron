import random
import math

# ---------- building blocks (same as before) ----------

def relu(x):
    return max(0, x)

def create_neuron(num_inputs):
    weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
    bias = random.uniform(-1, 1)
    return weights, bias

def neuron_forward(inputs, weights, bias, activation=relu):
    total = sum(i * w for i, w in zip(inputs, weights)) + bias
    return activation(total) if activation else total

def create_layer(num_neurons, num_inputs):
    return [create_neuron(num_inputs) for _ in range(num_neurons)]

def layer_forward(inputs, layer, activation=relu):
    return [neuron_forward(inputs, w, b, activation) for w, b in layer]

def create_network(num_layers, neurons_per_layer, num_inputs):
    layers = []
    inputs_for_this_layer = num_inputs
    for _ in range(num_layers):
        layers.append(create_layer(neurons_per_layer, inputs_for_this_layer))
        inputs_for_this_layer = neurons_per_layer
    return layers

def network_forward(inputs, layers):
    current = inputs
    for layer in layers:
        current = layer_forward(current, layer)
    return current

# ---------- new: final layer + softmax ----------

def create_final_layer(vocab, num_inputs):
    # one neuron per vocab word, each with its own random weights/bias
    return {word: create_neuron(num_inputs) for word in vocab}

def final_layer_forward(inputs, final_layer):
    # NOTE: no ReLU here — raw scores go straight to softmax
    scores = {}
    for word, (w, b) in final_layer.items():
        scores[word] = neuron_forward(inputs, w, b, activation=None)
    return scores

def softmax(scores: dict):
    values = list(scores.values())
    max_val = max(values)                       # for numerical stability
    exps = {k: math.exp(v - max_val) for k, v in scores.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}

# ---------- usage: full forward pass ----------

if __name__ == "__main__":
    VOCAB = ["blue", "green", "cat", "running", "happy"]

    inputs = [0.5, -0.3, 0.8, 0.1, -0.6]  # "the sky is" -> 5 floats (pretend embedding)

    hidden_network = create_network(num_layers=5, neurons_per_layer=5, num_inputs=5)
    final_layer = create_final_layer(VOCAB, num_inputs=5)

    hidden_output = network_forward(inputs, hidden_network)
    raw_scores = final_layer_forward(hidden_output, final_layer)
    probabilities = softmax(raw_scores)

    predicted_word = max(probabilities, key=probabilities.get)

    print("Input floats:      ", [round(x, 2) for x in inputs])
    print("Last hidden output:", [round(x, 2) for x in hidden_output])
    print("Raw scores:        ", {k: round(v, 2) for k, v in raw_scores.items()})
    print("Probabilities:     ", {k: round(v, 4) for k, v in probabilities.items()})
    print("Predicted word:    ", predicted_word)