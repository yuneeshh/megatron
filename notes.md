# Megatron — From-Scratch Neural Network Notes

A neural network / tiny language model built entirely in plain Python — no
ML frameworks (no PyTorch/TensorFlow) — to build an exact, ground-truth
mental model of how GenAI/LLMs work internally, the same way one might trace
a web server or program end to end.

---

## Concepts covered so far

1. **Neuron** — smallest unit: `output = activation(sum(inputs × weights) + bias)`.
   No branching logic; just multiply-add, then a small nonlinear "squash."
2. **Layer** — many neurons in parallel, each with independent weights, all
   receiving the same input list, producing one output per neuron.
3. **Network** — layers chained: each layer's output list becomes the next
   layer's input list.
4. **Final layer / vocabulary scoring** — one dedicated neuron per possible
   output word (real models: ~30k–200k neurons here). Each independently
   scores "how likely am I the next word" against the same shared hidden
   output.
5. **Softmax** — converts raw final-layer scores into probabilities summing to 1.
6. **Prediction** — pick the highest-probability word (greedy), or sample
   (adds variety — ties to "temperature").
7. **Training labels** — no manual labeling for language models. Real
   sentences already contain the answer: input = sentence prefix, label =
   the next word that actually appeared (self-supervised learning).
8. **Loss** — `-log(probability assigned to the correct word)`
   (negative log likelihood).
9. **Backpropagation** — computing, for every weight in every layer, "which
   direction (and how much) reduces the loss," walking backward from output
   to input via the chain rule.
10. **Gradient descent** — nudging every weight in the computed direction,
    repeated across many examples/epochs.
11. **Inference vs. training** — weights are frozen after training; a query
    never changes them, it only supplies input floats through an already-tuned,
    fixed function.
12. **Embeddings** — converting words into floats via a lookup table (one
    random vector per vocab word). The vectors are themselves weights, tuned
    by the same backprop process — not yet wired into training in this repo.

### Not yet covered
- Attention / Transformer architecture (this network is plain feedforward,
  not attention-based — real LLMs use self-attention to weigh which earlier
  words matter most).
- Subword tokenization (this repo uses whole-word vocab for simplicity).
- Batching best practices, learning rate scheduling, optimizers (Adam, etc.)
  beyond plain gradient descent.
- Multi-token generation loop (feeding a predicted word back in and repeating).

---

## Repo structure

| File | Responsibility |
|---|---|
| `main.py` | Entry point. Builds vocab + network, runs before/after prediction logging, calls training. Run with `python main.py`. |
| `neuron.py` | Single neuron creation, Leaky ReLU activation + its derivative. |
| `layer.py` | Builds a layer of neurons (all sharing the same inputs). |
| `network.py` | Chains layers into a full network; `forward_full` runs inference and saves everything backprop needs (activations, pre-activation `z` values); includes optional detailed trace logging. |
| `final_layer.py` | One scoring neuron per vocab word + softmax. |
| `loss.py` | Negative log likelihood loss. |
| `backprop.py` | Real analytic backpropagation — the closed-form gradient `probability − correct_answer_indicator` at the output, then chain-rule propagation backward through hidden layers. Optional detailed trace logging of every weight update. |
| `train.py` | Training loop — runs `backward()` once per example per epoch, preserving original example order (no shuffling/batching). |
| `log_config.py` | Central logging setup (level, format) used by all modules. |
| `embed.py` | Word → vector lookup table. **Not yet wired into training** — `main.py` still uses hand-picked fake input vectors; `embed.py` is imported as an integration point only. |

---

## Key lessons learned (worth keeping — real debugging, not obvious in hindsight)

1. **Dead ReLU collapse**: plain ReLU (`max(0, x)`) permanently zeroes a
   neuron once it lands in negative territory (zero gradient there = it can
   never recover). Fixed with **Leaky ReLU** (`x if x > 0 else 0.01 * x`),
   which keeps a small gradient alive on the negative side.

2. **Finite-difference gradients are unreliable near saturation**: an
   earlier version approximated gradients by nudging each weight by a tiny
   epsilon and measuring the loss change. It worked for single simple
   examples but had a real, measured failure mode — once softmax saturated
   toward 0/1 for the *wrong* word, the loss surface went locally flat there,
   so the epsilon-sized nudge couldn't detect a real gradient, and training
   got permanently stuck confidently predicting the wrong word. Measured
   success rate across learning rates: ~70–90%, not reliable. Symptom: avg
   loss locks onto a constant value (e.g. `20.7233 = -log(1e-9)`, the
   clamp floor) and never moves again.

3. **Fix: real analytic backpropagation** (current `backprop.py`) — using
   the exact calculus-derived gradient instead of an approximation.
   Verified **20/20**, then **15/15** independent runs converge correctly,
   smooth monotonic loss decrease, and cheaper to compute (one backward pass
   vs. thousands of extra forward passes for finite differences).

4. **Sequential vs. batch updates**: updating weights immediately after each
   individual example (rather than averaging loss/gradient across all
   examples first) can cause examples to "fight" each other with unstable
   gradient approximations — this mattered a lot for finite-difference
   training, less so once real backprop was used (sequential per-example
   updates work fine with real backprop, as verified).

---

## Suggested next steps
1. Wire `embed.py`'s `embed_sentence()` into `main.py` so training actually
   uses real (if untrained) embedding vectors instead of hand-picked ones.
2. Implement a basic self-attention layer — the real architectural
   difference between this project and actual LLMs.
3. Expand vocab size and training set size — watch whether sequential vs.
   batch update behavior changes at scale.
4. Optional: multi-token generation loop (autoregressive decoding — feed a
   predicted word back in as the next input, repeat until a stop condition).