import random

def create_embedding_table(vocab, embedding_dim):
    # one random vector per word -- this is a lookup table, nothing more
    return {word: [random.uniform(-1, 1) for _ in range(embedding_dim)] for word in vocab}

def embed(word, embedding_table):
    return embedding_table[word]

def embed_sentence(sentence, embedding_table):
    words = sentence.lower().split()
    return [embed(w, embedding_table) for w in words]

if __name__ == "__main__":
    VOCAB = ["the", "sky", "is", "blue", "green", "cat", "running", "happy"]
    EMBEDDING_DIM = 5

    table = create_embedding_table(VOCAB, EMBEDDING_DIM)

    print("Embedding for 'sky':", [round(x, 2) for x in embed("sky", table)])

    sentence_embeddings = embed_sentence("the sky is", table)
    for word, vec in zip("the sky is".split(), sentence_embeddings):
        print(f"{word:6s} -> {[round(x,2) for x in vec]}")