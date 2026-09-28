import random
import re
from collections import Counter, defaultdict


def simple_tokenizer(text, frequency_threshold=None):
    """Simple tokenizer that splits text into words."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not frequency_threshold:
        return tokens
    word_counts = Counter(tokens)
    return [token for token in tokens if word_counts[token] >= frequency_threshold]


class BigramModel:
    def __init__(self, corpus, frequency_threshold=None):
        """Compute bigram probabilities from a list of texts (from analyze_bigrams)."""
        text = " ".join(corpus)
        words = simple_tokenizer(text, frequency_threshold)
        bigrams = list(zip(words[:-1], words[1:]))

        bigram_counts = Counter(bigrams)
        unigram_counts = Counter(words)

        self.vocab = list(unigram_counts.keys())
        self.bigram_probs = defaultdict(dict)
        for (word1, word2), count in bigram_counts.items():
            self.bigram_probs[word1][word2] = count / unigram_counts[word1]

    def generate_text(self, start_word, length=20):
        """Generate text based on bigram probabilities (from generate_text)."""
        current_word = start_word.lower()
        generated_words = [current_word]

        for _ in range(length - 1):
            next_words = self.bigram_probs.get(current_word)
            if not next_words:  # no known next word, so stop
                break
            next_word = random.choices(
                list(next_words.keys()), weights=list(next_words.values())
            )[0]
            generated_words.append(next_word)
            current_word = next_word

        return " ".join(generated_words)