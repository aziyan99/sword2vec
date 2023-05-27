import logging

import numpy as np
from .preprocessor import (
    Preprocessor,
    Tokenizer,
    StopWordRemover,
    LowerCaser,
)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

import datetime
import time
import gzip
import gc
import pickle
from . import helpers

np.random.seed(1)


class SkipGramWord2Vec:
    def __init__(self, window_size=2, learning_rate=0.01, embedding_dim=100, epochs=10):
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.w1 = None
        self.w2 = None
        self.history = {}
        self.matrix = []
        self.vocabulary = []
        self.preprocessor = (
            Preprocessor()
            .set_tokenizer(Tokenizer())
            .set_stop_word_remover(StopWordRemover())
            .set_lower_caser(LowerCaser())
        )

    def preprocess_vocab(self, lines):
        logging.info("Preprocessing vocabularies")
        for title in lines:
            tokens = self.preprocessor.preprocess(title)
            for word in tokens:
                self.vocabulary.append(word)

    def build_vocab(self, text_array):
        self.word_to_idx = {}
        self.idx_to_word = {}
        logging.info("Building vocabularies")
        for idx, word in enumerate(text_array):
            if word not in self.word_to_idx:
                self.word_to_idx[word] = len(self.word_to_idx)
                self.idx_to_word[len(self.idx_to_word)] = word
        self.vocabulary = list(self.word_to_idx.keys())

    def one_hot_encoding(self, text_array):
        logging.info("Building matrix")
        for idx, word in enumerate(text_array):
            center_vec = [0] * len(self.word_to_idx)
            center_vec[self.word_to_idx[word]] = 1

            context_vec = []
            for i in range(-self.window_size, self.window_size + 1):
                if (
                    i == 0
                    or idx + i < 0
                    or idx + i >= len(text_array)
                    or word == text_array[idx + i]
                ):
                    continue

                temp = [0] * len(self.word_to_idx)
                temp[self.word_to_idx[text_array[idx + i]]] = 1
                context_vec.append(temp)

            self.matrix.append([center_vec, context_vec])

    def one_hot_encoding_streamable(self, text_array):
        for idx, word in enumerate(text_array):
            center_vec = np.zeros(len(self.word_to_idx))
            center_vec[self.word_to_idx[word]] = 1

            context_idxs = np.arange(idx - self.window_size, idx + self.window_size + 1)
            context_idxs = context_idxs[
                (context_idxs != idx)
                & (context_idxs >= 0)
                & (context_idxs < len(text_array))
            ]
            context_vec = np.zeros((len(context_idxs), len(self.word_to_idx)))
            context_vec[
                np.arange(len(context_idxs)),
                [self.word_to_idx[text_array[i]] for i in context_idxs],
            ] = 1

            yield center_vec, context_vec

    def train(self, lines):
        print("\n|------------------------------------")
        print("| Preprocess")
        print("|---------------------------------")
        print("|")
        print("|\n")

        self.preprocess_vocab(lines)
        self.build_vocab(self.vocabulary)

        print("\n|------------------------------------")
        print("| Summary")
        print("|---------------------------------")
        print("| Total line: " + str(len(lines)))
        print("| Total vocabulary: " + str(len(self.vocabulary)))
        print("|")
        print("|\n")

        self.w1 = np.round(
            np.random.uniform(-1, 1, (len(self.word_to_idx), self.embedding_dim)), 2
        )
        self.w2 = np.round(
            np.random.uniform(-1, 1, (self.embedding_dim, len(self.word_to_idx))), 2
        )

        logging.info("Training")

        try:
            for e in range(self.epochs):
                total_loss = 0
                start_time = time.time()

                logging.info(f"Epoch {e + 1} start")

                for center_vec, context_vec in self.one_hot_encoding_streamable(
                    self.vocabulary
                ):
                    word = [center_vec, context_vec]
                    out, h, u = helpers.forward_pass(word[0], self.w1, self.w2)
                    dw1, dw2 = helpers.backpropagation(out, word, h, self.w2)
                    self.w1, self.w2 = helpers.learning(
                        self.learning_rate, self.w1, self.w2, dw1, dw2
                    )
                    total_loss += helpers.loss(word[1], u)

                self.history[e] = total_loss

                logging.info(
                    f"Epoch {e + 1} Execution Time: {time.time() - start_time} seconds"
                )

        except KeyboardInterrupt:
            logging.info(f"\nTraining interrupted by user.")
            return

    # not calculating cosine similarity it just perform regular predict through forward_pass()
    def predict(self, word, topn=2):
        word = word.lower()
        center_vec = [0] * len(self.word_to_idx)
        center_vec[self.word_to_idx[word]] = 1

        out, _, _ = helpers.forward_pass(center_vec, self.w1, self.w2)
        most_likely_idxs = np.array(out).argsort()[-topn:][::-1]
        return [self.idx_to_word[w] for w in most_likely_idxs]

    def search_similar_words(self, word, topn=10):
        try:
            word_embedding = self.w1[self.word_to_idx[word]]
            word_embeddings = self.w1

            similarities = []
            for embedding in word_embeddings:
                similarity = helpers.cosine_similarity(word_embedding, embedding)
                similarities.append(similarity)

            similarities = np.array(similarities)

            # Get the indices of the most similar words (excluding the search word)
            similar_word_indices = similarities.argsort()[-(topn + 1) : -1][::-1]

            # Create a dictionary of {word: score} pairs
            similar_words = {}
            for idx in similar_word_indices:
                similar_words[self.idx_to_word[idx]] = similarities[idx]

            return similar_words
        except KeyError:
            logging.warning(f"{word}'s unknown.")

    def save_model(self, filename=None):
        logging.info("Saving model")
        if filename is None:
            current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
            filename = f"{current_time}.pkl"

        model_data = {
            "window_size": self.window_size,
            "learning_rate": self.learning_rate,
            "embedding_dim": self.embedding_dim,
            "epochs": self.epochs,
            "word_to_idx": self.word_to_idx,
            "idx_to_word": self.idx_to_word,
            "w1": self.w1,
            "w2": self.w2,
            "history": self.history,
            "matrix": self.matrix,
            "vocabulary": self.vocabulary,
        }

        with open(filename, "wb") as file:
            pickle.dump(model_data, file)

    def save_compressed_model(self, filename=None):
        logging.info("Saving compressed model")
        gc.disable()
        if filename is None:
            current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
            filename = f"{current_time}.pkl.gz"

        model_data = {
            "window_size": self.window_size,
            "learning_rate": self.learning_rate,
            "embedding_dim": self.embedding_dim,
            "epochs": self.epochs,
            "word_to_idx": self.word_to_idx,
            "idx_to_word": self.idx_to_word,
            "w1": self.w1,
            "w2": self.w2,
            "history": self.history,
            "matrix": self.matrix,
            "vocabulary": self.vocabulary,
        }

        with gzip.open(filename, "wb") as file:
            pickle.dump(model_data, file, protocol=-1)

    @staticmethod
    def load_model(filename):
        try:
            logging.info("Load model")
            with open(filename, "rb") as file:
                model_data = pickle.load(file)

            model = SkipGramWord2Vec(
                window_size=model_data["window_size"],
                learning_rate=model_data["learning_rate"],
                embedding_dim=model_data["embedding_dim"],
                epochs=model_data["epochs"],
            )
            model.word_to_idx = model_data["word_to_idx"]
            model.idx_to_word = model_data["idx_to_word"]
            model.w1 = model_data["w1"]
            model.w2 = model_data["w2"]
            model.history = model_data["history"]
            model.matrix = model_data["matrix"]
            model.vocabulary = model_data["vocabulary"]

            return model
        except FileNotFoundError:
            logging.warning(f"{filename}'s unknown.")

    @staticmethod
    def load_compressed_model(filename):
        try:
            logging.info("Load compressed model")
            with gzip.open(filename, "rb") as file:
                model_data = pickle.load(file)

            model = SkipGramWord2Vec(
                window_size=model_data["window_size"],
                learning_rate=model_data["learning_rate"],
                embedding_dim=model_data["embedding_dim"],
                epochs=model_data["epochs"],
            )
            model.word_to_idx = model_data["word_to_idx"]
            model.idx_to_word = model_data["idx_to_word"]
            model.w1 = model_data["w1"]
            model.w2 = model_data["w2"]
            model.history = model_data["history"]
            model.matrix = model_data["matrix"]
            model.vocabulary = model_data["vocabulary"]

            return model
        except FileNotFoundError:
            logging.warning(f"{filename}'s unknown.")
