from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class Preprocessor:
    def __init__(self):
        self.tokenizer = None
        self.stop_word_remover = None
        self.lemmatizer = None
        self.lower_caser = None
        self.stop_word_file_path = "./preprocess/id_stopword.txt"

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        return self

    def set_stop_word_remover(self, stop_word_remover):
        self.stop_word_remover = stop_word_remover
        return self

    def set_lemmatizer(self, lemmatizer):
        self.lemmatizer = lemmatizer
        return self

    def set_lower_caser(self, lower_caser):
        self.lower_caser = lower_caser
        return self

    def _apply_stop_words_from_file(self, tokens):
        with open(self.stop_word_file_path, "r") as f:
            stop_words = set(f.read().splitlines())
        tokens = [token for token in tokens if token.lower() not in stop_words]
        return tokens

    def preprocess(self, text):
        # basic preprocessing
        text = text.lower()
        text = text.replace(".", " <PERIOD> ")
        text = text.replace(",", " <COMMA> ")
        text = text.replace('"', " <QUOTATION_MARK> ")
        text = text.replace(";", " <SEMICOLON> ")
        text = text.replace("!", " <EXCLAMATION_MARK> ")
        text = text.replace("?", " <QUESTION_MARK> ")
        text = text.replace("(", " <LEFT_PAREN> ")
        text = text.replace(")", " <RIGHT_PAREN> ")
        text = text.replace("--", " <HYPHENS> ")
        text = text.replace("?", " <QUESTION_MARK> ")
        text = text.replace("\n", " <NEW_LINE> ")
        text = text.replace(":", " <COLON> ")

        tokens = self.tokenizer.tokenize(text)

        tokens = self.stop_word_remover.remove_stop_words(tokens)

        # Lemmatization process not neccesary in word2vec model
        # https://www.nature.com/articles/s41598-021-01460-7
        # https://arxiv.org/abs/1411.2738
        # https://arxiv.org/abs/1301.3781
        # tokens = self.lemmatizer.lemmatize_words(tokens)

        tokens = self.lower_caser.lower_case_words(tokens)

        # need to lower case first
        # tokens = self._apply_stop_words_from_file(tokens=tokens)

        # Remove all words with  5 or fewer occurences
        word_counts = Counter(tokens)
        trimmed_words = [word for word in tokens if word_counts[word] > 0]

        return trimmed_words


class Tokenizer:
    def tokenize(self, text):
        return word_tokenize(text)


class StopWordRemover:
    def __init__(self):
        self.stop_words = set(stopwords.words("indonesian"))

    def remove_stop_words(self, tokens):
        return [word for word in tokens if word.lower() not in self.stop_words]


class Lemmatizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize_words(self, tokens):
        return [self.lemmatizer.lemmatize(word) for word in tokens]


class LowerCaser:
    def lower_case_words(self, tokens):
        return [word.lower() for word in tokens]
