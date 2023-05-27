from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class Preprocessor:
    """
    A class responsible for preprocessing text data.

    The Preprocessor class performs various preprocessing steps such as tokenization, stop word removal,
    lemmatization, and lowercasing.

    Parameters:
    - stopword_file: An path to list of stopwords in txt file
    - tokenizer: An instance of a tokenizer class that implements a 'tokenize' method.
    - stop_word_remover: An instance of a stop word remover class that implements a 'remove_stop_words' method.
    - lemmatizer: An instance of a lemmatizer class that implements a 'lemmatize_words' method.
    - lower_caser: An instance of a lowercaser class that implements a 'lower_case_words' method.
    """

    def __init__(self, stopword_file=None):
        self.tokenizer = None
        self.stop_word_remover = None
        self.lemmatizer = None
        self.lower_caser = None
        self.stopword_file = stopword_file

    def set_tokenizer(self, tokenizer):
        """
        Sets the tokenizer for the preprocessor.

        Parameters:
        - tokenizer: An instance of a tokenizer class.
        """
        self.tokenizer = tokenizer
        return self

    def set_stop_word_remover(self, stop_word_remover):
        """
        Sets the stop word remover for the preprocessor.

        Parameters:
        - stop_word_remover: An instance of a stop word remover class.
        """
        self.stop_word_remover = stop_word_remover
        return self

    def set_lemmatizer(self, lemmatizer):
        """
        Sets the lemmatizer for the preprocessor.

        Parameters:
        - lemmatizer: An instance of a lemmatizer class.
        """
        self.lemmatizer = lemmatizer
        return self

    def set_lower_caser(self, lower_caser):
        """
        Sets the lowercaser for the preprocessor.

        Parameters:
        - lower_caser: An instance of a lowercaser class.
        """
        self.lower_caser = lower_caser
        return self

    def _apply_stop_words_from_file(self, tokens):
        """
        Applies stop words removal using a file-based stop word list.

        Parameters:
        - tokens: A list of tokens to remove stop words from.

        Returns:
        - tokens: A list of tokens with stop words removed.
        """
        tokens = []
        if self.stopword_file != None:
            with open(self.stopword_file, "r") as f:
                stop_words = set(f.read().splitlines())
            tokens = [token for token in tokens if token.lower() not in stop_words]
        return tokens

    def preprocess(self, text):
        """
        Preprocesses the given text.

        Parameters:
        - text: The input text to preprocess.

        Returns:
        - trimmed_words: A list of preprocessed words.
        """

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
    """
    A class responsible for tokenizing text.

    The Tokenizer class uses the NLTK library to tokenize text.

    """

    def tokenize(self, text):
        return word_tokenize(text)


class StopWordRemover:
    """
    Tokenizes the given text.

    Parameters:
    - text: The input text to tokenize.

    Returns:
    - tokens: A list of tokens.
    """

    def __init__(self):
        self.stop_words = set(stopwords.words("indonesian"))

    def remove_stop_words(self, tokens):
        """
        Removes stop words from the given list of tokens.

        Parameters:
        - tokens: A list of tokens.

        Returns:
        - tokens: A list of tokens with stop words removed.
        """
        return [word for word in tokens if word.lower() not in self.stop_words]


class Lemmatizer:
    """
    A class responsible for lemmatizing words.

    The Lemmatizer class uses the WordNet lemmatizer from the NLTK library.

    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize_words(self, tokens):
        """
        Lemmatizes the given list of tokens.

        Parameters:
        - tokens: A list of tokens.

        Returns:
        - lemmatized_tokens: A list of lemmatized tokens.
        """
        return [self.lemmatizer.lemmatize(word) for word in tokens]


class LowerCaser:
    """
    A class responsible for lowercasing words.

    The LowerCaser class converts all words to lowercase.

    """

    def lower_case_words(self, tokens):
        """
        Converts the given list of tokens to lowercase.

        Parameters:
        - tokens: A list of tokens.

        Returns:
        - lowercased_tokens: A list of lowercase tokens.
        """
        return [word.lower() for word in tokens]
