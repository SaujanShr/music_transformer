from model.word_embedding import WordEmbedding
from model.positional_embedding import PositionalEmbedding

class Transformer:
    def __init__(dictionary_size, input_length=50):
        self.word_embedding = WordEmbedding(dictionary_size)
        self.positional_embedding = PositionalEmbedding(input_length)