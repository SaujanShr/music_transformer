from data_representation.data_extraction import get_pieces_from_genre, get_instruments
from data_representation.tokeniser import piece_to_tokens
from data_representation.mapping import Mapping
from data_representation.detokeniser import tokens_to_piece
from data_representation.midi import piece_to_midi
from data_representation.common import Token

class Dataset:
    def __init__(self):
        self.mapping = Mapping()

    def get_samples(self, genre:str='test', sample_size:int=10) -> list[list[Token]]:
        pieces = get_pieces_from_genre(genre, sample_size)

        instruments = get_instruments(pieces)
        self.mapping.add_instruments(instruments)

        samples = [piece_to_tokens(piece) for piece in pieces]

        return samples

    def get_mapping(self, sample:list[Token]) -> list[float]:
        return self.mapping.tokens_to_mapping(sample)

    def get_vocabulary_size(self) -> int:
        return self.mapping.size()