from decimal import getcontext, ROUND_CEILING

from itertools import chain

from common.token import End
from processing.extract import extract
from processing.tokenise import tokenise
from processing.detokenise import detokenise
from processing.lookup import Lookup

getcontext().prec = 2
getcontext().rounding = ROUND_CEILING

class Processing:
    def __init__(self, genre, sample_size=None):
        self.lookup = Lookup()

        pieces = extract(genre, sample_size)
        self.tk_pieces = [tokenise(symbols) for symbols in pieces]

        tokens = list(chain.from_iterable(self.tk_pieces))
        self.lookup.update_mapping(tokens)

    def get_tk_pieces(self):
        return self.tk_pieces

    def get_mapping(self, tokens):
        return self.lookup.tokens_to_mapping(tokens)

    def get_tks(self, mapping):
        return self.lookup.mapping_to_tokens(mapping)

    def size(self):
        return self.lookup.size()

    def get_end_token_mapping(self):
        end_token = End()
        return self.lookup.token_to_mapping(end_token)
