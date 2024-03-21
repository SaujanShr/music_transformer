from pathlib import Path

from itertools import chain
from torch import Tensor
from common.token import Instrument

from six.moves import cPickle as pickle

from common.token import End
from common.config import SAMPLE_SIZE
from preprocessing.extract import extract
from preprocessing.tokenise import tokenise
from preprocessing.detokenise import detokenise
from preprocessing.lookup import Lookup

class Preprocessing:
    def __init__(self, genre):
        self.genre = genre
        self.sample_size = SAMPLE_SIZE

        dataset_fp = f'resources/{genre}_{self.sample_size}_preprocess.pickle'

        if Path(dataset_fp).is_file():
            print(f'Loading dataset: {dataset_fp}')
            with open(dataset_fp, 'rb') as handle:
                data = pickle.load(handle)

                self.lookup = data['lookup']
                self.samples = data['samples']

            print(f'Successfully loaded')
            return

        print(f'Creating dataset: {dataset_fp}')

        self.lookup = Lookup()

        pieces = extract(genre, self.sample_size)
        tk_pieces = [tokenise(symbols) for symbols in pieces]

        tokens = [
            token for token in chain.from_iterable(tk_pieces)
            if isinstance(token, Instrument)
        ]
        self.lookup.update_instruments(tokens)

        self.samples = [self.get_mapping(tokens) for tokens in tk_pieces]

        data = {
            'lookup':self.lookup,
            'samples':self.samples
        }

        with open(dataset_fp, 'wb') as handle:
            pickle.dump(data, handle)

        print(f'Successfully created and saved')

    def get_mapping(self, tokens):
        return self.lookup.tokens_to_mapping(tokens)

    def get_tks(self, mapping):
        return self.lookup.mapping_to_tokens(mapping)

    def get_samples(self):
        return self.samples

    def size(self):
        return self.lookup.size()

