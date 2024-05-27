from pathlib import Path

from itertools import chain
from common.token import Instrument

from six.moves import cPickle as pickle

from common import config
from preprocessing.extract import extract
from preprocessing.tokenise import tokenise
from preprocessing.lookup import Lookup

class Preprocessor:
    def __init__(self, genre=config.GENRE, sample_size=config.SAMPLE_SIZE):
        '''
        Initialise the pre-processor.

        Parameters:
            genre (str): The genre the model was trained on.
            sample_size (int): The sample size the model was trained on.
        '''
        dataset_fp = f'resources/{genre}_{sample_size}.pickle'

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

        pieces = extract(genre, sample_size)
        tk_pieces = [tokenise(symbols) for symbols in pieces]

        instruments = [
            token for token in chain.from_iterable(tk_pieces)
            if isinstance(token, Instrument)
        ]
        self.lookup.update_instruments(instruments)

        self.samples = [self.get_mapping(tokens) for tokens in tk_pieces]

        data = {
            'lookup':self.lookup,
            'samples':self.samples
        }

        with open(dataset_fp, 'wb') as handle:
            pickle.dump(data, handle)

        print(f'Successfully created and saved')

    def get_mapping(self, tokens):
        '''
        Get the integer tokens from the token objects.

        Parameters:
            tokens (list[Token]): The token objects.

        Returns:
            mapping (list[int]): The integer tokens.
        '''
        return self.lookup.tokens_to_mapping(tokens)

    def get_samples(self):
        '''
        Get the training samples.

        Returns:
            samples (list[list[int]]): The training samples.
        '''
        return self.samples

    def get_vocabulary_size(self):
        '''
        Get the number of tokens in the vocabulary.

        Returns:
            vocabulary_size (int): The number of tokens.
        '''
        return self.lookup.size()
