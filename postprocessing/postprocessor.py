from pathlib import Path
from six.moves import cPickle as pickle
from torch import load

from postprocessing.detokenise import detokenise
from postprocessing.midi import midi

from common import config

class Postprocessor:
    def __init__(self, genre=config.MODEL_GENRE, sample_size=config.MODEL_SAMPLE_SIZE):
        '''
        Initialise the post-processor.

        Parameters:
            genre (str): The genre the model was trained on.
            sample_size (int): The sample size the model was trained on.
        '''
        self.genre = genre
        self.sample_size = sample_size

        dataset_fp = f'resources/{genre}_{sample_size}.pickle'

        assert Path(dataset_fp).is_file()

        print(f'Loading dataset: {dataset_fp}')
        with open(dataset_fp, 'rb') as handle:
            data = pickle.load(handle)

            self.lookup = data['lookup']

        print(f'Successfully loaded')

    def load_model(self, model, epoch=config.MODEL_EPOCH):
        '''
        Load the state from the given epoch in training to the model.

        Parameters:
            model (MusicTransformer): The unloaded model.
            epoch (int): The epoch of the state.
        '''
        model_file = f'{self.genre}_{self.sample_size}_{epoch}'
        model_dict = load(f'bin/model/{model_file}')
        model.load_state_dict(model_dict)
    
    def get_mapping(self, tokens):
        '''
        Get the integer tokens from the token objects.

        Parameters:
            tokens (list[Token]): The token objects.

        Returns:
            mapping (list[int]): The integer tokens.
        '''
        mapping = self.lookup.tokens_to_mapping(tokens)

        return mapping

    def create_midi(self, mapping, filename=config.MIDI_FILE_NAME):
        '''
        Create a MIDI file with the given filename from a list of integer tokens.

        Parameters:
            mapping (list[int]): The integer tokens.
            filename (int): The name of the MIDI file.
        '''
        tokens = self.lookup.mapping_to_tokens(mapping)
        symbols = detokenise(tokens)
        midi(symbols, filename)

    def get_vocabulary_size(self):
        '''
        Get the number of tokens in the vocabulary.

        Returns:
            vocabulary_size (int): The number of tokens.
        '''
        vocabulary_size = self.lookup.size()

        return vocabulary_size