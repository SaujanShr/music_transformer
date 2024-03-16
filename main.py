from torch import tensor

from processing.processing import Processing
from processing.detokenise import detokenise
from processing.midi import midi

from model.music_transformer import MusicTransformer

from learning.learning_data import get_learning_data

max_seq_len = 100

data_processing = Processing('piano', 10)
transformer = MusicTransformer(data_processing.size(), max_seq_len=100)

pieces = data_processing.get_tk_pieces()
mapping = [data_processing.get_mapping(piece) for piece in pieces]

input_data, training_data = get_learning_data(mapping, max_seq_len)

# tks = data_processing.get_tks(mapping)
# symbols = detokenise(tks)

# midi(symbols, "test.midi")