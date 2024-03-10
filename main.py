from data_representation.dataset import Dataset

from model.word_embedding import WordEmbedding
from model.positional_embedding import PositionalEmbedding
from model.feed_forward import FeedForward

from torch import tensor

dataset = Dataset()
samples = dataset.get_samples('piano', 20)
mapping = dataset.get_mapping(samples[0])

print(mapping)

