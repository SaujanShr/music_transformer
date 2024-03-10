from data_representation.dataset import Dataset

from model.word_embedding import WordEmbedding
from model.positional_embedding import PositionalEmbedding
from model.feed_forward import FeedForward

from torch import tensor

dataset = Dataset()
samples = dataset.get_samples('piano', 20)

mapping = dataset.get_mapping(samples[0])

print(mapping)
# mapping = dictionary.tokens_to_mapping(first_vector)

# print(instruments)
# print(first_vector)
# print(mapping)

# lookup_tensor = tensor(mapping)



# we = WordEmbedding(dictionary_size, 8)
# pe = PositionalEmbedding(50, 8)

# embedding = we(lookup_tensor)
# pe_embedding = pe(embedding)

# print(embedding.shape)
# print(pe_embedding.shape)

# ff = FeedForward(8, 8)

# ff_out = ff(pe_embedding)

# print(ff_out.shape)

# print(pe_embedding)
# print(ff_out)

# print(embedding)
# print(pe.forward(embedding))

