from common import config
from preprocessing.preprocessor import Preprocessor
from model.music_transformer import MusicTransformer
from generating.generator import Generator

from postprocessing.postprocessor import Postprocessor

postprocessor = Postprocessor()

print(f'torch using device:{config.DEVICE}')

vocab_size = postprocessor.get_vocabulary_size()
transformer = MusicTransformer(vocab_size).to(config.DEVICE)

postprocessor.load_model(transformer)

generator = Generator(transformer)

primer = postprocessor.get_mapping(config.PRIMER)

result = generator.generate(primer)

postprocessor.create_midi(result)
