from common import config
from preprocessing.preprocessor import Preprocessor
from model.music_transformer import MusicTransformer
from training.trainer import Trainer

preprocessor = Preprocessor()

print(f'torch using device:{config.DEVICE}')

vocab_size = preprocessor.get_vocabulary_size()
transformer = MusicTransformer(vocab_size).to(config.DEVICE)

samples = preprocessor.get_samples()
trainer = Trainer(transformer, samples)

trainer.train()
