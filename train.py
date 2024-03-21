from argparse import ArgumentParser

from torch.optim import Adam
from torch import tensor

from common import config
from preprocessing.preprocessing import Preprocessing
from model.music_transformer import MusicTransformer
from training.trainer import Trainer

parser = ArgumentParser("train")
parser.add_argument("genre", help="Genre of music")
args = parser.parse_args()

genre = args.genre

preprocessing = Preprocessing(genre)

print(config.DEVICE)
transformer = MusicTransformer(preprocessing.size()).to(config.DEVICE)

trainer = Trainer(transformer)

samples = preprocessing.get_samples()

trainer.train(samples)
