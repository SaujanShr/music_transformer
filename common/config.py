from torch import device, tensor, cuda
from torch.cuda import is_available

# model

DEVICE = device("cuda" if cuda.is_available() else "cpu")

EMBEDDING_SIZE = 512
FEEDFORWARD_SIZE = 2048

NUM_LAYERS = 8
DROPOUT = 0.1

MAX_SEQUENCE_LENGTH = 100

MAX_GENERATION_LENGTH = 10000
PRIMER = tensor([[0]])

# learning

SAMPLE_SIZE = 100
LEARNING_RATE = 0.1
EPOCHS = 100
SPLIT = 0.9
BATCH_SIZE = 50