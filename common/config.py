from torch import device, cuda
from common import token

# DO NOT EDIT THE DEVICE
DEVICE = device("cuda" if cuda.is_available() else "cpu") # GPU if exists, else CPU


# The dimensions of the MuTr.

EMBEDDING_SIZE = 512 # Number of embedding dimensions (must be multiple of 64)
FEEDFORWARD_SIZE = 1024 # Number of inner hidden layer nodes in the feed-forward network
NUM_LAYERS = 6 # Number of decoder layers
DROPOUT = 0.1 # Proportion of neurons disabled in the dropout
MAX_SEQUENCE_LENGTH = 512 # Maximum input token sequence length


# Training

GENRE = 'example' # Genre folder name
SAMPLE_SIZE = 10 # Number of files to process
INITIAL_LEARNING_RATE = 0.1 # Initial learning rate
LEARNING_GAMMA = 0.9 # Learning decay
LEARNING_STEP = 4000 # Number of training steps for the learning rate to decay
EPOCHS = 30 # Number of training epochs
SPLIT = 0.9 # Fraction of training samples to validation samples
BATCH_SIZE = 2 # Size of the training batches


# Generation

MIDI_FILE_NAME = 'midi' # Name of the output MIDI file
MODEL_GENRE = 'example' # Genre of trained model
MODEL_SAMPLE_SIZE = 10 # Sample size of trained model
MODEL_EPOCH = 0 # Epoch of trained model
MAX_GENERATION_LENGTH = 1000 # Maximum generated output token sequence length
PRIMER = [token.Start()] # Initial tokens of the output token sequence