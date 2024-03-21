from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch import save
from torch.utils.data import DataLoader
from torch import no_grad
from torch.nn.functional import one_hot

from common.config import LEARNING_RATE, EPOCHS, MAX_SEQUENCE_LENGTH, BATCH_SIZE, SPLIT
from training.dataset import Dataset
from training.loader import get_data_loaders

class Trainer:
    def __init__(self, model):
        self.loss_fn = CrossEntropyLoss()

        self.optimiser = Adam(
            model.parameters(),
            lr = LEARNING_RATE,
            betas = (0.9, 0.98),
            eps = 1e-9
            )

        self.model = model
        self.max_seq_len = MAX_SEQUENCE_LENGTH

    def _train_epoch(self, x, t):
        y = self.model(x).transpose(-1, -2)

        self.optimiser.zero_grad()
        loss = self.loss_fn(y, t)
        loss.backward()
        self.optimiser.step()
        
        return float(loss)

    def _val_epoch(self, x, t):
        y = self.model(x).transpose(-1, -2)

        loss = self.loss_fn(y, t)

        return float(loss)

    def train(self, samples):
        training_loader, validation_loader = get_data_loaders(
            samples, MAX_SEQUENCE_LENGTH, BATCH_SIZE, SPLIT)
        train_losses = []
        val_losses = []

        for epoch in range(EPOCHS):
            train_epoch_losses = []
            val_epoch_losses = []

            self.model.train()
            for train_x, train_y in training_loader:
                loss = self._train_epoch(train_x, train_y)
                train_epoch_losses.append(loss)

            self.model.eval()
            for val_x, val_y in validation_loader:
                loss = self._val_epoch(val_x, val_y)
                val_epoch_losses.append(loss)

            train_mean = sum(train_epoch_losses) / len(train_epoch_losses)
            val_mean = sum(val_epoch_losses) / len(val_epoch_losses)

            print(f"Epoch:{epoch} Train loss:{train_mean} Val loss:{val_mean}")
        
            self.save_model("piano", "1000")


    def save_model(self, genre, sample_size):
        file_path = f'bin/{genre}_{sample_size}'

        save(self.model.state_dict(), file_path)
