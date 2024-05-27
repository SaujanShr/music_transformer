from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch import save

from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy import array

from common import config
from training.loader import get_data_loaders

class Trainer:
    def __init__(self, model, samples,
            genre=config.GENRE,
            epochs=config.EPOCHS,
            max_seq_len=config.MAX_SEQUENCE_LENGTH,
            batch_size=config.BATCH_SIZE,
            split=config.SPLIT,
            device=config.DEVICE,
            lr=config.INITIAL_LEARNING_RATE,
            learning_gamma=config.LEARNING_GAMMA,
            learning_step=config.LEARNING_STEP
        ):
        '''
        Initialise the trainer.

        Parameters:
            model (MusicTransformer): The transformer model.
            samples (list[list[int]]): The training samples.
            genre (str): The genre the model was trained on.
            epochs (int): The number of training epochs.
            max_seq_len (int): The maximum input sequence length.
            batch_size (int): The size of the training batches.
            split (float): The fraction of training samples to validation samples.
            device (DeviceLikeType): The device the model runs on (CPU/GPU).
            lr (float): The initial learning rate.
            learning_gamma (float): The learning decay.
            learning_step (int): The number of training steps for the learning rate to decay.
        '''
        self.loss_fn = CrossEntropyLoss()

        self.optimiser = Adam(
            model.parameters(),
            lr = lr,
            betas = (0.9, 0.98)
        )
        
        self.scheduler = StepLR(self.optimiser, learning_step, learning_gamma)

        self.model = model
        self.genre = genre
        self.epochs = epochs
        self.sample_size = len(samples)
        self.training_loader, self.validation_loader = get_data_loaders(
            samples, max_seq_len, batch_size, split, device
        )

    def _train_batch(self, x, t):
        '''
        Train the model with a batch of inputs and targets.

        Parameters:
            x (tensor[batch_size, max_seq_len]): The inputs.
            t (tensor[batch_size, max_seq_len]): The targets.

        Returns:
            loss (float): The training loss.
        '''
        y = self.model(x).transpose(-1, -2)

        self.optimiser.zero_grad()
        loss = self.loss_fn(y, t)
        loss.backward()

        self.optimiser.step()
        self.scheduler.step()
        
        return float(loss)

    def _val_batch(self, x, t):
        '''
        Evaluate the model with a batch of inputs and targets.

        Parameters:
            x (tensor[batch_size, max_seq_len]): The inputs.
            t (tensor[batch_size, max_seq_len]): The targets.

        Returns:
            loss (float): The validation loss.
        '''
        y = self.model(x).transpose(-1, -2)
        loss = self.loss_fn(y, t)

        return float(loss)
    
    def _train_epoch(self):
        '''
        Train the model over the training dataset.

        Returns:
            train_epoch_losses (list[float]): The training loss over all batches.
        '''
        train_epoch_losses = []

        self.model.train()
        for train_x, train_y in tqdm(self.training_loader):
            loss = self._train_batch(train_x, train_y)
            train_epoch_losses.append(loss)

        return train_epoch_losses
    
    def _val_epoch(self):
        '''
        Evaluate the model over the validation dataset.

        Returns:
            val_epoch_losses (list[float]): The validation loss over all batches.
        '''
        val_epoch_losses = []

        self.model.eval()
        for val_x, val_y in tqdm(self.validation_loader):
            loss = self._val_batch(val_x, val_y)
            val_epoch_losses.append(loss)

        return val_epoch_losses

    def train(self):
        '''
        Train the model over the trainer's epochs and save the model for each trained epoch.
        The training and validation loss is shown at the end of training.
        '''
        train_losses = []
        val_losses = []

        print("Starting training")
        try:
            for epoch in range(self.epochs):
                print(f'Training epoch:{epoch}')
                train_epoch_losses = self._train_epoch()

                print(f'Validating epoch:{epoch}')
                val_epoch_losses = self._val_epoch()

                if train_epoch_losses:
                    train_mean_loss = sum(train_epoch_losses) / len(train_epoch_losses)
                else: train_mean_loss = 0
                if val_epoch_losses:
                    val_mean_loss = sum(val_epoch_losses) / len(val_epoch_losses)
                else: val_mean_loss = 0

                train_losses.append(train_mean_loss)
                val_losses.append(val_mean_loss)

                print(f"Epoch:{epoch} Learning rate:{self.scheduler.get_last_lr()} Train loss:{train_mean_loss} Val loss:{val_mean_loss}\n")

                self.save_model(self.genre, self.sample_size, epoch)

        except KeyboardInterrupt:
            print(f'Cancelled at epoch:{epoch}')

        print("Finished training")

        plt.plot(array(range(len(train_losses))), array(train_losses), label="train")
        plt.plot(array(range(len(val_losses))), array(val_losses), label="validation")
        plt.legend()

        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.show()

        # print(train_losses)
        # print(val_losses)

        # input("Program is paused. Press <ENTER> to continue")


    def save_model(self, genre, sample_size, epoch):
        '''
        Save the model given the details of the model and the current epoch.

        Parameters:
            genre (str): The genre the model was trained on.
            sample_size (int): The sample size the model was trained on.
            epoch (int): The current epoch.
        '''
        file_path = f'bin/model/{genre}_{sample_size}_{epoch}'

        save(self.model.state_dict(), file_path)
