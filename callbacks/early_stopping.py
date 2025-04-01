import torch
import numpy as np

# source: https://github.com/Bjarten/early-stopping-pytorch


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.best_valid_loss = None
        self.valid_loss_min = np.Inf
        self.delta = delta

    def __call__(self, valid_loss, model, save_model_dir=None, save_model_name='checkpoint.pth'):
        # Check if validation loss is nan
        if np.isnan(valid_loss):
            print('Validation loss is Nan. Ignoring this epoch.')
            return

        if self.best_valid_loss is None:
            self.best_valid_loss = valid_loss
            self.save_checkpoint(valid_loss, model, save_model_dir, save_model_name)
        elif valid_loss < self.best_valid_loss - self.delta:
            # Significant improvement detected
            self.best_valid_loss = valid_loss
            self.save_checkpoint(valid_loss, model, save_model_dir, save_model_name)
            self.counter = 0  # Reset counter since improvement occurred
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, valid_loss, model, save_model_dir, save_model_name='checkpoint.pth'):
        # Saves model when validation loss decreases
        if self.verbose:
            print(f'Validation loss decreased ({self.valid_loss_min:.4f} --> {valid_loss:.4f}). Saving model ...')
        torch.save(model.state_dict(), save_model_dir + '/' + save_model_name + '.pth')
        self.valid_loss_min = valid_loss