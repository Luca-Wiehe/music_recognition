"""
Neural Networks for predicting monophonic score sequences.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MonophonicModel(nn.Module):
   
    def __init__(self, hparams, output_size):
        super().__init__()
        self.hparams = hparams

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.lstm1 = nn.LSTM(2048, 256, bidirectional=True)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True)

        self.output_block = nn.Sequential(
            nn.Linear(256*2, output_size + 1),  # times 2 because of bidirectional
            nn.LogSoftmax(dim=-1)
        )

        # set optimizer
        self.set_optimizer()

    def forward(self, x):
        # conv_block has output shape [batch_size, 256, H, W]
        x = self.conv_block(x)

        # reshape to [batch_size, width, 256*height] for compatibility with recurrent_block
        x = x.view(x.shape[0], x.shape[3], x.shape[1]*x.shape[2])

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # linear unit shape (batch_size, width, output_size + 1)
        x = self.output_block(x)

        return x

    def training_step(self, batch, loss_func, device):
        
        # training model
        self.train()
        self.optimizer.zero_grad() # reset gradients

        # load data
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        # predictions have shape (batch_size, width, pred_sequence)
        preds = self.forward(inputs) # make predictions

        # reshape to (width, batch_size, pred_sequence)
        preds = preds.permute(1, 0, 2)

        # input lenghts are width of predictions
        input_lengths = torch.full((preds.shape[1],), preds.shape[0], dtype=torch.int32, device=device)

        # target lengths are number of non-padding elements in target sequence
        target_lengths = calculate_target_lengths(targets)

        loss = loss_func(preds, targets, input_lengths, target_lengths) # compute loss
        loss.backward() # obtain weight updates
        self.optimizer.step() # update weights

        return loss

    def validation_step(self, batch, loss_func, device):
        
        # validation mode
        self.eval()

        with torch.no_grad():
            # load data
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            # predictions have shape (batch_size, width, pred_sequence)
            preds = self.forward(inputs) # make predictions

            # reshape to (width, batch_size, pred_sequence)
            preds = preds.permute(1, 0, 2)

            # input lenghts are width of predictions
            input_lengths = torch.full((preds.shape[1],), preds.shape[0], dtype=torch.int32, device=device)

            # target lengths are number of non-padding elements in target sequence
            target_lengths = calculate_target_lengths(targets)

            loss = loss_func(preds, targets, input_lengths, target_lengths) # compute loss

        return loss


   
    def set_optimizer(self):
        """
        Sets the optimizer for the neural network model according to hyperparameters
        of the model.
        """
        optim_hparams = self.hparams["optimizer"]

        self.optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=optim_hparams["learning_rate"],
            weight_decay=optim_hparams["weight_decay"]
        )

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

def calculate_target_lengths(targets):
    """
    Calculate the length of each target sequence in the batch.

    Parameters:
    - targets: tensor of shape (batch_size, width, 1)

    Returns:
    - target_lengths: tensor of shape (batch_size)
    """
    target_lengths = torch.sum(targets != 0, dim=1)
    return target_lengths