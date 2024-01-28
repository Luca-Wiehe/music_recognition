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

        self.lstm1 = nn.LSTM(256, 256, bidirectional=True)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True)

        self.output_block = nn.Sequential(
            nn.Linear(256*2, output_size + 1),  # times 2 because of bidirectional
            nn.LogSoftmax(dim=-1)
        )

        # set optimizer
        self.set_optimizer()

    def forward(self, x):
        print("\t[...] conv_block")
        # conv_block has output shape [batch_size, 256, H, W]
        x = self.conv_block(x)

        print("\t[...] reshape")
        # reshape to [batch_size, width, 256*height] for compatibility with recurrent_block
        x = x.view(x.shape[0], x.shape[3], x.shape[1]*x.shape[2])
        
        print("\t[...] recurrent_block")
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # linear unit shape (batch_size, width, output_size + 1)
        print("\t[...] output_block")
        x = self.output_block(x)

        return x

    def training_step(self, batch, loss_func, device):

        print("[...] Training Step")

        def calculate_input_lengths(inputs):
            return 
            
        def calculate_target_lengths(targets):
            """
            Calculate the length of each target sequence in the batch.

            Parameters:
            - targets: tensor of shape (batch_size, width, 1)

            Returns:
            - target_lengths: tensor of shape (batch_size)
            """
            target_lengths = torch.sum(targets != -1, dim=1)
            return target_lengths
        
        # training model
        self.train()
        self.optimizer.zero_grad() # reset gradients

        print("[...] Loading data")

        # load data
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        print("[...] Making predictions")

        # predictions have shape (batch_size, width, pred_sequence)
        preds = self.forward(inputs) # make predictions
        print(f"\nPredictions shape: {preds.shape}")

        # reshape to (width, batch_size, pred_sequence)
        preds = preds.permute(1, 0, 2)
        print(f"\nPredictions shape: {preds.shape}")

        # input lenghts are width of predictions
        print("\nAbout to calculate input lengths")
        input_lengths = calculate_input_lengths(inputs)
        print(f"Input lengths calculated: {input_lengths}")

        print("\nAbout to calculate target lengths")
        # target lengths are number of non-padding elements in target sequence
        target_lengths = calculate_target_lengths(targets)
        print(f"Target lengths calculated: {target_lengths}")

        print("\nAbout to calculate loss")
        loss = loss_func(preds, targets, input_lengths, target_lengths) # compute loss
        print(f"Loss calculated: {loss.item()}")
        loss.backward() # obtain weight updates
        self.optimizer.step() # update weights

        return loss

    def validation_step():
        pass
   
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