"""
Neural Networks for predicting monophonic score sequences.
"""
import torch
import torch.nn as nn

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
        # conv_block has output shape [batch_size, 256, H, W]
        x = self.conv_block(x)
        print(f"x.shape: {x.shape}")

        # reshape to [batch_size, width, 256*height] for compatibility with recurrent_block
        x = x.view(x.shape[0], x.shape[3], x.shape[1]*x.shape[2])
        print(f"x.reshaped: {x.shape}")
        
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # We take the output of the last time step
        x = x[-1]
        
        x = self.output_block(x)
        return x

    def training_step(self, batch, loss_func, device):
        
        # training model
        self.train()
        self.optimizer.zero_grad() # reset gradients

        # load data
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        print(f"\ninputs.shape: {inputs.shape}, targets.shape: {targets.shape}")

        pred = self.forward(inputs) # make predictions

        loss = loss_func(pred, targets) # compute loss
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