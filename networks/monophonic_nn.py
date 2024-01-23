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

        self.recurrent_block = nn.Sequential(
            nn.LSTM(256, 256, bidirectional=True),
            nn.LSTM(512, 256, bidirectional=True)
        )

        self.output_block = nn.Sequential(
            nn.Linear(256*2, output_size + 1),  # times 2 because of bidirectional
            nn.LogSoftmax(dim=-1)
        )

    def forward(self):
        x = self.convolutional_block(x)
        # Assuming the output of convolutional block is (batch_size, channels, height, width)
        # We need to permute it to (width, batch_size, channels*height) because the input to LSTM should have the sequence length as the first dimension
        batch_size = x.size(0)
        x = x.permute(3, 0, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)  # Flatten the channels and height dimensions
        
        x, _ = self.recurrent_block(x)
        
        # We take the output of the last time step
        x = x[-1]
        
        x = self.output_block(x)
        return x

    def training_step():
        pass

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