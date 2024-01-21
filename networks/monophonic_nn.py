"""
Neural Networks for predicting monophonic score sequences.
"""
import torch
import torch.nn as nn

class MonophonicModel(nn.Module):
   
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

    def forward():
        pass

    def training_step():
        pass

    def validation_step():
        pass
   
    def set_optimizer():
        pass

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)