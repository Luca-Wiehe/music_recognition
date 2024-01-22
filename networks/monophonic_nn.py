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
   
    def set_optimizer(self):
            """
            Sets the optimizer for the neural network model according to hyperparameters
            of the model.

            Parameters:
                None

            Returns:
                None
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