"""
Neural Networks for predicting monophonic score sequences.
"""
import torch
import torch.nn as nn

import data.primus_dataset as data
import utils.utils as utils
import copy

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

        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0) # clip gradients

        self.optimizer.step() # update weights

        if loss == None or torch.isnan(loss):
            print(f"Loss: {loss.item()}\ntargets: {targets}\npreds: {torch.argmax(preds, dim=-1).squeeze(0)}")

        # print(f"Loss: {loss.item()}\ntargets: {targets}\npreds: {torch.argmax(preds, dim=-1).squeeze(0)}")

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

            if loss == None or torch.isnan(loss):
                print(f"targets.shape: {targets.shape}, preds.shape: {preds.shape}")
                print(f"Loss: {loss.item()}\ntargets: {targets}\npreds: {torch.argmax(preds, dim=-1).squeeze(0)}")

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

def train_model(model, train_data, val_data, hparams, tb_logger, device, loss_func=torch.nn.CTCLoss(blank=0, zero_infinity=True), epochs=20):

    # obtain model optimizer
    optimizer = model.optimizer

    # decrease lr of optimizer when reaching a plateau
    scheduler_hparams = hparams["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode=scheduler_hparams["mode"],
        patience=scheduler_hparams["plateau_patience"],
        factor=scheduler_hparams["plateau_decay"],
        threshold=scheduler_hparams["threshold"],
        threshold_mode=scheduler_hparams["threshold_mode"],
        cooldown=scheduler_hparams["cooldown"],
        eps=scheduler_hparams["eps"],
    )

    # select device for model
    model = model.to(device)

    # initialize early stopping criteria
    stopping_hparams = hparams["early_stop"]
    best_loss, best_model, best_optimizer = -1, None, None
    patience, current_patience = stopping_hparams["patience"], stopping_hparams["patience"]

    # run epochs
    for epoch in range(epochs):

        # training for each minibatch
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=hparams["batch_size"], shuffle=False, collate_fn=data.collate_fn)
        train_loop = utils.create_tqdm_bar(train_loader, desc=f"Training Epoch [{epoch + 1}/{epochs}]")
        train_loss, val_loss = 0, 0

        for train_iteration, batch in train_loop:

            # perform training step
            loss = model.training_step(batch, loss_func, device)
            train_loss += loss.item()

            # update progress bar
            train_loop.set_postfix(curr_train_loss = "{:.8f}".format(
                train_loss / (train_iteration + 1)), val_loss = "{:.8f}".format(val_loss)
            )

            # log training loss to tensorboard
            tb_logger.add_scalar(f'CRNN/train_loss', loss.item(), epoch * len(train_loader) + train_iteration)

        # validation for each minibatch
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=hparams["batch_size"], shuffle=False, collate_fn=data.collate_fn)
        val_loop = utils.create_tqdm_bar(val_loader, f"Validation Epoch [{epoch + 1}/{epochs}]")
        val_loss = 0

        for val_iteration, batch in val_loop:

            # perform validation step
            loss = model.validation_step(batch, loss_func, device)
            val_loss += loss.item()

            # update progress bar
            val_loop.set_postfix(val_loss = "{:.8f}".format(val_loss / (val_iteration + 1)))

            # log validation loss to tensorboard
            tb_logger.add_scalar(f'CRNN/val_loss', loss.item(), epoch * len(val_loader) + val_iteration)

        # learning rate update for each epoch
        pre_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(train_loss)
        post_lr = optimizer.param_groups[0]['lr']
        if post_lr < pre_lr:
            print("Loading best model/scheduler due to learning rate decrease.")
            model.load_state_dict(best_model)
            scheduler.load_state_dict(best_scheduler)

        # check for early stopping
        if train_loss < best_loss or best_loss == -1:
            current_patience = patience
            best_loss = train_loss
            best_model = copy.deepcopy(model.state_dict())
            best_optimizer = copy.deepcopy(optimizer.state_dict())
            best_scheduler = copy.deepcopy(scheduler.state_dict())
        else:
            current_patience -= 1

            if current_patience == 0:
                print(f"\n{'===' * 10}\nStopping early at epoch {epoch}\n{'===' * 10}")
                model.load_state_dict(best_model)
                optimizer.load_state_dict(best_optimizer)
                scheduler.load_state_dict(best_scheduler)
                break

        val_loss /= len(val_loader)

    model.load_state_dict(best_model)
    optimizer.load_state_dict(best_optimizer)
    scheduler.load_state_dict(best_scheduler)

    model = model.to(device)
