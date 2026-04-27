import tqdm
import torch
from torch import nn
import numpy as np
from torchmetrics import Accuracy, MeanSquaredError, MeanAbsoluteError

class Trainer:
    """ Convenience wrapper to provide model fitting and evaluation functions. """

    def __init__(self, model, loss_fn, metric, device):
        """ Create a Trainer object.

            Arguments:
                model: the model
                loss_fn: loss function
                metric: TorchMetrics metric object
                device: Torch device e.g. cpu or cuda
        """
        super().__init__()
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.metric = metric.to(device)
        self.device = device

    def evaluate(self, dl, verbose=False):
        """ Compute the average metric over a dataset.
            Arguments:
                dl: dataloader
                verbose: if True, will show a progress bar
            Returns:
                average metric value
        """
        self.model.eval()
        self.metric.reset()
        if verbose:
            batches = tqdm.tqdm(dl,desc='evaluate')
        else:
            batches = dl
        for batch in batches:
            x_batch, y_batch = batch
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            y_pred = self.model(x_batch)
            self.metric(y_pred,y_batch)
        return self.metric.compute().item()        
    
    def predict(self, dl, verbose=False):
        """ Compute the model outputs for a dataset.

            Arguments:
                dl: dataloader
                verbose: if True, will show a progress bar
            Returns:
                Numpy array containing the model outputs
        """
        self.model.eval()
        self.metric.reset()
        if verbose:
            batches = tqdm.tqdm(dl,desc='predict')
        else:
            batches = dl
        preds = []
        for batch in batches:
            x_batch, _ = batch
            x_batch = x_batch.to(self.device)
            y_pred = self.model(x_batch)
            preds.append(y_pred.detach().cpu().numpy())
        return np.stack(preds)

    def fit(self, dl_train, optimizer, dl_val=None, num_epochs=10, checkpoint_path=None, verbose=False):
        """ Run model training.

            Arguments:
                dl_train: training data loader
                optimizer: optimizer pointing to the model parameters
                dl_val: optional validation data loader, for computing metrics at end of each epoch
                num_epochs: number of epochs
                checkpoint_path: optional path for saving weights of best model
                verbose: if True, will show progress bar for each epoch and report metrics during training
            Returns:
                train_metrics: training metric per epoch
                val_metrics: validation metric per epoch (if dl_val provided)
        """
        train_metrics = []
        val_metrics = []
        for epoch in range(num_epochs):
            self.model.train()
            self.metric.reset()
            if verbose:
                batches = tqdm.tqdm(dl_train,desc=f'epoch {epoch+1}/{num_epochs}')
            else:
                batches = dl_train
            for x_batch, y_batch in batches:
                # move data to device
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # compute model output and loss
                y_pred = self.model(x_batch)
                loss = self.loss_fn(y_pred,y_batch)

                # compute metric for train batch
                self.metric(y_pred,y_batch)

                # update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # compute average metric over train batches
            train_metric = self.metric.compute().item()
            train_metrics.append(train_metric)

            val_metric = None
            if dl_val is None:
                if verbose:
                    print(f'Epoch {epoch}: train {train_metric}')
            else:
                # compute metric on validation set
                val_metric = self.evaluate(dl_val,verbose=verbose)
                if checkpoint_path is not None:
                    do_save = False
                    if len(val_metrics) == 0:
                        do_save = True
                    else:
                        do_save = (self.metric.higher_is_better and (val_metric > np.max(val_metrics))) or (not self.metric.higher_is_better and (val_metric < np.min(val_metrics)))
                    
                    if do_save:
                        torch.save(self.model.state_dict(),checkpoint_path)
                    
                val_metrics.append(val_metric)

                if verbose:
                    print(f'Epoch {epoch}: train {train_metric} val {val_metric}')

        if dl_val is None:
            if checkpoint_path is not None:
                torch.save(self.model.state_dict(),checkpoint_path)
            return train_metrics
        else:
            if checkpoint_path is not None:
                self.model.load_state_dict(torch.load(checkpoint_path))
            return train_metrics, val_metrics

class ClassificationTrainer(Trainer):
    """ Trainer subclass for classification models using cross-entropy loss and accuracy metric. """
    def __init__(self, model, num_classes, device):
        super().__init__(
            model = model,
            loss_fn = nn.CrossEntropyLoss(),
            metric = Accuracy(task='multiclass',num_classes=num_classes),
            device = device
        )

class RegressionTrainer(Trainer):
    """ Trainer subclass for regression models using mean squared error loss and metric. """
    def __init__(self, model, device):
        super().__init__(
            model = model,
            loss_fn = nn.MSELoss(),
            metric = MeanSquaredError(),
            device = device
        )

class AutoencoderTrainer:
    """ Trainer for autoencoder models."""
    def __init__(self, encoder, decoder, device, use_mae=False, activity_reg=0.):
        """ Create an AutoencoderTrainer object.
        
            Arguments:
                encoder: encoder model
                decoder: decoder model
                device: Torch device
                use_mae: if True, use mean absolute error instead of mean squared error
                activity_reg: weight for L1 regularization on decoder output
        """
        super().__init__()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        if use_mae:
            self.loss_fn = nn.L1Loss().to(device)
            self.metric = MeanAbsoluteError().to(device)
        else:
            self.loss_fn = nn.MSELoss().to(device)
            self.metric = MeanSquaredError().to(device)
        self.device = device
        self.activity_reg = activity_reg
        
    def encode(self, dl, verbose=False):
        """ Compute output of encoder on a dataset.

            Arguments:
                dl: data loader
                verbose: if True, will show a progress bar
            Returns:
                Numpy array of encoder outputs
        """
        self.encoder.eval()
        if verbose:
            batches = tqdm.tqdm(dl,desc='predict')
        else:
            batches = dl
        preds = []
        for batch in batches:
            x_batch, _ = batch
            x_batch = x_batch.to(self.device)
            y_pred = self.encoder(x_batch)
            preds.append(y_pred.detach().cpu().numpy())
        return np.stack(preds)
    
    def fit(self, dl_train, optimizer, dl_val=None, num_epochs=10, verbose=False):
        """ Train the autoencoder on a dataset.

            Arguments:
                dl_train: training data loader
                optimizer: optimizer for training
                dl_val: validation data loader
                num_epochs: number of epochs
                verbose: if True, show progress bar and metrics at end of each epoch
            Returns:
                train_metrics: training metric per epoch
                val_metrics: validation metric per epoch (if dl_val provided)
        """
        train_metrics = []
        val_metrics = []
        for epoch in range(num_epochs):
            self.encoder.train()
            self.decoder.train()
            self.metric.reset()
            if verbose:
                batches = tqdm.tqdm(dl_train,desc=f'epoch {epoch+1}/{num_epochs}')
            else:
                batches = dl_train
            for x_batch in batches:
                if isinstance(x_batch,tuple) or isinstance(x_batch,list):
                    x_batch = x_batch[0]
                
                # move data to device
                x_batch = x_batch.to(self.device)

                # compute model output and loss
                h_pred = self.encoder(x_batch)
                x_pred = self.decoder(h_pred)
                loss = self.loss_fn(x_pred,x_batch)

                # compute activity regularization
                if self.activity_reg > 0:
                    reg = torch.mean(torch.abs(h_pred))
                    loss += self.activity_reg * reg

                # compute metric for train batch
                self.metric(x_pred,x_batch)

                # update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # compute average metric over train batches
            train_metric = self.metric.compute().item()
            train_metrics.append(train_metric)

            val_metric = None
            if dl_val is None:
                if verbose:
                    print(f'Epoch {epoch}: train {train_metric}')
            else:
                # compute metric on validation set
                self.encoder.eval()
                self.decoder.eval()
                self.metric.reset()
                if verbose:
                    batches = tqdm.tqdm(dl_val,desc='validation')
                else:
                    batches = dl_val
                for x_batch in batches:
                    if isinstance(x_batch,tuple) or isinstance(x_batch,list):
                        x_batch = x_batch[0]
                    x_batch = x_batch.to(self.device)
                    x_pred = self.decoder(self.encoder(x_batch))
                    self.metric(x_pred,x_batch)
                val_metric = self.metric.compute().item()                    
                val_metrics.append(val_metric)

                if verbose:
                    print(f'Epoch {epoch}: train {train_metric} val {val_metric}')

        if dl_val is None:
            return train_metrics
        else:
            return train_metrics, val_metrics

def vae_loss_fn(x,y,z_mean,z_log_var,beta=1):
    squared_error = (x-y)**2
    reconstruction_loss = torch.sum(squared_error,(1,2,3))

    kl_loss = 1 + z_log_var - z_mean**2 - torch.exp(z_log_var)
    kl_loss = -0.5 * torch.sum(kl_loss, -1)

    vae_loss = torch.mean(reconstruction_loss + beta * kl_loss)

    return vae_loss

class VariationalAutoencoderTrainer:
    """ Trainer for variational autoencoder models."""
    def __init__(self, encoder, decoder, device, beta=1):
        """ Create a VariationalAutoencoderTrainer object.
        
            Arguments:
                encoder: encoder model
                decoder: decoder model
                device: Torch device
                beta: beta coefficient in loss function
        """
        super().__init__()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.loss_fn = vae_loss_fn
        self.metric = MeanSquaredError().to(device)
        self.device = device
        self.beta = beta
        
    def encode(self, dl, verbose=False):
        """ Compute output of encoder on a dataset.

            Arguments:
                dl: data loader
                verbose: if True, will show a progress bar
            Returns:
                Numpy array of encoder outputs
        """
        self.encoder.eval()
        if verbose:
            batches = tqdm.tqdm(dl,desc='predict')
        else:
            batches = dl
        z_means = []
        z_log_vars = []
        for batch in batches:
            x_batch, _ = batch
            x_batch = x_batch.to(self.device)
            z_mean,z_log_var = self.encoder(x_batch)
            z_means.append(z_mean.detach().cpu().numpy())
            z_log_vars.append(z_log_vars.detach().cpu().numpy())
        return np.stack(z_means), np.stack(z_log_vars)
    
    def fit(self, dl_train, optimizer, dl_val=None, num_epochs=10, verbose=False):
        """ Train the autoencoder on a dataset.

            Arguments:
                dl_train: training data loader
                optimizer: optimizer for training
                dl_val: validation data loader
                num_epochs: number of epochs
                verbose: if True, show progress bar and metrics at end of each epoch
            Returns:
                train_metrics: training metric per epoch
                val_metrics: validation metric per epoch (if dl_val provided)
        """
        train_metrics = []
        val_metrics = []
        for epoch in range(num_epochs):
            self.encoder.train()
            self.decoder.train()
            self.metric.reset()
            if verbose:
                batches = tqdm.tqdm(dl_train,desc=f'epoch {epoch+1}/{num_epochs}')
            else:
                batches = dl_train
            for x_batch in batches:
                if isinstance(x_batch,tuple) or isinstance(x_batch,list):
                    x_batch = x_batch[0]
                
                # move data to device
                x_batch = x_batch.to(self.device)

                # compute model output and loss
                z_mean, z_log_var = self.encoder(x_batch)

                # sample latent vector z ~ q(z|x)
                # using the "reparameterization trick"
                b,d = z_mean.shape[:2]
                epsilon = torch.randn(z_mean.shape).to(self.device)
                z = z_mean + torch.exp(0.5 * z_log_var) * epsilon

                x_pred = self.decoder(z)

                loss = self.loss_fn(x_pred,x_batch,z_mean,z_log_var,self.beta)

                # compute metric for train batch
                self.metric(x_pred,x_batch)

                # update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # compute average metric over train batches
            train_metric = self.metric.compute().item()
            train_metrics.append(train_metric)

            val_metric = None
            if dl_val is None:
                if verbose:
                    print(f'Epoch {epoch}: train {train_metric}')
            else:
                # compute metric on validation set
                self.encoder.eval()
                self.decoder.eval()
                self.metric.reset()
                if verbose:
                    batches = tqdm.tqdm(dl_val,desc='validation')
                else:
                    batches = dl_val
                for x_batch in batches:
                    if isinstance(x_batch,tuple) or isinstance(x_batch,list):
                        x_batch = x_batch[0]
                    x_batch = x_batch.to(self.device)
                    z_mean,z_log_var = self.encoder(x_batch)
                    x_pred = self.decoder(z_mean)
                    self.metric(x_pred,x_batch)
                val_metric = self.metric.compute().item()                    
                val_metrics.append(val_metric)

                if verbose:
                    print(f'Epoch {epoch}: train {train_metric} val {val_metric}')

        if dl_val is None:
            return train_metrics
        else:
            return train_metrics, val_metrics

def fit_model(
          model, opt, loss_fn, metric, device,
          dl_train, dl_val,
          epochs, verbose=False
          ):
    train_metrics = []
    val_metrics = []
    for epoch in range(epochs):
        metric.reset()
        if verbose:
            batches = tqdm.tqdm(dl_train,desc=f'epoch {epoch+1}/{epochs}')
        else:
            batches = dl_train
        for batch in batches:
            # unpack data
            x_batch, y_batch = batch
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # compute model output and loss
            y_pred = model(x_batch)
            loss = loss_fn(y_pred,y_batch)

            # compute metric for train batch
            metric(y_pred,y_batch)

            # update weights
            opt.zero_grad()
            loss.backward()
            opt.step()

        # compute average metric over train batches
        train_metric = metric.compute().item()

        # compute metric on validation set
        metric.reset()
        if verbose:
            batches = tqdm.tqdm(dl_val,desc='validation')
        else:
            batches = dl_val
        for batch in batches:
            x_batch, y_batch = batch
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch)
            metric(y_pred,y_batch)
        val_metric = metric.compute().item()

        if verbose:
            print(f'Epoch {epoch}: train {train_metric} val {val_metric}')

        train_metrics.append(train_metric)
        val_metrics.append(val_metric)

    return train_metrics, val_metrics
