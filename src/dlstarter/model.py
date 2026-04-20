import tqdm
import torch
from torch import nn
import numpy as np
from torchmetrics import Accuracy, MeanSquaredError, MeanAbsoluteError

class Trainer:
    def __init__(self, model, loss_fn, metric, device):
        super().__init__()
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.metric = metric.to(device)
        self.device = device

    def evaluate(self, dl, verbose=False):
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
    def __init__(self, model, num_classes, device):
        super().__init__(
            model = model,
            loss_fn = nn.CrossEntropyLoss(),
            metric = Accuracy(task='multiclass',num_classes=num_classes),
            device = device
        )

class RegressionTrainer(Trainer):
    def __init__(self, model, device):
        super().__init__(
            model = model,
            loss_fn = nn.MSELoss(),
            metric = MeanSquaredError(),
            device = device
        )

class AutoencoderTrainer:
    def __init__(self, encoder, decoder, device, use_mae=False, activity_reg=0.):
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
