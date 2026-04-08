import tqdm
import torch
from torch import nn
from torchmetrics import Accuracy, MeanSquaredError

class Trainer:
    def __init__(self, model, loss_fn, metric, device):
        super().__init__()
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.metric = metric.to(device)
        self.device = device

    def fit(self, dl_train, optimizer, dl_val=None, num_epochs=10, verbose=False):
        train_metrics = []
        val_metrics = []
        for epoch in range(num_epochs):
            self.metric.reset()
            if verbose:
                batches = tqdm.tqdm(dl_train,desc=f'epoch {epoch+1}/{num_epochs}')
            else:
                batches = dl_train
            for batch in batches:
                # unpack data
                x_batch, y_batch = batch
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
                self.metric.reset()
                if verbose:
                    batches = tqdm.tqdm(dl_val,desc='validation')
                else:
                    batches = dl_val
                for batch in batches:
                    x_batch, y_batch = batch
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    y_pred = self.model(x_batch)
                    self.metric(y_pred,y_batch)
                val_metric = self.metric.compute().item()
                val_metrics.append(val_metric)

                if verbose:
                    print(f'Epoch {epoch}: train {train_metric} val {val_metric}')

        if dl_val is None:
            return train_metrics
        else:
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
