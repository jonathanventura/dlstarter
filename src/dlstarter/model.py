import tqdm
import torch

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
