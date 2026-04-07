import torch
import mlxtend.plotting

class ModelAdapter:
  def __init__(self, model):
      self.model = model

  def predict(self, X):
      y_pred = self.model(torch.tensor(X).float())
      return torch.argmax(y_pred, axis=-1)

def plot_decision_regions(model,x,y):
  mlxtend.plotting.plot_decision_regions(x, y, clf=ModelAdapter(model))