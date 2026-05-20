from ipywidgets import interact, IntSlider

import torch
import mlxtend.plotting
from matplotlib import pyplot as plt
import matplotlib.animation as animation

class ModelAdapter:
  def __init__(self, model):
      self.model = model

  def predict(self, X):
      y_pred = self.model(torch.tensor(X).float())
      return torch.argmax(y_pred, axis=-1)

def plot_decision_regions(model,x,y):
  mlxtend.plotting.plot_decision_regions(x, y, clf=ModelAdapter(model))

def show_video(images,captions=None):
    def f(i):
        plt.imshow(images[i])
        if captions is not None:
            plt.title(captions[i])
    return interact(f, i=IntSlider(min=0, max=len(images)-1, step=1, value=0))
