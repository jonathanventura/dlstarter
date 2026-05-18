from matplotlib import rc
rc('animation', html='jshtml')

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

def show_video(images,captions):
    """ Show a sequence of images as an interactive plot. """
    fig = plt.figure()
    ax = plt.axes()
    def f(i):
        ax.imshow(images[i])
        ax.set_title(captions[i])
    anim = animation.FuncAnimation(fig, f, frames=len(images), blit=False, repeat=True)
    return anim