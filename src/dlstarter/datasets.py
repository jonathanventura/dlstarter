import wget
import os
import numpy as np
from scipy.io import loadmat

def get_frey(path='data/'):
    os.makedirs(path,exist_ok=True)
    frey_url = 'https://www.dropbox.com/scl/fi/m70sh4ef39pvy01czc63r/frey_rawface.mat?rlkey=5v6meiap55z68ada2roxwxuql&dl=1'
    frey_path = os.path.join(path,'frey_rawface.mat')
    if not os.path.exists(frey_path):
        wget.download(
            frey_url,
            out=frey_path
        )
    data = np.transpose(loadmat(frey_path)['ff'])
    images = np.reshape(data,(-1,1,28,20))
    return images

def get_spiral(path='data/'):
    os.makedirs(path,exist_ok=True)
    outpath = os.path.join(path,'spiral.txt')
    if not os.path.exists(outpath):
        wget.download('http://cs.joensuu.fi/sipu/datasets/spiral.txt',outpath)
    data = np.loadtxt(outpath,delimiter='\t')
    x = data[:,0:2] # extract the 2D coordinates of the points
    y = data[:,2]-1 # extract the class labels and convert to zero-based indexing
    return x, y
