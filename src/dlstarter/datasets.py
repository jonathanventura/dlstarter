import wget
import os
import numpy as np
from scipy.io import loadmat

def get_frey(path='.'):
    frey_url = 'https://www.dropbox.com/scl/fi/m70sh4ef39pvy01czc63r/frey_rawface.mat?rlkey=5v6meiap55z68ada2roxwxuql&dl=1'
    frey_path = os.path.join(path,'frey_rawface.mat')
    if not os.path.exists(frey_path):
        wget.download(
            frey_url,
            out=frey_path
        )
    data = np.transpose(loadmat('frey_rawface.mat')['ff'])
    images = np.reshape(data,(-1,1,28,20))
    return images
