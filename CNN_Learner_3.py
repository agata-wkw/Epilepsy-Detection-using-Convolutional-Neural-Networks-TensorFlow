import os
import sys
import math
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)

x=np.linspace(-100,100,200)
logits = np.array([[-5.5,1.2],[0.2,0.1],[-0.2,-0.1]])
softmax=np.array([np.exp(-logits[0,:])/(np.exp(-logits[0,0])+np.exp(-logits[0,1])), \
                  np.exp(-logits[1, :]) / (np.exp(-logits[1, 0]) + np.exp(-logits[1, 1])), \
                  np.exp(-logits[2, :]) / (np.exp(-logits[2, 0]) + np.exp(-logits[2, 1]))])

print(softmax)