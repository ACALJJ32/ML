```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```
```python
# Import Libraries
np.random.seed(0) 
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline
```
While learning Pytorch, I set a ANN(4 layers) whose hidden layers have (28*28*2 + 1) inter nodes. It takes more than 1 hr to train this 4-Layer NN, in this notebook, I wrote how to use gpu in kaggle.

# Put everything on gpu

First, turn gpu accelerator on.
```python
train = pd.read_csv(r"../input/fashionmnist/fashion-mnist_train.csv",dtype = np.float32)
test =  pd.read_csv(r"../input/fashionmnist/fashion-mnist_test.csv",dtype = np.float32)
```
```python
#set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```
```python
y_train = train['label']
X_train = train.drop(labels = ['label'], axis = 1)/255 
X_test = test.drop(labels = ['label'], axis = 1)/255
```


