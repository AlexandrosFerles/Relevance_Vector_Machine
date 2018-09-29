
# coding: utf-8

# In[35]:

from sklearn import datasets
import numpy as np

def generate_Dataset(name_dataset):
    # default 
    ind = f(name_dataset)
    if ind == 1:
        x,y = datasets.make_friedman1(n_samples=100, n_features=10, noise=0.0, random_state=None)
    elif ind == 2:
        x,y = datasets.make_friedman2(n_samples=100, noise=0.0, random_state=None)
    elif ind == 3:
        x,y = datasets.make_friedman3(n_samples=100, noise=0.0, random_state=None)
    else :
        x, y = datasets.load_boston(return_X_y=True)
     
    x = x.tolist()
    return x,y

def f(x):
        return {
            'friedman1': 1,
            'friedman2': 2,
            'friedman3': 3,
            'boston': 4
        }.get(x, 1) 

x,y = generate_Dataset('friedman1')
print x


# In[ ]:




# In[ ]:



