# -*- coding: utf-8 -*-
"""
@author: Nicolás Nieto
    Example for the developed pruning algorithm 
    
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

from RBP_Utilitys import RBP_class

# Create an object RBP with all the necessary functions
RBP=RBP_class()

# In[]: Parameters
test_size= 0.3

N_nodes_max=1000

# Change to your data direction
data_dir= ''

file_name= data_dir+'Synthetic_data.mat'

# In[]: Data Load
X_load= sio.loadmat(file_name=file_name)
X_data= X_load['X']
Y_data= X_load['Y']
   
del (X_load , data_dir, file_name)

x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=test_size,shuffle=True)

del (X_data , Y_data, test_size)

# In[]
# Variable initialization for appending
acc_prun=0

# Generate a random network
W , b = RBP.generate_rand_network(x_train, N_nodes_max)

# Fitting with pinv2
B = RBP.fit(x=x_train,W=W,b=b,y=y_train)
        

# Pruning: the pruning step could be changed for faster implementations
for N_nodes_to_prune in range(N_nodes_max):
    
    # Prune the network
    W_prun , b_prun, B_prun = RBP.fix_prunning(W, b, B, prn_perc=N_nodes_to_prune)
    
    # Make a prediction with the pruned netwokr
    y_pred_tst = RBP.predict(x_test, W_prun, b_prun, B_prun)
    
    # Calculaty the accuracy
    acc_aux=accuracy_score(y_test,y_pred_tst)
    
    # Append the accuracy for network size
    acc_prun=np.append(acc_prun,acc_aux)
    


## Ploting
plt.plot(acc_prun)
