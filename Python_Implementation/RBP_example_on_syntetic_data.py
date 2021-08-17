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

from RBP_Utilitys import RBP

# Create an object RBP with all the necessary functions
RBP=RBP()

# In[]: Parameters
test_size= 0.9

N_nodes_max=500

# Change to your data direction
data_dir= ''

file_name= data_dir+'Synthetic_data.mat'

# In[]: Data Load
X_load= sio.loadmat(file_name=file_name)
X_data= X_load['X']
Y_data= X_load['Y']
   
del (X_load , data_dir, file_name)

K=5
CV=2

# In[]
for cv in range (CV):
    x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=test_size,shuffle=True)
    
    for k in range (K):
        acc_prun=0

        # Generate a random network
        W , b = RBP.generate_rand_network(x_train, N_nodes_max)
        
        # Fitting with pinv2
        B = RBP.fit(x=x_train,W=W,b=b,y=y_train)
                
        # Pruning: the pruning step could be changed for faster implementations
        for N_nodes_to_keep in range(N_nodes_max):
            
            if not(N_nodes_to_keep == 0):
                    
                # Prune the network
                W_prun , b_prun, B_prun = RBP.fix_prunning(W, b, B, prn_perc = N_nodes_to_keep, mode = "keep")
                
                # Make a prediction with the pruned netwokr
                y_pred_tst = RBP.predict(x_test, W_prun, b_prun, B_prun)
                
                # Calculaty the accuracy
                acc_aux=accuracy_score(y_test,y_pred_tst)
                
                # Append the accuracy for network size
                acc_prun=np.append(acc_prun,acc_aux)
            
        
        acc_prun=np.delete(acc_prun,0)
        
        if k == 0:    
            #Inicialiced
            Acc_prun=acc_prun
        else:
            #Stack trials
            Acc_prun = np.vstack((Acc_prun,acc_prun))
    
    
    if cv == 0:    
        #Inicialiced
        Acc_final=Acc_prun
    else:
        #Stack trials
        Acc_final = np.vstack((Acc_final,Acc_prun))
    
## Ploting
plt.plot(np.mean(Acc_final,0))
