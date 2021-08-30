# -*- coding: utf-8 -*-
"""
@author: Nieto Nicolás -  nnieto@sinc.unl.edu.ar

@code : Updated codes in https://github.com/N-Nieto/Relevance_Based_Pruning/

Please cite: Nieto, N., Ibarrola, F., Peterson, V., Rufiner, H., & Spies, R. (2019). 
Extreme Learning Machine design for dealing with unrepresentative features. 
arXiv preprint https://arxiv.org/abs/1912.02154.

@article{nieto2019extreme,
  title={Extreme Learning Machine design for dealing with unrepresentative features},
  author={Nieto, Nicol{\'a}s and Ibarrola, Francisco and Peterson, Victoria and Rufiner, Hugo and Spies, Ruben},
  journal={arXiv preprint arXiv:1912.02154},
  year={2019}
}


"""
class RBP:
    
    
    import numpy as np
    import scipy.linalg as sc_lin
# =============================================================================
#     import torch 
# =============================================================================
    
    def __init__(self):
        from scipy.stats import kurtosis
        
        self.kurtosis = kurtosis
        return
    
    def generate_rand_network(self, x, N_nodes, distr="uniform" , random_state = None):
        
        """
        Parameters
        ----------
        x : array [examples x dimension]
            data.
        N_nodes : int 
            Network size.
        distr : str, optional
            distributions . The default is "uniform".
    
        Returns
        -------
        W : array [dimension x N_nodes]
            weights matrix.
        b : array [1 x N_nodes]
            biases array.
    
        """
        

        # Data dimension
        dim = x.shape[1]
        
        if distr=="uniform":
            
            W = (self.np.random.rand(dim, N_nodes) *2) -1
            b = (self.np.random.rand(1, N_nodes) *2) -1
            
        if distr == "normal":   
            
            W = self.np.random.normal(size=[dim,N_nodes])
            b = self.np.random.normal(size=[N_nodes])
    
        return W , b 
    
    
    
    def generate_H(self, x, W, b, act_fun="sigmoid"):
        """
        Parameters
        ----------
        x : array [examples x dimension]
            data.
        W : array [dimension x N_nodes]
            weights matrix.
        b : array [1 x N_nodes]
            biases array.
        act_fun : str, optional
            activation function. The default is "sigmoid".
    
        Returns
        -------
        H : array []
            Hidden activation function.
    
        """
        
        H = self.np.dot(x, W)
        
        H = H + b
            
        if act_fun == "sigmoid":
            
            H = self.sigmoid(H)
            
        if act_fun == "ReLU":
            
            H = self.ReLU(H)
              
        return H
    
    def ReLU (self, H):   
        z = self.np.maximum(H, 0, H) 
        return z
    
    
    def sigmoid (self, H):
        # sigmoid function between [-1,1]
        z = (1 - self.np.exp(-H))/(1 + self.np.exp(-H));
        return z
    
    def fit(self, y, x=[], W = [], b = [], H = [],  Reg = 0, device = "cpu"):
        
        
        if  len(H) == 0:
            # If not H was precomputed, check that W and b are not empty
            if len(W) == 0 and len(b) == 0:
                H = self.generate_H(x, W, b)
                
            else:
                Warning("W and b need to be given if H is not precomputed")
            
        if device == "cpu" or device =="CPU":
            if Reg == 0:
                # Train traditional ELM
                Ht = self.sc_lin.pinv2(H)
                B  = self.np.dot(Ht, y)
            else:
                # Train Regularized ELM
                I = self.np.identity(H.shape[1]);
                
                H_t = H.T
                
                p1 = self.np.linalg.inv((self.np.matmul(H_t , H) + Reg * I ))
                
                p2 = self.np.dot(H_t, y)
                
                B = self.np.dot(p1, p2)
                
        if device == "cuda" or device == "CUDA":
            
            if Reg == 0:
                # Train traditional ELM
                H_T = self.torch.from_numpy(H).to("cuda")
                
                Ht = self.torch.linalg.pinv(H_T)
                
                Ht = Ht.to("cpu").numpy()
                
                B  = self.np.dot(Ht, y)  
            else:
                # Train Regularized ELM
                
                I = self.np.identity(H.shape[1]);
                
                H_t = H.T
                
                arg = self.np.matmul(H_t , H) + Reg * I
                
                arg = self.torch.from_numpy(arg).to("cuda")
                
                p1 = self.torch.linalg.inv(arg)
                
                p1 = p1.to("cpu").numpy()
                
                p2 = self.np.dot(H_t, y)
                
                B = self.np.dot(p1, p2)
     
        return B
    
    
    def predict(self, B, x=[], W = [], b = [],  H = []):
        
        
        if  not len(H) == 0:
            # If not H was precomputed, check that W and b are not empty
            if not len(W) == 0 and not len(b) == 0:
                H = self.generate_H(x, W, b)
                
            else:
                Warning("W and b need to be given if H is not precomputed")

        # Make a prediction
        y = self.np.dot(H,B)
        
        # Use the sign as class predictor
        y = self.np.sign(y)
        
        return y
    
    def Relevance_based_pruning(self, B , prn_perc, W = [], b = [], H = [], H_test = [],  mode="keep"):
        """
        Parameters
        ----------
        W : array [dimension x N_nodes]
            weights matrix.
        b : array [1 x N_nodes]
            biases array.
        B : array
            Ouputs weights.
        prn_perc : TYPE
            Pruning percentage.
            If [int] prune/keep (prn_perc) nodes.
            If [float], prune/keep the percentage set in prn_perc
    
        mode : str
            "keep"  = Keep the amount of nodes
            "prune" = Prun the amount of nodes
        Returns
        -------
        W_pruned : array [dimension x Final_nodes]
            weights matrix.
        b_pruned : array [1 x Final_nodes]
            biases array.
        B_pruned : array [1 x Final_nodes]
            Pruned output weights.
    
        """
        
        # For Fix nodes
        if type(prn_perc) == int:
            n = prn_perc
        
        # For Fix percent
        if type(prn_perc) == float:
            
            if prn_perc<1:
                n = int ( self.np.floor(prn_perc * self.np.size(B)))
                # Leave at least 1 neuron
                if n == 0:
                    n = 1
            else:
                Warning('Wrong pruning percent')
        
        
        if n > B.shape[0]:
            n = B.shape[0]
          
        # Calculate the nodes ranking
 
        B_abs = self.np.abs(B)        
        
        idx = self.np.argsort(B_abs,axis=0)
        
        if mode == "keep":
            idx_prun  = idx[len(idx)-n::]
            
        if mode == "prune":
            idx_prun  = idx[0:-n]
            
            
        # Prune the network
        B_pruned = B[idx_prun]

        if not len(H) == 0:
            
            H_prun = H[:,idx_prun]
            H_test_pruned = H_test[:,idx_prun]
            
            return B_pruned , H_prun , H_test_pruned
        
            # If not H was precomputed, check that W and b are not empty
            if not len(W) == 0 and not len(b) == 0:
                
                W_pruned = W[:,idx_prun]
                b_pruned = b[0,idx_prun]
                
                return B_pruned, W_pruned, b_pruned
            
            else:
                Warning("W and b need to be given if H is not precomputed")

