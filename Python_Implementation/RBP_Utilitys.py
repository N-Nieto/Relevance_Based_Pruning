# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 19:12:16 2020

@author: Nicolás
"""

class RBP:
    
    
    import numpy as np
    import scipy.linalg as sc_lin
    
    
    def __init__(self):
        return
    
    def generate_rand_network(self, x, N_nodes, distr="uniform"):
        
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
            W= (self.np.random.rand(dim, N_nodes) *2) -1
            b= (self.np.random.rand(1, N_nodes) *2) -1
        if distr=="normal":   
            W=self.np.random.normal(size=[dim,N_nodes])
            b=self.np.random.normal(size=[N_nodes])
    
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
    
    def fit(self, x, W, b, y, Reg=0):
        
        H = self.generate_H(x, W, b)
            
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
     
        return B
    
    
    def predict(self, x, W, b, B):
        # Generate H
        H = self.generate_H(x,W,b)
        
        # Make a prediction
        y = self.np.dot(H,B)
        
        # Use the sign as class predictor
        y = self.np.sign(y)
        
        return y
    
    def fix_prunning(self, W, b, B, prn_perc, mode="keep"):
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
        if type(prn_perc)==int:
            n = prn_perc
        
        # For Fix percent
        if type(prn_perc)==float:
            if prn_perc<1:
                n=int ( self.np.floor(prn_perc * self.np.size(B)))
                # Leave at least 1 neuron
                if n==0:
                    n=1
            else:
                Warning('Wrong pruning percent')
        
        
        if n > B.shape[0]:
            n = B.shape[0]
          
        # Calculate the nodes ranking
        B_abs = self.np.abs(B)        
        idx = self.np.argsort(B_abs,axis=0)
        
        if mode == "keep":
            idx_prun  = idx[0:n]
            
        if mode == "prune":
            idx_prun  = idx[0:-n]
            
        # Prune the network
        B_pruned = B[idx_prun]
        W_pruned = W[:,idx_prun]
        b_pruned = b[0,idx_prun]
        
        return W_pruned,b_pruned,B_pruned
    

