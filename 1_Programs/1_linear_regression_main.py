# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 22:04:19 2021

@author: yanis
"""

import os
from datetime import date
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import ndtri



# Main class : Linear_Regression
# Main methods : fit
#              : Predict

# Additional classes :
    
# Plots
# 



class Linear_Regression:
    
    
    def __init__(self,data,varX,varY,intercept=True):
        """
        

        Parameters
        ----------
        intercept : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

            """
        
        self.coef_=None
        self.intercept_= None
        
        self.data=data
        self.varX=varX
        self.varY=varY
        self.intercept=intercept
        
        
    Nobs=len(X)

    X=X.to_numpy()
    
    tX=X.transpose()
    
    Y=Data[varY].to_numpy()
    
    Px=X.dot((np.linalg.inv(tX.dot(X)))).dot(tX)
    
    Mx=np.identity(Nobs)-Px
    
    
    if Const==True:
        df_model=np.size(X,1)-1
    else:
        df_model= np.size(X,1)
    
    df_resid=Nobs-np.size(X,1)
        
    
    def __repr__(self):
        return " This programm provide a Linear regression model"
    
    
    def data_preparation(self):
        # adding a intercept to the data if needed 
        if self.intercept==True:
            try:
                self.data.insert(0,"const",1)
            except:
                pass
            
            self.varX.insert(0,"const")
            self.X=self.data[self.varX]
        else:
            self.X=self.data[self.varX]
            
            
        return self.X
    
    
    
    def fit(self,Estimation="AL",Nb_ite=100000,Learning_rate=0.001,Precision=0.0000000000001):
        
        # Tree type of fitting methods :
        #   Classical econometric approach I  : Least squared
        #   Classical econometric approach II : Log likelihood
        #   Statistical learning approach     : Lost function
        
        # For the Log likelihood and the lost function methods
        # two optimisation methods can be used :
        #   gradiant descent and newpthon raphston
        
        if 
    
    
LR= Linear_Regression()
print(LR)
    
