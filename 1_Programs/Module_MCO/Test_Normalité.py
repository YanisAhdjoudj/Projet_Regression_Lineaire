# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 21:07:18 2020

@author: Yanis
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ndtri
import seaborn as sns
from scipy import stats




def Jarque_Bera_Test(X):

    Nobs=len(X)
    b1_1=(np.sum((X-X.mean())**3))/Nobs
    b1_2=((np.sum((X-X.mean())**2))/Nobs)**(3/2)
    b1=b1_1/b1_2
    
    
    b2_1=(np.sum((X-X.mean())**4))/Nobs
    b2_2=((np.sum((X-X.mean())**2))/Nobs)**(2)
    b2=b2_1/b2_2
    
    Tstat_JB=Nobs*(((b1**2)/6)+(((b2-3)**2)/24))
    
    Pvalue_JB=stats.chi2.sf(Tstat_JB,2)
    
    print("Statistique de test de JB : "+str(Tstat_JB))
    print("Pvalue du test de JB : "+str(Pvalue_JB))  

    return Tstat_JB,Pvalue_JB

