# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd
import numpy as np

# <codecell>

#intialize parameters for bi-variate mean-reverting process
lam1=0.015
lam2=0.02
mu1=900
mu2=1000
sigma1=25
sigma2=30
rho=0.7

# <codecell>

#number of periods to simulate
T=2000

# <codecell>

#initialize the MR dataframe to store the simulated process
MR=pd.DataFrame(0.1, columns=['MR1','MR2'], index=arange(T))
#intial value of the process
MR.ix[0]=[mu1, mu2]

# <codecell>

#set random seed
from numpy.random import RandomState
prng = RandomState(12345)
#simulate the process
for t in xrange(T-1):
    dW = prng.multivariate_normal([0,0],[[1,rho],[rho,1]])
    next1 = exp(-lam1)*MR['MR1'][t] + (1-exp(-lam1))*mu1 + sigma1*sqrt((1-exp(-2*lam1))/(2*lam1))*dW[0]
    next2 = exp(-lam2)*MR['MR2'][t] + (1-exp(-lam2))*mu2 + sigma2*sqrt((1-exp(-2*lam2))/(2*lam2))*dW[1]
    MR.ix[t+1]=[next1,next2]

# <codecell>

MR.index = pd.date_range('1/1/2000', periods=T) #add an arbitrary date index

# <codecell>

MR.tail()

# <codecell>

MR.plot()

# <codecell>

MR.corr()

# <codecell>

#save the simulated process to a .csv file
MR.to_csv("/Users/buxx/Dropbox/Boyabatli_Nguyen/MPOB_Data/Palm_Price/CPO_PME_Simulated.csv")

# <codecell>

MR.index

# <codecell>

MR['MR1_prev'] = MR.MR1.shift(1)
MR['MR2_prev'] = MR.MR2.shift(1)
MR = MR.dropna()

# <codecell>

MR.head()

# <codecell>

model1 = pd.ols(y=MR.MR1, x=MR.MR1_prev)
model2 = pd.ols(y=MR.MR2, x=MR.MR2_prev)

# <codecell>

model1, model2

# <codecell>

aa1 = model1.beta['x']
bb1 = model1.beta['intercept']
sd1 = model1.rmse

aa2 = model2.beta['x']
bb2 = model2.beta['intercept']
sd2 = model2.rmse

# <codecell>

lambda1 = -log(aa1)
mu1 = bb1/(1-aa1)
delta1 = sd1*sqrt(2*lambda1/(1-aa1**2))

lambda2 = -log(aa2)
mu2 = bb2/(1-aa2)
delta2 = sd2*sqrt(2*lambda2/(1-aa2**2))

# <codecell>

lambda1, mu1, delta1

# <codecell>

lambda2, mu2, delta2

# <codecell>

from statsmodels.sandbox.sysreg import *

# <codecell>

MR_sys = []
MR_sys.append(MR.MR1.values)
MR_sys.append(MR.MR1_prev.values)
MR_sys.append(MR.MR2.values)
MR_sys.append(MR.MR2_prev.values)

# <codecell>

MR.MR1.values

# <codecell>

MR_sys

# <codecell>

SUR(MR_sys)

# <codecell>


