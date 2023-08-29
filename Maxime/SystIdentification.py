#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import compute_residues_autocorrelation, compute_cross_correlation


# ### Data 

# In[18]:


url = "C:/Users/mfavre4/Desktop/test/2023/U_core_T4B_08June23.csv"
df = pd.read_csv(url, sep=",")
dataset = df.astype(float)

#Inputs - Current of each poles
ds_I = (dataset.loc[:,"I_1":"I_30"])

#Output - Total torque
frames = [(dataset.loc[0:701,"totalTorque_0":"totalTorque_29"]),(dataset.loc[0:701,'totalTorque_30']),(dataset.loc[0:701,'Total_Torque']),(dataset.loc[0:701,'Solved']) ]
ds_T = pd.concat(frames,axis=1)

print(ds_T.shape)

I_id, I_val, xlags = [None for n in range(ds_I.shape[1])],[None for n in range(ds_I.shape[1])],[None for n in range(ds_I.shape[1])]

for i,c in enumerate(ds_I):
    #Creating id arrays with the current of each pole
    exec(str(c)+'_id'+'= ds_I[str(c)][0:492].values.reshape(-1, 1)')
    exec('I_id[i]='+str(c)+'_id')
    
    #Creating validation arrays with the current of each pole
    exec(str(c)+'_val'+'= ds_I[str(c)][492:702:].values.reshape(-1, 1)')
    exec('I_val[i]='+str(c)+'_val')

#Input id and validation sets
x_id = np.concatenate([x for x in I_id], axis =1 )  
x_val = np.concatenate([x for x in I_val], axis =1 ) 

#Output id and validation sets
y_id, y_val = ds_T['Total_Torque'][0:492].values.reshape(-1, 1), ds_T['Total_Torque'][492::].values.reshape(-1, 1)

#Lags for each inputs [[1,2], [1,2]...]
for i,x in enumerate(xlags): xlags[i] = ([i for i in range(1,3)])


# In[19]:


basis_function = Polynomial(degree=3)
model = FROLS(
    order_selection=True,
    n_info_values=39,
    extended_least_squares=False,
    ylag=20,
    xlag=xlags,
    info_criteria='aic',
    estimator='least_squares',
    basis_function=basis_function
)
model.fit(X=x_id, y=y_id)


# In[23]:


yhat = model.predict(X=x_val, y=y_val)
rrse = root_relative_squared_error(y_val, yhat)
print(rrse)
r = pd.DataFrame(
    results(
        model.final_model, model.theta, model.err,
        model.n_terms, err_precision=8, dtype='sci'
        ),
    columns=['Regressors', 'Parameters', 'ERR'])
print(r)
plot_results(y=y_val[0:100], yhat=yhat[0:100], n=1000)


# In[22]:


y_hat = model.predict(X=x_val, y=y_val, steps_ahead=1)
rrse = root_relative_squared_error(y_val, y_hat)
print(rrse)
r = pd.DataFrame(
    results(
        model.final_model, model.theta, model.err,
        model.n_terms, err_precision=8, dtype='sci'
        ),
    columns=['Regressors', 'Parameters', 'ERR'])
print(r)

plot_results(y=y_val, yhat = y_hat, n=1000)


# In[24]:


y_hat = model.predict(X=x_val, y=y_val, steps_ahead=5)
rrse = root_relative_squared_error(y_val, y_hat)
print(rrse)
plot_results(y=y_val, yhat = y_hat, n=1000)


# ### Residue

# In[ ]:


ee = compute_residues_autocorrelation(y_val, yhat)
plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
x1e = compute_cross_correlation(y_val, yhat, x2_val)
plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")

