#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


# In[28]:


url = "C:/Users/mfavre4/Desktop/test/2022/df_Ucore_ACW_AE.csv"
df = pd.read_csv(url, sep=",")
dataset = df.astype(float)



ds_CW = (dataset.loc[:,"I_S1":"total_torque"])

data = np.arange(0,len(ds_CW),1)
xdata = ds_CW[['I_S1']].fillna(0).to_numpy()
#pd.DataFrame(data)

ydata = ds_CW[['total_torque']].fillna(0).to_numpy()

# Generate a dataset of a simulated dynamical system
# using the train test split function

x_train, x_valid,y_train, y_valid = train_test_split(ydata,xdata ,
                                   random_state=104, 
                                   test_size=0.25, 
                                   shuffle=True)
ds_CW.head()


# In[26]:


basis_function = Polynomial(degree=3)
model = FROLS(
    order_selection=True,
    n_info_values=39,
    extended_least_squares=False,
    ylag=20,
    xlag=10,
    info_criteria='aic',
    estimator='least_squares',
    basis_function=basis_function
)
model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
rrse = root_relative_squared_error(y_valid, yhat)
print(rrse)
r = pd.DataFrame(
    results(
        model.final_model, model.theta, model.err,
        model.n_terms, err_precision=8, dtype='sci'
        ),
    columns=['Regressors', 'Parameters', 'ERR'])
print(r)
plot_results(y=y_valid[0:100], yhat=yhat[0:100], n=1000)
ee = compute_residues_autocorrelation(y_valid, yhat)
plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
x1e = compute_cross_correlation(y_valid, yhat, x2_val)
plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")


# In[ ]:




