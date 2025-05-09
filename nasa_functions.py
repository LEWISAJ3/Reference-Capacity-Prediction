#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from skfda.preprocessing.dim_reduction import FPCA
import os
import random
from scipy.interpolate import CubicSpline


def partition_files(folder_path,selected_files=None,samples=1,remaining=None,remaining_files=None,seed=120):
# Step 1: Set the folder path
    random.seed(seed)
# Step 2: Get all file names in the folder
    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    if selected_files==None:
        selected_files = random.sample(all_files,k=samples)
    if type(selected_files)!=list:
        selected_files=[selected_files]
    # Step 4: Get the remaining files
    if remaining_files==None:
        remaining_files = [f for f in all_files if f not in  selected_files]
        random.shuffle(remaining_files)
        if type(remaining)==int and remaining<len(remaining_files)-1:
            remaining_files=remaining_files[0:remaining]
    
    # Output
    print("Selected files:", selected_files)
    print("Remaining files:", remaining_files)
    return selected_files,remaining_files
num_obs=2000
def create_matrix(nasa,num_obs=num_obs,partition=None):
    cells=nasa["Cell"].unique()
    amount=len(nasa["Cycle"].unique()) #get number of cycles
    spline_dict={}
    volt_dict={}
    df_dict={}
    temp_dict={}
    c_dict={}
    discharges=[]
    ambients=[]
    sohs=[]
    for cell in cells:
        nasadf=nasa[nasa["Cell"]==cell]
        nasadf=nasadf.copy()
        nasadf.loc[:,"SOH"]=nasadf.loc[:,"Reference Capacity"]/max(nasadf["Reference Capacity"])
        amount=len(nasadf["Cycle"].unique())
        
        for num in nasadf["Cycle"].unique():
    
            name=f"cell{cell}cycle{num}"
            
            df=nasadf[nasadf["Cycle"]==num].reset_index(drop=True)# get subset of cell df that is a certain cycle
            #print(f'original lenghth ={df.shape}')
            df["Time"]=df["Time"]-min(df["Time"])# make time start at 0
            df=df[df["Time"].diff()>0].reset_index(drop=True)# remove anomolous zero-time
            discharges.append(max(abs(df["Reference Capacity"])))
            sohs.append(max(abs(df["SOH"])))
            ambients.append(max(df["Ambient_Temperature"]))
            #print(f'new lenghth ={df.shape}')

            df_dict[name]=df

            spline_v=CubicSpline(df["Time"],df["Voltage"])#fit splines
            spline_c=CubicSpline(df["Time"],df["Current"])
            spline_t=CubicSpline(df["Time"],df["Temperature"])
            t_common=np.linspace(0,1,num_obs)
            t_hat=t_common*max(df["Time"])
            volt_dict["v"+name]=spline_v(t_hat)# save
            c_dict["c"+name]=spline_c(t_hat)
            temp_dict["t"+name]=spline_t(t_hat)
            spline_dict["c"+name]=spline_v.c

            spline_dict["t"+name]=t_hat
    volt=pd.DataFrame(volt_dict).values
    temp=pd.DataFrame(temp_dict).values
    curr=pd.DataFrame(c_dict).values
    if partition!= None:
        volt=volt[partition[0]:partition[1]]
        temp=temp[partition[0]:partition[1]]
        curr=curr[partition[0]:partition[1]]
    print(volt.shape)
    mat=np.vstack([volt,temp,curr])
    return volt,temp,curr,mat,ambients,discharges,sohs
def create_matrix_split(nasa,num_obs=num_obs,partition=None):
    cells=nasa["Cell"].unique()
    amount=len(nasa["Cycle"].unique()) #get number of cycles
    spline_dict={}
    volt_dict={}
    df_dict={}
    temp_dict={}
    c_dict={}
    discharges=[]
    ambients=[]
    sohs=[]
    for cell in cells:
        nasadf=nasa[nasa["Cell"]==cell]
        nasadf=nasadf.copy()
        nasadf.loc[:,"SOH"]=nasadf.loc[:,"Reference Capacity"]/max(nasadf["Reference Capacity"])
        amount=len(nasadf["Cycle"].unique())
        
        for num in nasadf["Cycle"].unique():
    
            name=f"cell{cell}cycle{num}"
            
            df=nasadf[nasadf["Cycle"]==num].reset_index(drop=True)# get subset of cell df that is a certain cycle
            #print(f'original lenghth ={df.shape}')
            df["Time"]=df["Time"]-min(df["Time"])# make time start at 0
            df=df[df["Time"].diff()>0].reset_index(drop=True)# remove anomolous zero-time
            if df.loc[0,"Cycle_Type"]=="discharge":
                
                discharges.append(max(abs(df["Reference Capacity"])))
            else:
                sohs.append(max(abs(df["SOH"])))
                ambients.append(max(df["Ambient_Temperature"]))
                #print(f'new lenghth ={df.shape}')

                df_dict[name]=df

                spline_v=CubicSpline(df["Time"],df["Voltage"])#fit splines
                spline_c=CubicSpline(df["Time"],df["Current"])
                spline_t=CubicSpline(df["Time"],df["Temperature"])
                t_common=np.linspace(0,1,num_obs)
                t_hat=t_common*max(df["Time"])
                volt_dict["v"+name]=spline_v(t_hat)# save
                c_dict["c"+name]=spline_c(t_hat)
                temp_dict["t"+name]=spline_t(t_hat)
                spline_dict["c"+name]=spline_v.c

                spline_dict["t"+name]=t_hat
    volt=pd.DataFrame(volt_dict).values
    temp=pd.DataFrame(temp_dict).values
    curr=pd.DataFrame(c_dict).values
    if partition!= None:
        volt=volt[partition[0]:partition[1]]
        temp=temp[partition[0]:partition[1]]
        curr=curr[partition[0]:partition[1]]
        
    if len(discharges)<len(volt.T):
        print(len(discharges),len(volt.T))
        volt=volt[:,0:len(discharges)-1]
        temp=temp[:,0:len(discharges)-1]
        curr=temp[:,0:len(discharges)-1]
        ambients=ambients[0:len(discharges)-1]
    if len(volt.T)<len(discharges):
        print(len(discharges),len(volt.T))
        discharges=discharges[0:len(volt.T)-1]
       
        
    mat=np.vstack([volt,temp,curr])
    return volt,temp,curr,mat,ambients,discharges,sohs
def smooth_fd(smoother, data, method='fit_transform'):
    return getattr(smoother, method)(data)
from sklearn.ensemble import RandomForestRegressor as RF
import skfda
from skfda.representation.basis import BSplineBasis
from skfda.preprocessing.smoothing import BasisSmoother
def make_fpc(volt,curr,temp,ambients,num_basis=200,rangeval=(0,100),num_components=2,
             func='fit_transform'):
    num_obs = volt.shape[0] 
    time_grid = np.linspace(rangeval[0], rangeval[1], num=num_obs)
    if type(num_components)==int:
        components=np.array([1,1,1])*num_components
    else:
        components=num_components
    # Create a B-spline basis
    
    
    # Convert data to functional objects
    voltage_fd = skfda.FDataGrid(data_matrix=volt.T, grid_points=time_grid)
    current_fd = skfda.FDataGrid(data_matrix=curr.T, grid_points=time_grid)
    temperature_fd = skfda.FDataGrid(data_matrix=temp.T, grid_points=time_grid)
    print("Creating Objects")
    # Smooth the data using the basis representation
    if func=='fit_transform':
        globals()['basis'] = BSplineBasis(n_basis=num_basis, domain_range=rangeval)
        globals()['smoother_v']= BasisSmoother(basis)
        globals()['smoother_c'] = BasisSmoother(basis)
        globals()['smoother_t'] = BasisSmoother(basis)
        globals()['fpca_voltage'] = FPCA(n_components=components[0])
        globals()['fpca_current'] = FPCA(n_components=components[1])
        globals()['fpca_temperature'] = FPCA(n_components=components[2])
        
        


    voltage_fd_smooth = smooth_fd(smoother_v, voltage_fd, method=func)
    current_fd_smooth = smooth_fd(smoother_c, current_fd, method=func)  # or 'fit_transform'
    temperature_fd_smooth = smooth_fd(smoother_t, temperature_fd,method=func)
    print("Smoothing")
    # Perform Functional PCA (FPCA)
   
    print("Getting Components")
    # Fit FPCA models
    voltage_scores = smooth_fd(fpca_voltage,voltage_fd_smooth,func)
    current_scores = smooth_fd(fpca_current,current_fd_smooth,func)
    temperature_scores = smooth_fd(fpca_temperature,temperature_fd_smooth,func)
    print("Fitting")
    # Extract the first FPCA scores
    voltage_scores_comp = voltage_scores[:, :components[0]]
    current_scores_comp = current_scores[:, :components[1]]
    temperature_scores_comp = temperature_scores[:, :components[2]]
    print("Getting Scores")
    
    scoremat=np.hstack([voltage_scores_comp,current_scores_comp,
temperature_scores_comp,np.array(ambients).reshape(-1,1)])
    return scoremat
def make_fpc_complete(mat,ambients,num_basis=200, rangeval=(0, 100), num_components=2,  func='fit_transform'):
    num_obs = mat.shape[0]  # Assuming volt is a (2000, 4883) matrix
    
    # Define time grid
    time_grid = np.linspace(rangeval[0], rangeval[1], num=num_obs)
    
    # Set components as an array if it's an integer
    
    if func=='fit_transform':
        globals()['completebasis'] = BSplineBasis(n_basis=num_basis, domain_range=rangeval)
        globals()['smoother_mat'] = BasisSmoother(completebasis)
        globals()['fpca']= FPCA(n_components=num_components)
        
    
    print("Creating Objects")
    # Convert the data to functional objects
    mat_fd = skfda.FDataGrid(data_matrix=mat.T, grid_points=time_grid)
    

    # Create smoother objects
    
    
    print("Smoothing")
    # Smooth the data using fit_transform or transform
    mat_fd_smooth = smooth_fd(smoother_mat, mat_fd, method=func)
    
    
    
    print("Getting Scores")
    
    # Perform Functional PCA on the concatenated data
    
    scores = smooth_fd(fpca,mat_fd_smooth,method=func)
    scores_comp = scores[:, :num_components]
    scoremat=np.hstack([scores_comp,np.array(ambients).reshape(-1,1)])
    
    
    
    return scoremat

