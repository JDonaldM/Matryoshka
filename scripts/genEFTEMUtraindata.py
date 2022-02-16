import numpy as np
from classy import Class
import pybird
import tqdm
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--inputX", help="Directroy with files containg the training cosmologies.", 
                    required=True)
parser.add_argument("--save_dir", help="Path to save outputs.", 
                    required=True)
parser.add_argument("--redshift", help="Redshift at which to generate the data.", 
                    required=True)
parser.add_argument("--optiresum", help="Boolean. Use pybird optimal resummation. Can be 1 or 0.", 
                    required=False, default=0)
args = parser.parse_args()

# Inputs #########################################
            
# Check inputX
input_path = str(args.inputX)
if not os.path.isfile:
    raise ValueError("Input cosmologies list not found at specified location.")

# Check cache
save_dir = str(args.save_dir)
if not os.path.isfile:
    raise ValueError("Save directory not found.")
else:
    if save_dir[-1] is not "/":
        save_dir += "/"
    for component in ["P110", "P112", "Ploop0", "Ploop2", "Pct0", "Pct2"]:
        if not os.path.isdir(save_dir+component):
            os.mkdir(save_dir+component)

redshift = float(args.redshift)
optiresum = bool(int(args.optiresum))

# Setup ##########################################

# Load cosmology list.
cosmos = np.load(input_path)

# k for linear power.
kk = np.logspace(-5, 0, 200)

# Class setup
M = Class()
M.set({'output': 'mPk',
       'P_k_max_1/Mpc': 1.0,
       'z_max_pk': redshift})

# PyBird setup
common = pybird.Common(optiresum=optiresum)
nonlinear = pybird.NonLinear(load=True, save=True, co=common)
resum = pybird.Resum(co=common)

# Define some empty arrays.
P110_array = np.zeros((cosmos.shape[0], 3, 50))
P112_array = np.zeros((cosmos.shape[0], 3, 50))
Ploop0_array = np.zeros((cosmos.shape[0], 12, 50))
Ploop2_array = np.zeros((cosmos.shape[0], 12, 50))
Pct0_array = np.zeros((cosmos.shape[0], 6, 50))
Pct2_array = np.zeros((cosmos.shape[0], 6, 50))

# Loop ###########################################

for i in tqdm.tqdm(range(cosmos.shape[0])):
    M.set({'ln10^{10}A_s': cosmos[i,3],
           'n_s': cosmos[i,4],
           'h': cosmos[i,2],
           'omega_b': cosmos[i,1],
           'omega_cdm': cosmos[i,0],
          })
    M.compute()
    
    Pk = [M.pk(ki*M.h(), redshift)*M.h()**3 for ki in kk]
        
    f = M.scale_independent_growth_factor_f(redshift)
    
    bird = pybird.Bird(kk, Pk, f, z=redshift, which='all', co=common)
    
    nonlinear.PsCf(bird)
    bird.setPsCfl()
    resum.Ps(bird)
    
    P110_array[i,:,:] = bird.P11l[0]
    P112_array[i,:,:] = bird.P11l[1]
    
    Ploop0_array[i,:,:] = bird.Ploopl[0]
    Ploop2_array[i,:,:] = bird.Ploopl[1]
    
    Pct0_array[i,:,:] = bird.Pctl[0]
    Pct2_array[i,:,:] = bird.Pctl[1]

# Save ###########################################

np.save(save_dir+"Ploop0/Ploop0_array-z{z}_optiresum-{o}.npy".format(z=redshift, o=optiresum), Ploop0_array)
np.save(save_dir+"Ploop2/Ploop2_array-z{z}_optiresum-{o}.npy".format(z=redshift, o=optiresum), Ploop2_array)
np.save(save_dir+"P110/P110_array-z{z}_optiresum-{o}.npy".format(z=redshift, o=optiresum), P110_array)
np.save(save_dir+"P112/P112_array-z{z}_optiresum-{o}.npy".format(z=redshift, o=optiresum), P112_array)
np.save(save_dir+"Pct0/Pct0_array-z{z}_optiresum-{o}.npy".format(z=redshift, o=optiresum), Pct0_array)
np.save(save_dir+"Pct2/Pct2_array-z{z}_optiresum-{o}.npy".format(z=redshift, o=optiresum), Pct2_array)
