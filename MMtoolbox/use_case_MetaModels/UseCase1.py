import numpy as np
import os
from MetaModels import MetaModelConstructor as MMC

""" Use Case 1 - Constructing a meta-model

"""

# Data paths to change directories
path_to_data = r'C:\Users\s119104\Documents\Studie\Stage Linköping\Metamodellering\data'
path_to_script = r'C:\Users\s119104\Documents\GitHub\Meta-modelling\MMtoolbox\use_case_MetaModels'


# Change to data path
os.chdir(path_to_data)

# Retrieve data
input_data = np.mat(np.fromfile('Parameters_used_longer_sampling.dat', sep=' '))
output_data = np.mat(np.fromfile('Simulated_output_longer_sampling.dat', sep=' '))

# Reprocess the data into the proper matrix
input_data = input_data.reshape(45168, 4)
output_data = output_data.reshape(45168, 46)

# Restore script path (not needed, but possible
os.chdir(path_to_script)

# Create three meta-models
PLSR_MM = MMC.construct_meta_model(input_data, output_data, {'type': 'PLSR'})
DLU_MM = MMC.construct_meta_model(input_data, output_data, {'type': 'DLU'})
Input_MM = MMC.construct_meta_model(input_data, output_data, {'type': 'PLSR', 'n_comp': 4, 'input': 'Polynomial'})


# Select PLSR with input for good results
MMC.save_meta_model(Input_MM, 'UseCaseMetaModel', False)