# functions to io things
import numpy as np
import pandas as pd
import h5py
import os
import tarfile
import gzip
import shutil
import random

from tqdm.notebook import tqdm
from datetime import datetime
from methods.data_proc import downscale_to_shape

ROCKS = pd.read_excel('rock_properties.xlsx', index_col=0)

def del_folder(mydir):
    '''
    deleting temp folder
    '''
    try:
        shutil.rmtree(mydir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror)) 


def extract_tar_to_temp_folder(tar_path):
    '''
    Extract tar files to temp folder
    '''
    with tarfile.open(tar_path) as tar:
        for t in tar:
            if t.isreg():
                if t.name.split('.')[-2] in ['g00', 'g12']:
                    tar.extract(t.name)
            elif t.isdir():
                gz_root_dirname = t.name
    
    return gz_root_dirname


def get_labels_from_g12(g12_path):
    '''
    reading g12.gz file with labels
    '''
    with gzip.open(g12_path, 'r') as f:
        rock_labels = np.loadtxt(f, skiprows=0).astype('int8')
        rock_labels = rock_labels.reshape((200, 200, 200))
        rock_labels = np.transpose(rock_labels, (1, 2, 0))
    
    return rock_labels


def get_rocks_from_g00(g00_path):
    '''
    reading g00.gz file with rock type information
    '''
    with gzip.open(g00_path, 'r') as f:
        rock_dict = dict()
        for line in f:
            if "ROCK DEFINITION" in str(line):
                rock_type = str(line).split(" ")[2][0:-3]
                rock_label = int(str(line).split(" ")[-1][0:-3])
                rock_dict[rock_label] = {'type': rock_type}

            if "Density" in str(line):
                rock_dens = float(str(line).split(" ")[-1][0:-3])
                rock_dict[rock_label]['dens'] = rock_dens
    
    return rock_dict


def random_perm_1(log_mean, log_sigma):
    '''
    from logmean and logsigma
    '''
    return 10**(log_sigma * np.random.randn() + log_mean)


def random_perm_2(log_min, log_max):
    '''
    from genetic class
    '''
    return 10**(log_min + np.random.rand()*(log_max - log_min))


def add_random_perm_by_rock_type(some_rock_dict):
    '''
    add perm to rock dict
    '''
    for key in some_rock_dict.keys():
        rock_type = some_rock_dict[key]['type']
        log_perm_mean, log_perm_sigma = ROCKS.loc[rock_type, 'Mean permeability (Log10(mD))'], ROCKS.loc[rock_type, 'Std permeability']
        perm = random_perm_1(log_perm_mean, log_perm_sigma)
              
    #     GEN_CLASS_PERM_LOG_MIN_MAX = {
    #         'intrusive' : [-1, 0],
    #         'metamorphic' : [0, 1],
    #         'volcanic' : [1, 2],
    #         'sedimentary' : [2, 3]
    # }
    #     gen_class = ROCKS[rock_type]['genetic_class']
    #     log_min, log_max = GEN_CLASS_PERM_LOG_MIN_MAX[gen_class]
    #     perm = random_perm_2(log_min, log_max)
        
        some_rock_dict[key]['perm'] = perm
     
    return some_rock_dict


def map_labels_with_some_prop(labels, rock_dict, prop):
    '''
    prop is 'perm' or 'dens' - property keys from rock_dict 
    '''
    model = np.zeros_like(labels, dtype='float16') 
    for key in rock_dict.keys():
        model[labels==key] = rock_dict[key][prop]
    
    return model


def get_random_models(models_list, nmodels):
    '''
    choosing random nmodels from list
    '''
    if len(models_list) > nmodels:
        short_random_list = random.sample(models_list, nmodels)
    else:
        short_random_list = models_list

    return short_random_list


def write_to_h5dataset(idx, data_to_write, h5dataset):
    '''
    write to h5 with file resizing
    '''
    if idx >= h5dataset.shape[0]:
        h5dataset.resize(h5dataset.shape[0]+1, axis=0)
    h5dataset[idx] = data_to_write


def add_data_to_h5dataset(fname, idx, data, set_name):
    with h5py.File(fname, 'a') as f:
        if set_name not in f.keys():
            max_len  = np.max([f[k].shape[0] for k in f.keys()])
            ev_set = f.create_dataset(set_name,  (max_len,) + data.shape, dtype='float16') # creating dataset with size like maximum other set size in the file

        f[set_name][idx] = data


def tar_to_downscaled_models(path_to_models, nmodels, target_shape: tuple, fname='downscaled_models'):
    '''
    folder with tar files processing
    tar with .gz ---> h5 file with downscaled models 
    '''
    CURR_DIR = os.getcwd()
    yield_path = f'{CURR_DIR}/{fname}_{datetime.now().strftime("%m_%d_%Y__%H_%M_%S")}.h5'
    with h5py.File(yield_path, 'w') as targ:
        perm_h5_set = targ.create_dataset("perm",  (1,)+target_shape, dtype='float16', maxshape=(None,)+target_shape) # resized every iteration
        dens_h5_set = targ.create_dataset("dens",  (1,)+target_shape, dtype='float16', maxshape=(None,)+target_shape)     
        idx = int(0)

        tar_list = os.listdir(path_to_models)
        for tar in tqdm(tar_list):
            tar_path = f'{path_to_models}{tar}'
            print(f'Extracting ==> {tar_path}')
            gz_root_dirname = extract_tar_to_temp_folder(tar_path) # extracting
            gz_list = os.listdir(gz_root_dirname)
            g00_path_list = [f'{gz_root_dirname}/{gz}' for gz in gz_list if 'g00' in gz] # paths to all g00 files with rock properties
            g12_path_list = [f'{gz_root_dirname}/{gz}' for gz in gz_list if 'g12' in gz] # paths to all g12 files with labels
            print(f'Done')

            models_in_tar_zipped_list = list(zip(g00_path_list, g12_path_list))
            # models_random_short_list = get_random_models(models_in_tar_zipped_list, nmodels)
            models_random_short_list = models_in_tar_zipped_list[0:nmodels]
                
            for g00_path, g12_path in tqdm(models_random_short_list):          
                try:
                    rock_dict = get_rocks_from_g00(g00_path)
                    labels = get_labels_from_g12(g12_path)
                    rock_dict = add_random_perm_by_rock_type(rock_dict) # add permeability
                    perm_model = map_labels_with_some_prop(labels, rock_dict, 'perm') # permrability model mD
                    dens_model = map_labels_with_some_prop(labels, rock_dict, 'dens') # density model g/cm^3

                    if np.min(perm_model) > 0 and np.min(dens_model) > 0:

                        perm_model = downscale_to_shape(perm_model, target_shape) # downscaling
                        dens_model = downscale_to_shape(dens_model, target_shape)

                        write_to_h5dataset(idx, perm_model, perm_h5_set) # writing
                        write_to_h5dataset(idx, dens_model, dens_h5_set)

                        name = g00_path.split('.')[0] # names of initial files just in case for future needs
                        idx += 1
                        
                    else:
                        print(f'something wrong with data in {g00_path.split('.')[0]}')

                except:
                    print(f'Erron while reading {g00_path.split('.')[0]}')
                    continue
                
            del_folder(gz_root_dirname.split('/')[0]) #clear it

    return yield_path


def check_tar_integrity(folder_path):
  """
  Checks the integrity of all tar files in a folder with progress bar.

  Args:
    folder_path: The path to the folder containing the tar files.
  """
  total_files = sum(1 for filename in os.listdir(folder_path) if filename.endswith(".tar"))
  pbar = tqdm(total=total_files, desc="Checking tar integrity")

  for filename in os.listdir(folder_path):
    if not filename.endswith(".tar"):
      continue

    filepath = os.path.join(folder_path, filename)
    try:
      # Open the tar file in read mode
      with tarfile.open(filepath, "r") as tar:
        # Iterate through each member (file) in the archive
        for member in tar:
          # Check if member is a regular file (skip directories)
          if not member.isfile():
            continue
          # Try to get member information (reading header only)
          try:
            tar.getmember(member.name)
          except (tarfile.ReadError, tarfile.HeaderError) as e:
            print(f"Error: Integrity issue found in {filepath}: {e}")
          break
    except tarfile.ReadError as e:
      print(f"Error: Could not open {filepath}: {e}")
    finally:
      pbar.update(1)  # Update progress bar after each file

  pbar.close()  # Close progress bar
