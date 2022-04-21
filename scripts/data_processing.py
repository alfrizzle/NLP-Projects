# unzipping 7z file
pip install py7zr
import py7zr

file = 'file_path'
with py7zr.SevenZipFile(file, mode='r', password='file_password') as z:
    z.extractall('outputh_path')

with open(file, 'r') as data_file:
    for line in data_file:
        data = line.split('.')
        print(data)

########################################################################

# function to create dataframe from files in directory structure

import os
import pandas as pd

def transform(path):
    '''
    Takes all files in a specified directory and creates a dataframe containing each sentence within each file as a row

    path: path to root directory
    '''

    # get list of files
    files - [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    # sort files by name
    file.sort()

    # build list of dataframes, adding "file_name" column
    dfs = [pd.read_csv(os.path.join(path, f), sep='\n', header-None).assign(file_name=f) for f in files]

    # add column with sentence numbers in each file
    for i in dfs:
        i.insert(0, 'sent_number', range(1, 1+len(i)))
    
    # concatenate dataframes into one dataframe
    df_combined = pd.concat(dfs, ignore_index=True)

    #rename text column
    df_combined = df_combined.rename({0:'text'}, axis=1)

    #reorder columns
    new_df = df_combined[['file_name', 'sent_number', 'text']]

    return new_df

########################################################################

# function to list all paths nested under a directory

import glob

def list_all_paths(path):
    '''
    returns list of all relevant directories we can run the transform function on

    path: path to root directory
    '''

    top_level_dirs = []
    for i in next(os.walk(root_dir))[1]:
        top_level_dirs.append(os.path.join(root_dir, i))

    all_path = []
    for i in top_level_dirs:
        path = glob(i + '/*/*', recursive=True)
        all_paths.extend(path)
    
    return list_all_paths

########################################################################

# function to turn all files under a directory into a single dataframe

from tqdm import tqdm

def list_to_df(list_name):
    '''
    returns single dataframe with rows representing a single sentence using all file paths in a given list

    list_name: list of all top paths withing a directory
    '''

    dfs = []
    for i in tqdm(list_name):
        df = transform(i)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    return combined_df

########################################################################

# implementation of all functions above

root_dir = 'root_directory_path'
top_paths = list_all_paths(root_dir)
df = list_to_df(top_paths)