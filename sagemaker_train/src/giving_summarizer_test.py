import argparse
import pandas as pd
import datetime

import subprocess
import sys


####### method #1 on installing pip package #######

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
# install('transformers==4.9.2')


####### method #2 on installing pip package #######

# import os
# os.system('pip install transformers==4.9.2')

import transformers
from transformers import AutoTokenizer

def summarize_amounts(file):
    df = pd.read_csv(file)
    df['amount'] = df['amount'].apply(lambda x: x.replace('$', '').replace(',', '') 
                                      if isinstance(x, str) else x).astype(float)
    df = df.groupby('payment_method').agg({'amount': ['sum']})
    df.loc['total']= df.sum()
    return df

def main(args):
    # read and parse input csv file
    df = summarize_amounts(args.infile)
    
    # create variable for current datetime
    ct = datetime.datetime.now() 
    current_time = str(ct.now()).replace(":", "-").replace(" ", "-")[:19]
    
    # export parsed dataframe as csv to output path
    outFileName=f'giving_summary_{current_time}'+'.csv'
    df.to_csv(args.output_path + outFileName)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Parse donations export file from Planning Center")

    parser.add_argument("--infile", type=str, help="name of the input file csv")
    parser.add_argument("--output_path", default="./", 
                        type=str, help="name of the output path -- default path is current working directory")
                        
    # Parse command-line args and run main.
    args = parser.parse_args()
    main(args)
