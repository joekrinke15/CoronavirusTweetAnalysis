import pandas as pd
import gzip
import shutil
import random

#Read in a random sample of the file. 
#With this you get every nth line of the data. Filename is the filename. Sep and compression are as named.
def read_data(n, filename, compression, sep):
    df = pd.read_csv(
         filename,
         compression = compression,
         sep = sep,
         header=0, 
         skiprows=lambda i: i % n != 0 #Check if the row is a multiple of 1000. If so, read it in!
    )
    return(df)

#Read every 1000th row of data.
twitter_sample = read_data(1000, 'full_dataset.tsv.gz', 'gzip', '\t')

#Output the data as a new dataset.
twitter_sample.to_csv('sample_tweets.csv')