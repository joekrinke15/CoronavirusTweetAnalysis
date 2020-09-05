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