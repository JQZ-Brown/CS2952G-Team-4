import numpy as np
import os,struct
import random
import straw
from scipy.sparse import csr_matrix, coo_matrix, vstack, hstack
from scipy import sparse
import numpy as np


# Taken from hicplus repo
# See hicplus directory or https://github.com/wangjuan001/hicplus 
# Bugfixes in train_divide and donwsampling due to Tyler

def matrix_extract(chrN1, chrN2, binsize, hicfile):

    result = straw.straw('NONE', hicfile, str(chrN1),str(chrN2),'BP',binsize)

    row = [r//binsize for r in result[0]]
    col = [c//binsize for c in result[1]]
    value = result[2]
    Nrow = max(row) + 1
    Ncol = max(col) + 1
    N = max(Nrow, Ncol)

    M = csr_matrix((value, (row,col)), shape=(N,N))
    M = csr_matrix.todense(M)

    return(M)

def train_divide(HiCmatrix):
    subImage_size = 40
    step = 25
    result = []
    index = []

    total_loci = HiCmatrix.shape[0]
    #print(HiCmatrix.shape)
    for i in range(0, total_loci, step):
        for j in range(0, total_loci, ):
            if (abs(i-j)>201 or i + subImage_size >= total_loci or j + subImage_size >= total_loci):
                continue
            subImage = HiCmatrix[i:i + subImage_size, j:j + subImage_size]

            result.append([subImage, ])
            tag = 'test'
            index.append((tag, i, j))
    result = np.array(result)
    #print(result.shape)
    result = result.astype(np.double)
    index = np.array(index)
    return result, index

# Downsampling technique taken from HiCPlus
# Downsamples to a lower res by randomly dropping our reads with probability rate
def genDownsample(original_sample, rate):
    result = np.zeros(original_sample.shape).astype(float)
    for i in range(0, original_sample.shape[0]):
        for j in range(0, original_sample.shape[1]):
            for k in range(0, int(original_sample[i,j])):
                if (random.random() < rate):
                    result[i][j] += 1
    return result


if __name__ == "__main__":
    main()
