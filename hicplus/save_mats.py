from __future__ import print_function
import argparse as ap
from math import log10

#import torch
#import torch.nn as nn
#import torch.optim as optim
#from torch.autograd import Variable
#from torch.utils.data import DataLoader
from hicplus import utils
#import model
import argparse
from hicplus import trainConvNet
import numpy as np

chrs_length = [249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566]

#chrN = 21
#scale = 16

def main():
    
    inputfile = "../../GSE63525_K562_combined.hic"
    chromosome = 19
    scalerate = 40

    highres = utils.matrix_extract(chromosome, chromosome, 10000, inputfile)

    print('dividing, filtering and downsampling files...', flush=True)

    highres_sub, index = utils.train_divide(highres)

    print(highres_sub.shape)
    #np.save(infile+"highres",highres_sub)

    lowres = utils.genDownsample(highres,1/float(scalerate))
    lowres_sub,index = utils.train_divide(lowres)
    print(lowres_sub.shape)
    #np.save(infile+"lowres",lowres_sub)

    print('start training...', flush=True)
    #trainConvNet.train(lowres_sub,highres_sub,args.outmodel)
    # Reshape to match input expecations for hicgan
    lowres_sub = np.reshape(lowres_sub, (lowres_sub.shape[0], 40, 40, 1))
    highres_sub = np.reshape(highres_sub, (highres_sub.shape[0], 40, 40, 1))

    np.save("../lowres.npy", lowres_sub)
    np.save("../highres.npy", highres_sub)

    print('finished...', flush=True)

if __name__=='__main__':
  main()
