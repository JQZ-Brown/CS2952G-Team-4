'''
Description:
    Hi-C evaluation metrics.
'''

import numpy as np
import random
from scipy.sparse import csr_matrix
from skimage.metrics import structural_similarity, mean_squared_error, peak_signal_noise_ratio
from scipy.sparse.csgraph import laplacian
from scipy.stats import pearsonr
from numpy.linalg import eig
from numpy.linalg import multi_dot


def _preprocessing(mat1, mat2):
    '''
    Unify data format.
    :param mat1: A list of matrices (num of samples, num of features, num of features).
    :param mat2: Another list of matrices (num of samples, num of features, num of features).
    :return: mat1, mat2
    '''
    if mat1.shape[0] != mat2.shape[0]:
        raise ValueError("Mat1 {} and Mat2 {} have different number of samples.".format(mat1.shape, mat2.shape))
    # Outputs of HiCPlus and hicGAN have different shapes, convert them to the same shape
    if mat1.shape[1] == 1:
        mat1 = np.reshape(mat1, (mat1.shape[0], mat1.shape[2], mat1.shape[3]))
    if mat2.shape[1] == 1:
        mat2 = np.reshape(mat2, (mat2.shape[0], mat2.shape[2], mat2.shape[3]))
    if mat1.shape != mat2.shape:
        raise ValueError("Mat1 {} and Mat2 {} have different shapes.".format(mat1.shape, mat2.shape))
    return  mat1, mat2


def _normalize(mat):
    '''
    Normalize matrices.
    :param mat: A list of matrices (num of samples, num of features, num of features).
    :return: mat
    '''
    for i in range(mat.shape[0]):
        if np.nansum(mat[i, :]) > 0:
            mat[i,:] = mat[i, : ] / np.nansum(mat[i, :])
    return mat


def Spector(mat1, mat2, r = 20):
    '''
    Compute Spector measurement between two list of Hi-C matrices.
    :param mat1: A list of matrices (num of samples, num of features, num of features).
    :param mat2: Another list of matrices (num of samples, num of features, num of features).
    :param r: Parameter of Spector (default = r).
    :return: Spector measurements of all samples (num of samples, ).
    '''
    mat1, mat2 = _preprocessing(mat1, mat2)
    shape = mat1.shape
    num_of_samples = shape[0]
    metric_record = []
    for index in range(num_of_samples):
        mat1_sub = mat1[index]
        mat2_sub = mat2[index]
        # Compute Spector
        mat1_sub_laplacian = laplacian(mat1_sub)
        mat2_sub_laplacian = laplacian(mat2_sub)
        mat1_sub_eig_vals, mat1_sub_eig_vecs = eig(mat1_sub_laplacian)
        mat2_sub_eig_vals, mat2_sub_eig_vecs = eig(mat2_sub_laplacian)
        spector = np.sum([np.linalg.norm(mat1_sub_eig_vecs[index]-mat2_sub_eig_vecs[index]) for index in range(r-1)])
        spector = 1 - (1/r) * (spector/np.sqrt(2))
        metric_record.append(spector)
    return metric_record


def GenomeDISCO(mat1, mat2, t = 5):
    '''
    Compute GenomeDISCO measurement between two list of Hi-C matrices.
    :param mat1: A list of matrices (num of samples, num of features, num of features).
    :param mat2: Another list of matrices (num of samples, num of features, num of features).
    :param r: Parameter of Spector (default = r).
    :return: Spector measurements of all samples (num of samples, ).
    '''
    mat1, mat2 = _preprocessing(mat1, mat2)
    shape = mat1.shape
    num_of_samples = shape[0]
    metric_record = []
    for index in range(num_of_samples):
        # Normalization is necessary or GenomeDISCO
        mat1_sub = _normalize(mat1[index])
        mat2_sub = _normalize(mat2[index])
        # Multiple t times
        if t > 1:
            mat1_sub_t = multi_dot([mat1_sub for _ in range(t)])
            mat2_sub_t = multi_dot([mat2_sub for _ in range(t)])
        else:
            mat1_sub_t = mat1_sub
            mat2_sub_t = mat2_sub
        # Compute GenomeDISCO
        mat1_non_zero = np.sum(np.sum(mat1_sub, axis = 1) > 0)
        mat2_non_zero = np.sum(np.sum(mat2_sub, axis=1) > 0)
        genome_disco = np.sum(np.abs(mat1_sub_t-mat2_sub_t)) / (0.5 *(mat1_non_zero + mat2_non_zero))
        metric_record.append(genome_disco)
    return metric_record


def HiCRep(mat1, mat2):
    '''
    Compute HiCRep measurement between two list of Hi-C matrices.
    :param mat1: A list of matrices (num of samples, num of features, num of features).
    :param mat2: Another list of matrices (num of samples, num of features, num of features).
    :param r: Parameter of Spector (default = r).
    :return: Spector measurements of all samples (num of samples, ).
    '''
    # For now, assume the distance shard is the same as bin size
    mat1, mat2 = _preprocessing(mat1, mat2)
    shape = mat1.shape
    num_of_samples = shape[0]
    metric_record = []
    for index in range(num_of_samples):
        mat1_sub = mat1[index]
        mat2_sub = mat2[index]
        K_index = list(range(shape[1]))
        mat1_K = [[] for _ in K_index]
        mat2_K = [[] for _ in K_index]
        Nk = [0 for _ in K_index]
        # distance shards
        for i in range(0, shape[1]):
            for j in range(i, shape[1]):
                index = K_index.index(j - i)
                if mat1_sub[i, j] == 0 and mat2_sub[i, j] == 0:
                    continue
                mat1_K[index].append(mat1_sub[i, j])
                mat2_K[index].append(mat2_sub[i, j])
                Nk[index] += 1
        mat1_K = [np.array(each) for each in mat1_K]
        mat2_K = [np.array(each) for each in mat2_K]
        temp_mat1 = []
        temp_mat2 = []
        temp_Nk = []
        for index in range(len(Nk)):
            if (len(mat1_K[index]) != 0 and len(mat2_K[index]) != 0):
                temp_mat1.append(mat1_K[index])
                temp_mat2.append(mat2_K[index])
                temp_Nk.append(Nk[index])
        mat1_K = temp_mat1
        mat2_K = temp_mat2
        Nk = np.array(temp_Nk)
        # compute rho for every shard
        r1 = [np.nanmean(mat1_K[k]*mat2_K[k])-np.nanmean(mat1_K[k])*np.nanmean(mat2_K[k]) for k in range(len(mat1_K))]
        r1 = [each if not np.isnan(each) else 0.0 for each in r1]
        r2 = [np.sqrt(np.nanvar(mat1_K[k]) * np.nanvar(mat2_K[k])) for k in range(len(mat1_K))]
        r2 = [each if not np.isnan(each) else 0.0 for each in r2]
        rho = np.array(r1) / np.array(r2)
        rho = np.nan_to_num(rho)
        weight = r2*Nk / np.sum(r2*Nk)
        weighted_rho = weight * rho
        hic_rep = np.sum(weighted_rho)
        metric_record.append(hic_rep)
    return metric_record


def MSE(mat1, mat2):
    '''
    Compute MSE measurement between two list of Hi-C matrices.
    :param mat1: A list of matrices (num of samples, num of features, num of features).
    :param mat2: Another list of matrices (num of samples, num of features, num of features).
    :param r: Parameter of Spector (default = r).
    :return: Spector measurements of all samples (num of samples, ).
    '''
    mat1, mat2 = _preprocessing(mat1, mat2)
    shape = mat1.shape
    num_of_samples = shape[0]
    metric_record = []
    for index in range(num_of_samples):
        mat1_sub = mat1[index]
        mat2_sub = mat2[index]
        metric_record.append(mean_squared_error(mat1_sub, mat2_sub))
    return metric_record


# def SSIM(mat1, mat2):
#     raise NotImplementedError("Some problems with calculating the SSIM.")
#     mat1, mat2 = _preprocessing(mat1, mat2)
#     shape = mat1.shape
#     num_of_samples = shape[0]
#     metric_record = []
#     for index in range(num_of_samples):
#         mat1_sub = mat1[index]
#         mat2_sub = mat2[index]
#         MAX = np.max(mat1_sub) - np.min(mat1_sub)
#         C1 = (0.01 * MAX)**2
#         C2 = (0.01 * MAX) ** 2
#         mat1_mean = np.mean(mat1_sub)
#         mat2_mean = np.mean(mat2_sub)
#         mat1_sigma = np.std(mat1_sub)
#         mat2_sigma = np.std(mat2_sub)
#         sigma = np.cov(mat1_sub, mat2_sub)
#         # ssim = ((2*mat1_mean*mat2_mean + C1) * (2*sigma + C2)) / ((mat1_mean**2 + mat2_mean**2 + C1) * (mat1_sigma**2 + mat2_sigma**2 + C2))
#         ssim = structural_similarity(mat1_sub, mat2_sub, data_range= mat1_sub.max() - mat1_sub.min())
#         metric_record.append(ssim)
#     return metric_record


def PSNR(mat1, mat2):
    '''
    Compute PSNR measurement between two list of Hi-C matrices.
    :param mat1: A list of matrices (num of samples, num of features, num of features).
    :param mat2: Another list of matrices (num of samples, num of features, num of features).
    :param r: Parameter of Spector (default = r).
    :return: Spector measurements of all samples (num of samples, ).
    '''
    # Mat1 should be true high-res data while mat2 should be our stimations
    mat1, mat2 = _preprocessing(mat1, mat2)
    shape = mat1.shape
    num_of_samples = shape[0]
    metric_record = []
    for index in range(num_of_samples):
        mat1_sub = mat1[index]
        mat2_sub = mat2[index]
        metric_record.append(peak_signal_noise_ratio(mat1_sub, mat2_sub, data_range=mat1_sub.max() - mat1_sub.min()))
    return metric_record


def PearsonCorre(mat1, mat2):
    '''
    Compute Pearson correlation measurement between two list of Hi-C matrices.
    :param mat1: A list of matrices (num of samples, num of features, num of features).
    :param mat2: Another list of matrices (num of samples, num of features, num of features).
    :param r: Parameter of Spector (default = r).
    :return: Spector measurements of all samples (num of samples, ).
    '''
    # For now, assume the distance shard is the same as bin size
    mat1, mat2 = _preprocessing(mat1, mat2)
    shape = mat1.shape
    num_of_samples = shape[0]
    metric_record = []
    for index in range(num_of_samples):
        mat1_sub = mat1[index]
        mat2_sub = mat2[index]
        K_index = list(range(shape[1]))
        mat1_K = [[] for _ in K_index]
        mat2_K = [[] for _ in K_index]
        Nk = [0 for _ in K_index]
        # distance shards
        for i in range(0, shape[1]):
            for j in range(i, shape[1]):
                index = K_index.index(j - i)
                mat1_K[index].append(mat1_sub[i, j])
                mat2_K[index].append(mat2_sub[i, j])
                Nk[index] += 1
        mat1_K = [np.array(each) for each in mat1_K]
        mat2_K = [np.array(each) for each in mat2_K]
        Nk = np.array(Nk)
        # compute Pearson correlation for every shard
        correlation = [pearsonr(mat1_K[k], mat2_K[k])[0] if Nk[k] > 1 else np.nan for k in range(len(K_index))]
        # mat1_sub = mat1[index].reshape((shape[1]*shape[2], ))
        # mat2_sub = mat2[index].reshape((shape[1]*shape[2], ))
        # correlation = pearsonr(mat1_sub, mat2_sub)[0]
        metric_record.append(correlation)
    return np.array(metric_record)


def PearsonCorreAll(mat1, mat2):
    '''
    Compute Pearson correlation measurement between two list of Hi-C matrices.
    :param mat1: A list of matrices (num of samples, num of features, num of features).
    :param mat2: Another list of matrices (num of samples, num of features, num of features).
    :param r: Parameter of Spector (default = r).
    :return: Spector measurements of all samples (num of samples, ).
    '''
    # For now, assume the distance shard is the same as bin size
    mat1, mat2 = _preprocessing(mat1, mat2)
    shape = mat1.shape
    num_of_samples = shape[0]
    metric_record = []
    for index in range(num_of_samples):
        mat1_sub = mat1[index]
        mat2_sub = mat2[index]
        # compute Pearson correlation for every shard
        correlation = pearsonr(mat1_sub.reshape((shape[1]*shape[2],)), mat2_sub.reshape((shape[1]*shape[2],)))[0]
        # mat1_sub = mat1[index].reshape((shape[1]*shape[2], ))
        # mat2_sub = mat2[index].reshape((shape[1]*shape[2], ))
        # correlation = pearsonr(mat1_sub, mat2_sub)[0]
        metric_record.append(correlation)
    return np.array(metric_record)


# For test. Ignore this function
def _extractData():
    def matrix_extract(chrN1, chrN2, binsize, hicfile):
        import straw
        result = straw.straw('NONE', hicfile, str(chrN1), str(chrN2), 'BP', binsize)

        row = [r // binsize for r in result[0]]
        col = [c // binsize for c in result[1]]
        value = result[2]
        Nrow = max(row) + 1
        Ncol = max(col) + 1
        N = max(Nrow, Ncol)

        # print(N)
        M = csr_matrix((value, (row, col)), shape=(N, N))
        M = csr_matrix.todense(M)

        return (M)

    def train_divide(HiCmatrix):
        subImage_size = 40
        step = 25
        result = []
        index = []
        # chrN = 21  ##need to change.

        total_loci = HiCmatrix.shape[0]
        # print(HiCmatrix.shape)
        for i in range(0, total_loci, step):
            for j in range(0, total_loci, ):
                if (abs(i - j) > 201 or i + subImage_size >= total_loci or j + subImage_size >= total_loci):
                    continue
                subImage = HiCmatrix[i:i + subImage_size, j:j + subImage_size]

                result.append([subImage, ])
                tag = 'test'
                index.append((tag, i, j))
        result = np.array(result)
        # print(result.shape)
        result = result.astype(np.double)
        index = np.array(index)
        return result, index

    def genDownsample(original_sample, rate):
        print('here')
        print(original_sample.shape)
        print(original_sample[0, 0].shape)
        result = np.zeros(original_sample.shape).astype(float)
        for i in range(0, original_sample.shape[0]):
            for j in range(0, original_sample.shape[1]):
                for k in range(0, int(original_sample[i, j])):
                    if (random.random() < rate):
                        result[i][j] += 1
        return result


    inputfile = "/home/jiaqi/python/HiC/data/GSE63525_K562_combined.hic"
    chromosome = 19
    scalerate = 40

    highres = matrix_extract(chromosome, chromosome, 10000, inputfile)
    lowres = genDownsample(highres, 1 / float(scalerate))
    print("High res shape : ", highres.shape)
    print("Low res shape : ", lowres.shape)
    np.save("high_res.npy", highres)
    np.save("low_res.npy", lowres)

    print('dividing, filtering and downsampling files...', flush=True)

    highres_sub, _ = train_divide(highres)
    lowres_sub, _ = train_divide(lowres)
    np.save("high_res_sub.npy", highres_sub)
    np.save("low_res_sub.npy", highres_sub)
    print("High res sub shape : ", highres_sub.shape)
    print("Low res sub shape : ", lowres_sub.shape)

    print(highres_sub[0])
    print(lowres_sub[0])

    np.save("high_res_example.npy", highres_sub[0:5])
    np.save("low_res_example.npy", lowres_sub[0:5])

    np.savez("contact_matrix", highres=highres, lowres=lowres, highres_sub=highres_sub, lowres_sub=lowres_sub)


if __name__ == '__main__':
    # For now
    # _extractData()

    highres_sub = np.load("high_res_example.npy")
    lowres_sub = np.load("low_res_example.npy")
    print("High res sub shape : ", highres_sub.shape)
    print("Low res sub shape : ", lowres_sub.shape)

    spector = Spector(highres_sub, lowres_sub, r = 20)
    genome_disco = GenomeDISCO(highres_sub, lowres_sub, t = 1)
    hic_rep = HiCRep(highres_sub, lowres_sub)
    mse = MSE(highres_sub, lowres_sub)
    psnr = PSNR(highres_sub, lowres_sub)
    pearson = PearsonCorre(highres_sub, lowres_sub)
    person_all = PearsonCorreAll(highres_sub, lowres_sub)
    # Do not use SSIM for now.
    # ssim = SSIM(highres_sub, lowres_sub)

    print("Spector : ", spector)
    print("GenomeDISCO : ", genome_disco)
    print("HiCRep : ", hic_rep)
    print("MSE : ", mse)
    print("PSNR : ", psnr)
    print("Pearson Correlation : ", np.nanmean(pearson,axis=1))
    print("Pearson Correlation All : ", person_all)