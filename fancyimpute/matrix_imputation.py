'''
Dscription:
    Estimate high-resolution data with matrix imputation methods.
'''

from scipy.ndimage import gaussian_filter, maximum_filter
import numpy as np
import copy

def matrixFilter(lowres_sub, method = "gaussian"):
    '''
    Estimate high-resolution matrix with Gaussian filters / maximum filters.
    :param lowres_sub: A list of low-resolution samples (num of samples, num of features, num of features).
    :param method: "gaussian" or "maximum".
    :return: A list of high-resolution estimations (num of samples, num of features, num of features).
    '''
    highres_ests = []
    for each in lowres_sub:
        if "gaussian" == method:
            highres_ests.append(gaussian_filter(each, sigma = 1))
        elif "maximum" == method:
            highres_ests.append(maximum_filter(each, size = 2))
        else:
            raise NotImplementedError("Filter is not implemented for \"{}\" method.".format(method))
    return highres_ests


if __name__ == '__main__':
    import seaborn as sns
    import matplotlib.pyplot as plt

    highres_sub = np.load("../evaluation/high_res_example.npy")
    lowres_sub = np.load("../evaluation/low_res_example.npy")
    print("High res sub shape : ", highres_sub.shape)
    print("Low res sub shape : ", lowres_sub.shape)

    if highres_sub.shape[1] == 1:
        highres_sub = np.reshape(highres_sub, (highres_sub.shape[0], highres_sub.shape[2], highres_sub.shape[3]))
    if lowres_sub.shape[1] == 1:
        lowres_sub = np.reshape(lowres_sub, (lowres_sub.shape[0], lowres_sub.shape[2], lowres_sub.shape[3]))
    gaussian_estimations = matrixFilter(lowres_sub, method="gaussian")
    maximum_estimations = matrixFilter(lowres_sub, method="maximum")

    plt.subplot(1, 3, 1)
    sns.heatmap(highres_sub[1], cmap="binary")
    plt.subplot(1, 3, 2)
    sns.heatmap(gaussian_estimations[1], cmap="binary")
    plt.subplot(1, 3, 3)
    sns.heatmap(maximum_estimations[1], cmap="binary")
    plt.show()