import math
import numpy as np

def cal_gaussian_kernel_at_ij(i,j,k,sigma):
    return (1./(2*math.pi*pow(sigma,2)))*math.exp(-(pow(i,2)+pow(j,2))/(2*pow(sigma,2)))

def cal_kernel(kernel_size=3, sigma=2.):
    kernel = np.ones((kernel_size, kernel_size))
    k = (kernel_size - 1) // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            print(-(kernel_size//2)+j, (kernel_size//2)-i)
            kernel[i,j] = cal_gaussian_kernel_at_ij(-(kernel_size//2)+j,(kernel_size//2)-i,k=k, sigma=sigma)

    kernel = kernel / np.sum(kernel)
    print(kernel)
    return kernel


cal_kernel(5,5)