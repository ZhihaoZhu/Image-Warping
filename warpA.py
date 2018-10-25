import numpy as np
import scipy.special as ss

def warp(im, A, output_shape):
    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor interpolation."""

    x = np.zeros((output_shape[0], output_shape[1]))
    for i in range(1, output_shape[0]):
        for j in range(1, output_shape[1]):
            M = np.linalg.inv(A).dot(np.array([i,j,1]))
            p, q = M[0], M[1]
            rp = int(round(p))
            rq = int(round(q))
            if rp<im.shape[0] and rp>=0 and rq<im.shape[1] and rq>=0:
                x[i, j] = im[rp, rq]

    return x