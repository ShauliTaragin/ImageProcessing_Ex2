import math
import numpy as np
import cv2

eps = 0.004


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    # Working by convention i shall first flip the kernel
    ker_fliped = np.zeros_like(k_size)
    for i in range(len(k_size)):
        ker_fliped[-1 - i] = k_size[i]

    # pad the kernels with zeros
    side_padding_len = len(in_signal) - 1
    padded_ker = np.zeros(len(ker_fliped) + 2 * side_padding_len)
    rng = len(padded_ker) - side_padding_len
    k = 0
    for i in range(side_padding_len, rng):
        padded_ker[i] = ker_fliped[k]
        k += 1

    # create vec to return
    vec_2_return = np.zeros(len(in_signal) + len(ker_fliped) - 1)

    # multiply the values
    k = 0
    for i in range(len(vec_2_return)):
        for j in range(len(in_signal)):
            vec_2_return[i] += in_signal[-1 - j] * padded_ker[-1 - k - j]
        k += 1
    return vec_2_return


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    # Working by convention i shall first flip the kernel
    ker_flipped = np.flip(kernel)
    # pad the picture
    padding_len = [int(ker_flipped.shape[0] / 2), int(ker_flipped.shape[1] / 2)]
    padded_signal = cv2.copyMakeBorder(in_image, padding_len[0], padding_len[0], padding_len[1], padding_len[1],
                                       cv2.BORDER_REPLICATE, None, value=0)
    # create img to return
    pic_to_return = np.zeros_like(in_image)
    # multiply the values
    for i in range(pic_to_return.shape[0]):
        for j in range(pic_to_return.shape[1]):
            pic_to_return[i, j] = (padded_signal[i:i + ker_flipped.shape[0],
                                   j:j + ker_flipped.shape[1]] * ker_flipped).sum().round()
    return pic_to_return


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    vector = np.array([[1, 0, -1]])
    G_X = cv2.filter2D(in_image, -1, vector)
    G_Y = cv2.filter2D(in_image, -1, vector.T)
    magnitude = np.sqrt(G_X ** 2 + G_Y ** 2).astype(np.float64)
    direction = np.arctan2(G_Y, G_X).astype(np.float64)
    return direction, magnitude


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    # sigma from the Gaussian blurring formula
    sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
    # create the kernel to work on. 2d array of size k_size*k_size
    kernel = np.empty(shape=(k_size, k_size))
    kernel.fill(0)
    # sqrt variable to make code neat
    sqrt_k_size = np.sqrt(k_size)
    for i in range(0, k_size):
        for j in range(0, k_size):
            x_plus_y_sqrd = ((i - sqrt_k_size) ** 2 + (j - sqrt_k_size) ** 2)
            kernel[i][j] = np.exp(-x_plus_y_sqrd / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    return conv2D(in_image, kernel)


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    kernel = cv2.getGaussianKernel(k_size, 0)
    return cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)


def create_neighbor_array(laplacian_pic: np.ndarray, r: int, c: int) -> (np.ndarray, np.ndarray):
    up = np.array([laplacian_pic[r, c - 1], laplacian_pic[r - 1, c - 1],
                   laplacian_pic[r - 1, c], laplacian_pic[r - 1, c + 1]])
    down = np.array([laplacian_pic[r, c + 1], laplacian_pic[r + 1, c + 1], laplacian_pic[r + 1, c],
                     laplacian_pic[r + 1, c - 1]])
    return up, down


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using the "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    # create laplacian array
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    laplacian_pic = cv2.filter2D(img, -1, laplacian, borderType=cv2.BORDER_REPLICATE)
    laplacian_pic[np.abs(laplacian_pic) < eps] = 0
    # the edge image which we return
    edg_img = np.zeros(img.shape)
    # iterating all pixels in image and determining the edges
    for i in range(img.shape[0] - (laplacian.shape[0] - 1)):
        for j in range(img.shape[1] - (laplacian.shape[1] - 1)):
            # now we have 3 cases , if pixel is zero then we check all his neighbours
            # else if its positive , and last case for negative
            if laplacian_pic[i][j] == 0:
                # check all neighbours and determine whether that pixel is an edge
                if (laplacian_pic[i - 1][j] < 0 and laplacian_pic[i + 1][j] > 0) or (
                        laplacian_pic[i - 1][j] > 0 and laplacian_pic[i + 1][j] < 0):
                    edg_img[i][j] = 1
                elif (laplacian_pic[i][j - 1] < 0 and laplacian_pic[i][j + 1] > 0) or (
                        laplacian_pic[i][j - 1] < 0 and laplacian_pic[i][j + 1] < 0):
                    edg_img[i][j] = 1
                else:
                    continue
            elif laplacian_pic[i][j] > 0:
                up, down = create_neighbor_array(laplacian_pic, i, j)
                if ((up < 0).sum() + (down < 0).sum()) > 0:
                    edg_img[i][j] = 1
            else:
                if (laplacian_pic[i][j - 1] > 0) or (laplacian_pic[i][j + 1] > 0) or (laplacian_pic[i - 1][j] > 0) or (
                        laplacian_pic[i + 1][j] > 0):
                    edg_img[i][j] = 1
    return edg_img


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges usint "ZeroCrossingLOG" method
    :param img: Input image
    :return: opencv solution, my implementation
    """

    img_gaus = cv2.GaussianBlur(img, (5, 5), 0)
    return edgeDetectionZeroCrossingSimple(img_gaus)


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """

    return


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """

