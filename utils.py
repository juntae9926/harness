import numpy as np
import cv2

from typing import NamedTuple
from scipy.signal import convolve2d
from skimage import measure

class Distribution(NamedTuple):
    mean: np.ndarray
    cov: np.ndarray


def minmax_norm(arr):
    M, m = np.max(arr), np.min(arr)

    return (arr - m) / (M - m)


def gaussian_density(data, gaussian):
    d = data.shape[1]
    sigma = np.diag(np.diag(gaussian.cov))  # 2d-diag
    denominator = ((2 * 3.14) ** (d / 2)) * np.linalg.det(sigma) ** 0.5
    inv_sigma = np.linalg.inv(sigma)
    diff = data - gaussian.mean
    results = np.exp(np.sum(-0.5 * diff * np.diag(inv_sigma) * diff, axis=1)) / denominator

    return results


# split data into k subsample and get gaussian density
def get_gaussian_density(data, distribution, k=1):
    density = []
    for _data in np.array_split(data, k):
        density.extend(gaussian_density(_data, distribution))

    return np.array(density)


# split data into k subsample and get mean, cov
def get_k_gaussian(data, k=1):

    return [Distribution(np.mean(_data, axis=0), np.cov(_data, rowvar=False)) for _data in np.array_split(data, k)]


def get_mask(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_orange = np.array([0, 150, 150])
    high_orange = np.array([5, 255, 255])
    mask = cv2.inRange(hsv_img, low_orange, high_orange)

    return mask


def train_gmm(path, train_scale=0.35, target_threshold=200):
    img = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
    height, width, _ = img.shape
    img = cv2.resize(img, [int(round(width * train_scale)), int(round(height * train_scale))])

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) / 255.
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB) / 255.

    target = cv2.imread(path.with_suffix('.tif').as_posix(), cv2.IMREAD_GRAYSCALE)
    if target is None:
        target = cv2.imread(path.with_suffix('.tiff').as_posix(), cv2.IMREAD_GRAYSCALE)
    target = cv2.resize(target, [int(round(width * train_scale)), int(round(height * train_scale))])

    t_idx = np.where(target < target_threshold)
    bg_idx = np.where(target == 255)

    return rgb[t_idx], hsv[t_idx], ycrcb[t_idx], rgb[bg_idx], hsv[bg_idx], ycrcb[bg_idx]

def scaling(path, test_scale):
    img = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
    height, width, _ = img.shape
    img = cv2.resize(img, [int(round(width * test_scale)), int(round(height * test_scale))], interpolation=cv2.INTER_AREA)
    
    return img

def frame_scaling(img, test_scale, ratio=None):
    height, width, _ = img.shape
    img = cv2.resize(img, [int(round(width * test_scale)), int(round(height * test_scale))], interpolation=cv2.INTER_AREA)
    if ratio:
        img = img[int(height*(ratio[0]-0.05)):int(height*(ratio[0]+0.1)),int(width*0.2):int(width*0.8)]
        
    return img


def test_gmm(i, img, gaussians, pixel_threshold=175/255, correct_threshold=1, K=10):

    height, width, _ = img.shape
    img = img[int(height*0.3):int(height*0.7),int(width*0.2):int(width*0.8)]
    height, width, _ = img.shape

    # threshold by count(num of high value pixel), max = 30 (3*10 : k-distributions, rgb,hsv,ycrcb)
    correct_threshold = correct_threshold

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).reshape(-1, 3) / 255.
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB).reshape(-1, 3) / 255.   

    # get Prob
    correct = np.zeros((height, width), dtype=np.uint8)
    for j in range(K):
        t_rgb_density = get_gaussian_density(rgb, gaussians["t_rgb"][j], k=K).reshape(height, width)
        bg_rgb_density = get_gaussian_density(rgb, gaussians["bg_rgb"][j], k=K).reshape(height, width)

        t_hsv_density = get_gaussian_density(hsv, gaussians["t_hsv"][j], k=K).reshape(height, width)
        bg_hsv_density = get_gaussian_density(hsv, gaussians["bg_hsv"][j], k=K).reshape(height, width)

        t_ycrcb_density = get_gaussian_density(ycrcb, gaussians["t_ycrcb"][j], k=K).reshape(height, width)
        bg_ycrcb_density = get_gaussian_density(ycrcb, gaussians["bg_ycrcb"][j], k=K).reshape(height, width)

        rgb_prob = t_rgb_density / (t_rgb_density + bg_rgb_density + 1 / 256)
        rgb_prob_norm = minmax_norm(rgb_prob)
        correct += rgb_prob_norm > pixel_threshold

        hsv_prob = t_hsv_density / (t_hsv_density + bg_hsv_density + 1 / 256)
        hsv_prob_norm = minmax_norm(hsv_prob)
        correct += hsv_prob_norm > pixel_threshold

        ycrcb_prob = t_ycrcb_density / (t_ycrcb_density + bg_ycrcb_density + 1 / 256)
        ycrcb_prob_norm = minmax_norm(ycrcb_prob)
        correct += ycrcb_prob_norm > pixel_threshold 

        target = correct > correct_threshold
    
    return img, target


def blob_detection(target, blur_threshold=15, small_threshold=100, flag=False):
    kernel1d = cv2.getGaussianKernel(3, 2.1)  # original 21, 15
    gaussian1 = np.outer(kernel1d, kernel1d.transpose())
    blob = convolve2d(target, gaussian1, mode='same') * 255 

    binary = (blob > blur_threshold).astype(np.uint8)
    labeled = measure.label(binary, connectivity= 2) # (up, down, cross) check connectivity
    blob_properties = measure.regionprops(labeled, blob) # linked pixel count
    
    mask = np.zeros_like(binary, np.uint8)
    coord_list = list()

    wire = list()
    for prop in blob_properties:
        if (flag == True) and (len(prop.coords) > 1000):
            temp = list()
            for r, c in prop.coords:
                wire.append([r, c])

        if len(prop.coords) > small_threshold:
            for r, c in prop.coords:
                mask[r, c] = 1
                coord_list.append([r, c])
    if flag:
        return blob, binary, mask, np.array(coord_list), wire
    else:
        return blob, binary, mask, np.array(coord_list)