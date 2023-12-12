import time
from pathlib import Path
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


if __name__ == '__main__':
    train_dir = Path('images/orange_data')
    train_images = sorted(train_dir.rglob('*.jpg'), key=lambda x: x.stem)

    test_dir = Path('images/test')
    test_images = sorted(test_dir.rglob('*.jpg'), key=lambda x: x.stem)

    K = 10
    train_scale = 0.35
    test_scale = 0.1

    target_threshold = 200
    pixel_threshold = 175 / 255  # threshold by pixel
    correct_threshold = 15  # threshold by count(num of high value pixel), max = 30 (3*10 : k-distributions, rgb,hsv,ycrcb)
    blur_threshold = 15  # threshold to remove blur
    small_threshold = 20

    ################################################
    # Read Train Images
    # : Get target/background RGB, HSV, YCrCb data
    ################################################
    t_rgb = []
    t_hsv = []
    t_ycrcb = []

    bg_rgb = []
    bg_hsv = []
    bg_ycrcb = []
    for path in train_images:
        img = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
        height, width, _ = img.shape
        img = cv2.resize(img, [int(round(width * train_scale)), int(round(height * train_scale))])

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) / 255.
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB) / 255.

        target = cv2.imread(path.with_suffix('.tif').as_posix(), cv2.IMREAD_GRAYSCALE)
        target = cv2.resize(target, [int(round(width * train_scale)), int(round(height * train_scale))])

        # target
        t_idx = np.where(target < target_threshold)
        t_rgb.extend(rgb[t_idx])
        t_hsv.extend(hsv[t_idx])
        t_ycrcb.extend(ycrcb[t_idx])

        # background
        bg_idx = np.where(target == 255)
        bg_rgb.extend(rgb[bg_idx])
        bg_hsv.extend(hsv[bg_idx])
        bg_ycrcb.extend(ycrcb[bg_idx])

    ################################################
    # Get K Gaussian Distributions
    # : split data into k subsamples and get mean, cov
    ################################################
    t_rgb_gdist = get_k_gaussian(t_rgb, K)
    t_hsv_gdist = get_k_gaussian(t_hsv, K)
    t_ycrcb_gdist = get_k_gaussian(t_ycrcb, K)

    bg_rgb_gdist = get_k_gaussian(bg_rgb, K)
    bg_hsv_gdist = get_k_gaussian(bg_hsv, K)
    bg_ycrcb_gdist = get_k_gaussian(bg_ycrcb, K)

    print('TEST')
    # Test
    for path in test_images:
        print(path)
        img = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
        height, width, _ = img.shape

        if test_scale < 1:
            img = cv2.resize(img, [int(round(width * test_scale)), int(round(height * test_scale))])
            height, width, _ = img.shape

        # get RGB, HSV, YCrCb
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).reshape(-1, 3) / 255.
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB).reshape(-1, 3) / 255.

        # get Prob
        correct = np.zeros((height, width), dtype=np.uint8)
        for j in range(K):
            t_rgb_density = get_gaussian_density(rgb, t_rgb_gdist[j], k=K).reshape(height, width)
            bg_rgb_density = get_gaussian_density(rgb, bg_rgb_gdist[j], k=K).reshape(height, width)

            t_hsv_density = get_gaussian_density(hsv, t_hsv_gdist[j], k=K).reshape(height, width)
            bg_hsv_density = get_gaussian_density(hsv, bg_hsv_gdist[j], k=K).reshape(height, width)

            t_ycrcb_density = get_gaussian_density(ycrcb, t_ycrcb_gdist[j], k=K).reshape(height, width)
            bg_ycrcb_density = get_gaussian_density(ycrcb, bg_ycrcb_gdist[j], k=K).reshape(height, width)

            # cv2.imshow('rgb_density', cv2.applyColorMap((t_rgb_density * 255).astype(np.uint8), cv2.COLORMAP_JET))
            # cv2.imshow('hsv_density', cv2.applyColorMap((t_hsv_density * 255).astype(np.uint8), cv2.COLORMAP_JET))
            # cv2.imshow('ycrcb_density', cv2.applyColorMap((t_ycrcb_density  * 255).astype(np.uint8), cv2.COLORMAP_JET))
            # cv2.waitKey()

            rgb_prob = t_rgb_density / (t_rgb_density + bg_rgb_density + 1 / 256)
            rgb_prob_norm = minmax_norm(rgb_prob)
            correct += rgb_prob_norm > pixel_threshold

            hsv_prob = t_hsv_density / (t_hsv_density + bg_hsv_density + 1 / 256)
            hsv_prob_norm = minmax_norm(hsv_prob)
            correct += hsv_prob_norm > pixel_threshold

            ycrcb_prob = t_ycrcb_density / (t_ycrcb_density + bg_ycrcb_density + 1 / 256)
            ycrcb_prob_norm = minmax_norm(ycrcb_prob)
            correct += hsv_prob_norm > pixel_threshold

            # cv2.imshow('rgb_prob', cv2.applyColorMap((rgb_prob * 255).astype(np.uint8), cv2.COLORMAP_JET))
            # cv2.imshow('hsv_prob', cv2.applyColorMap((hsv_prob * 255).astype(np.uint8), cv2.COLORMAP_JET))
            # cv2.imshow('ycrcb_prob', cv2.applyColorMap((ycrcb_prob * 255).astype(np.uint8), cv2.COLORMAP_JET))
            # cv2.waitKey()

        target = correct > correct_threshold
        # cv2.imshow('correct', cv2.applyColorMap((correct / 30).astype(np.uint8) * 255, cv2.COLORMAP_JET))
        # cv2.imshow('target', cv2.applyColorMap(target.astype(np.uint8) * 255, cv2.COLORMAP_JET))
        # cv2.waitKey()

        # Blob Detection ???
        kernel1d = cv2.getGaussianKernel(3, 2.1)  # original 21, 15
        gaussian1 = np.outer(kernel1d, kernel1d.transpose())
        blob = convolve2d(target, gaussian1, mode='same') * 255

        binary = (blob > blur_threshold).astype(np.uint8)
        labeled = measure.label(binary, connectivity=1)
        blob_properties = measure.regionprops(binary, blob)

        mask = np.zeros_like(binary, np.uint8)
        for prop in blob_properties:
            if len(prop.coords) > small_threshold:
                for r, c in prop.coords:
                    mask[r, c] = 1

        # cv2.imshow('blob', cv2.applyColorMap(blob.astype(np.uint8), cv2.COLORMAP_JET))
        # cv2.imshow('binary', cv2.applyColorMap(binary * 255, cv2.COLORMAP_JET))
        # cv2.imshow('mask', cv2.applyColorMap(mask * 255, cv2.COLORMAP_JET))

        img[mask == 1] = [255, 0, 255]
        cv2.imshow('out', img)
        cv2.waitKey()
