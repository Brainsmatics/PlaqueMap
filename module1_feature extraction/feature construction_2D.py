'''

Feature construction - 2D version

Includes low-level feature architecture:
1. axial_ratio computation,
2. gray-scale curve acquisition and fitting,
3. gray-scale step detection.

Usage: call analysis_single function, input original plaque data array.

'''


import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import filters
from scipy import signal
import SimpleITK as sitk
import pandas as pd



def Area_outline(image, height, width):
    """Extract the largest contour and its mask."""
    image = image.astype(np.uint8)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return [], np.array([])
    cnt_max = max(contours, key=cv2.contourArea)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [cnt_max], 255)
    return cnt_max, mask


def calGrayHist(image):
    grayHist = np.zeros([256])
    for h in range(image.shape[0]):
        for w in range(image.shape[1]):
            grayHist[image[h][w]] += 1
    return grayHist


def threshEntropy(image):
    # Maximum entropy threshold segmentation
    grayHist = calGrayHist(image)
    normgrayHist = grayHist / float(image.shape[0] * image.shape[1])
    zeroCumuMoment = np.zeros([256], np.float32)
    for k in range(256):
        if k == 0:
            zeroCumuMoment[k] = normgrayHist[k]
        else:
            zeroCumuMoment[k] = zeroCumuMoment[k - 1] + normgrayHist[k]
    entropy = np.zeros([256], np.float32)
    for k in range(256):
        if k == 0:
            if normgrayHist[k] == 0:
                entropy[k] = 0
            else:
                entropy[k] = -normgrayHist[k] * np.log10(normgrayHist[k])
        else:
            if normgrayHist[k] == 0:
                entropy[k] = entropy[k - 1]
            else:
                entropy[k] = entropy[k - 1] - normgrayHist[k] * np.log10(normgrayHist[k])
    ft = np.zeros([256], np.float32)
    ft1, ft2 = 0., 0.
    totalEntropy = entropy[255]
    for k in range(255):
        maxfornt = np.max(normgrayHist[:k + 1])
        maxback = np.max(normgrayHist[k + 1:256])
        if (maxfornt == 0 or zeroCumuMoment[k] == 0 or maxfornt == 1 or zeroCumuMoment[k] == 1 or totalEntropy == 0):
            ft1 = 0
        else:
            ft1 = entropy[k] / totalEntropy * (np.log10(zeroCumuMoment[k]) / np.log10(maxfornt))
        if (maxback == 0 or 1 - zeroCumuMoment[k] == 0 or maxback == 1 or 1 - zeroCumuMoment[k] == 1):
            ft2 = 0
        else:
            if totalEntropy == 0:
                ft2 = (np.log10(1 - zeroCumuMoment[k]) / np.log10(maxback))
            else:
                ft2 = (1 - entropy[k] / totalEntropy) * (np.log10(1 - zeroCumuMoment[k]) / np.log10(maxback))
        ft[k] = ft1 + ft2
    thresloc = np.where(ft == np.max(ft))
    thresh = thresloc[0][0]
    threshold = np.copy(image).astype(np.uint8)
    threshold[threshold > thresh] = 255
    threshold[threshold <= thresh] = 0
    return threshold


def min_box(cnt_max, image_thresh, analysis_dir, file_name):
    """Calculate minimum bounding rectangle and save annotated image."""
    try:
        rect = cv2.minAreaRect(cnt_max)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        img = image_thresh.copy()
        cv2.drawContours(img, [box], 0, (255, 0, 0), 1)
        label_file = os.path.join(analysis_dir + file_name + "_z.png")
        cv2.imwrite(label_file, img)
        return rect, box
    except Exception as e:
        print("min_box error:", e)
        return 0, 0


def remove_nan_inf_extended(original_list):
    cleaned = []
    for item in original_list:
        if isinstance(item, (int, float, np.floating, np.integer)):
            if not (np.isnan(item) or np.isinf(item)):
                cleaned.append(item)
        else:
            try:
                val = float(item)
                if not (np.isnan(val) or np.isinf(val)):
                    cleaned.append(val)
            except (ValueError, TypeError):
                continue
    return cleaned


def draw_gravity(image, analysis_dir, file_name):
    """Calculate gravity center gray values and perform 50-order polynomial fitting."""
    h, w = image.shape
    # Gravity along x
    gravity_x = []
    for i in range(w):
        column = image[:, i]
        if np.sum(column) == 0:
            gravity_x.append(0)
        else:
            coordinate = np.where(column > 0)
            gravity_x.append(np.sum(coordinate[0] * column[coordinate].astype(np.float64)) / np.sum(column.astype(np.float64)))
    # Gravity along y
    gravity_y = []
    for i in range(h):
        row = image[i, :]
        if np.sum(row) == 0:
            gravity_y.append(0)
        else:
            coordinate = np.where(row > 0)
            gravity_y.append(np.sum(coordinate[0] * row[coordinate].astype(np.float64)) / np.sum(row.astype(np.float64)))
    # Polynomial fitting
    x = range(w)
    y = range(h)
    coefs_x = np.polyfit(x, gravity_x, 50)
    poly_x = np.poly1d(coefs_x)
    fit_x = poly_x(x)
    coefs_y = np.polyfit(y, gravity_y, 50)
    poly_y = np.poly1d(coefs_y)
    fit_y = poly_y(y)
    # Save fitting plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(x, gravity_x, 'b-', label='Original')
    ax1.plot(x, fit_x, 'r--', label='Fitted (50)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Gravity_x value')
    ax1.legend()
    ax2.plot(y, gravity_y, 'b-', label='Original')
    ax2.plot(y, fit_y, 'r--', label='Fitted (50)')
    ax2.set_xlabel('y')
    ax2.set_ylabel('Gravity_y value')
    ax2.legend()
    save_path = os.path.join(analysis_dir + file_name + "_fit_x.jpg")
    plt.savefig(save_path, dpi=300)
    plt.close()
    save_path2 = os.path.join(analysis_dir + file_name + "_fit_y.jpg")
    plt.savefig(save_path2, dpi=300)
    plt.close()
    return gravity_x, gravity_y


def analysis_single(cnt_max, image_thresh, arr, analysis_dir, file_name):
    """Main analysis function: calculate axial ratio, gravity center peak features."""
    h, w = arr.shape
    # Calculate minimum bounding rectangle
    rect, box = min_box(cnt_max, image_thresh, analysis_dir, file_name)
    if rect == 0:
        ratio = 0
        value_min = 0
    else:
        w1 = rect[1][0]
        h1 = rect[1][1]
        value_max = max(w1, h1)
        value_min = min(w1, h1)
        if value_min == 0:
            ratio = 0
        else:
            ratio = value_max / value_min
    # Calculate gravity center curves
    gravity_x, gravity_y = draw_gravity(arr, analysis_dir, file_name)
    # Peak detection for x
    peak_x = []
    num_peak_x = signal.find_peaks(np.array(gravity_x))
    if len(num_peak_x[0]) == 1:
        max_peak_x1 = gravity_x[num_peak_x[0][0]]
        max_peak_x2 = 0
    elif len(num_peak_x[0]) == 0:
        max_peak_x1 = 0
        max_peak_x2 = 0
    else:
        for index in range(len(num_peak_x[0])):
            peak = gravity_x[num_peak_x[0][index]]
            peak_x.append(peak)
        max_peak_x1 = max(peak_x)
        peak_index_x = []
        for i in range(len(peak_x)):
            if peak_x[i] == max_peak_x1:
                peak_index_x.append(i)
        for ii in sorted(peak_index_x)[::-1]:
            del (peak_x[ii])
        if len(peak_x) == 0:
            max_peak_x2 = 0
        else:
            max_peak_x2 = max(peak_x)
    # Peak detection for y
    peak_y = []
    num_peak_y = signal.find_peaks(np.array(gravity_y))
    if len(num_peak_y[0]) == 1:
        max_peak_y1 = gravity_y[num_peak_y[0][0]]
        max_peak_y2 = 0
    elif len(num_peak_y[0]) == 0:
        max_peak_y1 = 0
        max_peak_y2 = 0
    else:
        for index in range(len(num_peak_y[0])):
            peak = gravity_y[num_peak_y[0][index]]
            peak_y.append(peak)
        max_peak_y1 = max(peak_y)
        peak_index_y = []
        for i in range(len(peak_y)):
            if peak_y[i] == max_peak_y1:
                peak_index_y.append(i)
        for ii in sorted(peak_index_y)[::-1]:
            del (peak_y[ii])
        if len(peak_y) == 0:
            max_peak_y2 = 0
        else:
            max_peak_y2 = max(peak_y)
    return [ratio, value_min, gravity_x, gravity_y,
            max_peak_x1, max_peak_x2, max_peak_y1, max_peak_y2]


def volume_single(bbox_arr):
    # Calculate volume (number of non-zero pixels)
    volume = np.sum(bbox_arr > 0)
    return volume


def out_line(image, file_name, output_dir):
    """Otsu threshold segmentation (forward: image >= thresh)"""
    arr = image
    if np.max(arr) > 255:
        arr = (arr / 65535.0 * 255).astype(np.uint8)
    thresh = filters.threshold_otsu(arr)
    image_thresh = (arr >= thresh).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(output_dir, file_name + '_thresh.png'), image_thresh)
    height, width = arr.shape
    cnt_max, mask = Area_outline(image_thresh, height, width)
    if len(cnt_max) == 0:
        print("Warning: No contour found for", file_name)
        return image_thresh, [], np.array([]), 0
    cv2.imwrite(os.path.join(output_dir, file_name + '_mask.png'), mask)
    volume = volume_single(mask)
    return image_thresh, cnt_max, mask, volume


def outline(image, file_name, output_dir):
    """Otsu threshold segmentation (inverse: image < thresh)"""
    arr = image
    if np.max(arr) > 255:
        arr = (arr / 65535.0 * 255).astype(np.uint8)
    thresh = filters.threshold_otsu(arr)
    image_thresh = (arr < thresh).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(output_dir, file_name + '_thresh.png'), image_thresh)
    height, width = arr.shape
    cnt_max, mask = Area_outline(image_thresh, height, width)
    if len(cnt_max) == 0:
        print("Warning: No contour found for", file_name)
        return image_thresh, [], np.array([]), 0
    cv2.imwrite(os.path.join(output_dir, file_name + '_outline.png'), mask)
    volume = volume_single(mask)
    return image_thresh, cnt_max, mask, volume


def jpg_to_8bit_grayscale_advanced(img, output_path, normalize=True):
    # Convert RGB image to 8-bit grayscale
    arr = sitk.GetArrayFromImage(img)
    if arr.ndim == 3:
        if arr.shape[0] == 3:
            r, g, b = arr[0, :, :], arr[1, :, :], arr[2, :, :]
        elif arr.shape[2] == 3:
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        else:
            raise ValueError("Unexpected image shape: {}".format(arr.shape))
    else:
        raise ValueError("Input image is not 3D (2D + channel), got shape: {}".format(arr.shape))
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    if normalize:
        gray = gray / 255.0
    gray = (gray * 255).astype(np.uint8)
    gray_img = sitk.GetImageFromArray(gray)
    sitk.WriteImage(gray_img, output_path)
    return gray_img


if __name__ == '__main__':
    # test
    img = sitk.ReadImage("./test_2D/test_AD.jpg")
    img_gray = jpg_to_8bit_grayscale_advanced(img, "./test_2D/test_AD_gray.tif")
    arr = sitk.GetArrayFromImage(img_gray)
    ana_dir = "./test_2D/"
    thresh, cnt, mask, volume = outline(arr, "test_AD", ana_dir)
    out = analysis_single(cnt, thresh, arr, ana_dir, "test_AD")
    # Save results to CSV
    index = ['ratio', 'value_min', 'gravity_x', 'gravity_y',
             'max_peak_x1', 'max_peak_x2', 'max_peak_y1', 'max_peak_y2']
    df = pd.DataFrame({'Values': out}, index=index)
    df.to_csv('./test_2D/test_AD_out.csv', encoding='utf-8')