'''

Feature construction - 3D version

Includes low-level feature architecture:
1. Axial_ratio computation,
2. gray-scale curve acquisition and fitting,
3. gray-scale step detection.

Usage: call analysis_single function, input original plaque data array.

'''



import numpy as np
import os
import cv2
from skimage import filters, morphology
from scipy import signal
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import tifffile


def projection(image, analysis_dir):
    """Project 3D image along z and y axes."""
    l, h, w = image.shape
    pro_z = np.max(image, axis=0)
    pro_y = np.max(image, axis=2)
    # Save projections
    plt.imsave(os.path.join(analysis_dir + "_z.png"), pro_z, cmap='gray')
    plt.imsave(os.path.join(analysis_dir + "_y.png"), pro_y, cmap='gray')
    return pro_z, pro_y


def Area_outline(image, height, width):
    """Extract the largest contour and its mask."""
    image = image.astype(np.uint8)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return [], np.array([])
    cnt_max = max(contours, key=cv2.contourArea)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [cnt_max], -1, 255, -1)
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


def min_box(image, analysis_dir, index):
    """Minimum bounding rectangle with robust empty contour handling."""
    image = np.array(image, dtype='uint8')
    h, w = image.shape
    img_lens = h * w
    if (len(np.where(image == 255)[0]) == img_lens) or (len(np.where(image == 0)[0]) == img_lens):
        image_thresh = threshEntropy(image)
    else:
        thresh = filters.threshold_otsu(image)
        image_thresh = (image >= thresh) * 255.0
    image_thresh = np.array(image_thresh, dtype='uint8')
    cnt_max, mask = Area_outline(image_thresh, h, w)
    # Robust empty contour check
    if cnt_max is None or len(cnt_max) == 0:
        return 0, 0
    if isinstance(cnt_max, (list, tuple)):
        if len(cnt_max) > 0 and isinstance(cnt_max[0], np.ndarray) and cnt_max[0].shape[0] == 0:
            return 0, 0
    try:
        rect = cv2.minAreaRect(cnt_max)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        img = image_thresh.copy()
        cv2.drawContours(img, [box], 0, (255, 0, 0), 1)
        if index == 1:
            label_file = os.path.join(analysis_dir + "_z.png")
        else:
            label_file = os.path.join(analysis_dir + "_y.png")
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


def draw_gravity(image, analysis_dir):
    l, h, w = image.shape
    # Projection along z axis
    pro_z = np.max(image, axis=0)
    # Gravity along x
    gravity_x = []
    for i in range(w):
        column = pro_z[:, i]
        if np.sum(column) == 0:
            gravity_x.append(0)
        else:
            coordinate = np.where(column > 0)
            gravity_x.append(np.sum(coordinate[0] * column[coordinate].astype(np.float64)) / np.sum(column.astype(np.float64)))
    # Gravity along y
    gravity_y = []
    for i in range(h):
        row = pro_z[i, :]
        if np.sum(row) == 0:
            gravity_y.append(0)
        else:
            coordinate = np.where(row > 0)
            gravity_y.append(np.sum(coordinate[0] * row[coordinate].astype(np.float64)) / np.sum(row.astype(np.float64)))
    # Projection along y axis
    pro_y = np.max(image, axis=2)
    gravity_z = []
    for i in range(l):
        row = pro_y[i, :]
        if np.sum(row) == 0:
            gravity_z.append(0)
        else:
            coordinate = np.where(row > 0)
            gravity_z.append(np.sum(coordinate[0] * row[coordinate].astype(np.float64)) / np.sum(row.astype(np.float64)))
    # Save gravity plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    ax1.plot(range(w), gravity_x, 'b-', label='Gravity_x')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Gravity_x value')
    ax1.legend()
    ax2.plot(range(h), gravity_y, 'r-', label='Gravity_y')
    ax2.set_xlabel('y')
    ax2.set_ylabel('Gravity_y value')
    ax2.legend()
    ax3.plot(range(l), gravity_z, 'g-', label='Gravity_z')
    ax3.set_xlabel('z')
    ax3.set_ylabel('Gravity_z value')
    ax3.legend()
    save_path = os.path.join(analysis_dir + '_gravity.jpg')
    plt.savefig(save_path, dpi=300)
    plt.close()
    return gravity_x, gravity_y, gravity_z


def analysis_single(arr, analysis_dir):
    """Single 3D analysis with robust empty contour handling."""
    l, h, w = arr.shape
    pro_z, pro_y = projection(arr, analysis_dir)
    rect_z, box_z = min_box(pro_z, analysis_dir, 1)
    rect_y, box_y = min_box(pro_y, analysis_dir, 0)
    # Guard: if both rects are zero, return defaults
    # Fixed: simplified condition
    if rect_z == 0 and rect_y == 0:
        value_min = 0
        ratio = 0
    elif rect_z == 0:
        w1 = rect_y[1][0]
        h1 = rect_y[1][1]
        value1_max = max(w1, h1)
        value1_min = min(w1, h1)
        ratio = value1_max / value1_min
        value_min = value1_min
    elif rect_y == 0:
        w1 = rect_z[1][0]
        h1 = rect_z[1][1]
        value1_max = max(w1, h1)
        value1_min = min(w1, h1)
        ratio = value1_max / value1_min
        value_min = value1_min
    else:
        w1 = rect_z[1][0]
        h1 = rect_z[1][1]
        value1_max = max(w1, h1)
        value1_min = min(w1, h1)
        w2 = rect_y[1][0]
        h2 = rect_y[1][1]
        value2_max = max(w2, h2)
        value2_min = min(w2, h2)
        value_max = math.sqrt(value1_max * value1_max + value2_max * value2_max)
        value_min = max(value1_min, value2_min)
        if value_min == 0:
            ratio = 0
        else:
            ratio = value_max / value_min
    gravity_x, gravity_y, gravity_z = draw_gravity(arr, analysis_dir)
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
    # Peak detection for z
    peak_z = []
    num_peak_z = signal.find_peaks(np.array(gravity_z))
    if len(num_peak_z[0]) == 1:
        max_peak_z1 = gravity_z[num_peak_z[0][0]]
        max_peak_z2 = 0
    elif len(num_peak_z[0]) == 0:
        max_peak_z1 = 0
        max_peak_z2 = 0
    else:
        for index in range(len(num_peak_z[0])):
            peak = gravity_z[num_peak_z[0][index]]
            peak_z.append(peak)
        max_peak_z1 = max(peak_z)
        peak_index_z = []
        for i in range(len(peak_z)):
            if peak_z[i] == max_peak_z1:
                peak_index_z.append(i)
        for ii in sorted(peak_index_z)[::-1]:
            del (peak_z[ii])
        if len(peak_z) == 0:
            max_peak_z2 = 0
        else:
            max_peak_z2 = max(peak_z)
    return [ratio, value_min, gravity_x, gravity_y, gravity_z,
            max_peak_x1, max_peak_x2, max_peak_y1, max_peak_y2, max_peak_z1, max_peak_z2]



if __name__ == '__main__':
    # test
    ana_dir = "./test/"
    arr = tifffile.imread(ana_dir + "test_tif.tif")
    out = analysis_single(arr, ana_dir)
    index = ['ratio', 'value_min', 'gravity_x', 'gravity_y', 'gravity_z',
            'max_peak_x1', 'max_peak_x2', 'max_peak_y1', 'max_peak_y2', 'max_peak_z1', 'max_peak_z2']
    df = pd.DataFrame({'Values': out}, index=index)
    df.to_csv(ana_dir + 'test_out.csv', encoding='utf-8')


