'''

Plaque segmentation - 2D version

For processing standard 2D plaque data.
For whole slide image (WSI) data, conversion to greyscale images is required beforehand.
To use this script, call the "outline" function.

'''


import cv2
import SimpleITK as sitk
import imutils
import numpy as np
from skimage import filters
import os


def outline(image, file_name, output_dir):
    h,w = image.shape
    image = np.array(image,dtype='uint8')  #uint8
    # image_thresh = threshEntroy(image)     #maxEntroy
    thresh = filters.threshold_otsu(image)   #otsu
    # image_thresh = (image>=thresh)*255.0       #otsu
    image_thresh = (image < thresh) * 255.0   #otsu inverse
    image_thresh = np.array(image_thresh, dtype='uint8')
    thresh_file = os.path.join(output_dir, file_name + "_thresh.png")
    cv2.imwrite(thresh_file, image_thresh)
    # image_thresh = np.array(image_thresh,dtype='uint8')
    cnt_max,mask = Area_outline(image_thresh,h,w)
    label_file = os.path.join(output_dir, file_name + "_outline.png")
    mask = np.array(mask)
    if len(mask) == 0:
        volume = 0
    else:
        volume = volume_single(mask)
        cv2.imwrite(label_file, mask)
    return image_thresh, cnt_max, mask, volume


def calGrayHist(image):
    rows, cols = image.shape
    grayHist = np.zeros([256])  # 8bit 256
    for r in range(rows):
        for c in range(cols):
            grayHist[int(image[r, c])] += 1
    return grayHist



def threshEntroy(image):
    rows, cols = image.shape
    grayHist = calGrayHist(image)
    normgrayHist = grayHist / float(rows * cols)
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
    if thresloc[0].size == 0:
        threshold = np.copy(image)
        threshold[threshold > 0] = 255
    else:
        thresh = thresloc[0][0]
        threshold = np.copy(image)
        threshold[threshold > thresh] = 255
        threshold[threshold <= thresh] = 0
    # return threshold, thresh, max(ft), entropy
    return threshold



def volume_single(bbox_arr):
    volume = 0
    if len(bbox_arr.shape) == 2:
        l, h = bbox_arr.shape
        for i in range(l):
            for j in range(h):
                if bbox_arr[i, j] > 0:
                    volume += 1
    else:
        l, h, w = bbox_arr.shape
        for i in range(l):
            for j in range(h):
                for k in range(w):
                    if bbox_arr[i, j, k] > 0:
                        volume += 1
    return volume



def resampleVolume(outspacing, vol):
    outsize = [0, 0, 0]
    inputsize = vol.GetSize()
    inputspacing = vol.GetSpacing()

    transform = sitk.Transform()
    transform.SetIdentity()
    outsize[0] = round(inputsize[0] * inputspacing[0] / outspacing[0])
    outsize[1] = round(inputsize[1] * inputspacing[1] / outspacing[1])
    outsize[2] = round(inputsize[2] * inputspacing[2] / outspacing[2])

    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkBSpline)  # sitk.sitkLinear
    resampler.SetOutputOrigin(vol.GetOrigin())
    resampler.SetOutputSpacing(outspacing)
    resampler.SetOutputDirection(vol.GetDirection())
    resampler.SetSize(outsize)
    newvol = resampler.Execute(vol)
    return newvol


def Area_outline(image, height, width):
    # blurred = cv2.GaussianBlur(image,(3, 3),1,2)
    # edged = cv2.Canny(blurred ,1,2)
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) == 0:
        cnt_max = []
        mask = []
    else:
        cnt_max = max(cnts, key=cv2.contourArea)
        mask_zero = np.zeros((height, width))
        mask = cv2.fillPoly(mask_zero, [cnt_max], 255)
    return cnt_max, mask


def jpg_to_8bit_grayscale_advanced(img, output_path, normalize=True):
    try:
        color_array = sitk.GetArrayFromImage(img)
        if len(color_array.shape) == 3 and color_array.shape[2] == 3:
            print("rgb fig")
        else:
            print("no rgb fig")
        R = color_array[:, :, 0].astype(np.float32)
        G = color_array[:, :, 1].astype(np.float32)
        B = color_array[:, :, 2].astype(np.float32)
        gray_float = 0.299 * R + 0.587 * G + 0.114 * B
        gray_8bit = np.round(gray_float).astype(np.uint8)
        result_image = sitk.GetImageFromArray(gray_8bit)
        sitk.WriteImage(result_image, output_path)
        return result_image
    except Exception as e:
        print(f"errors: {e}")
        return None



if __name__ == '__main__':
    # Input the plaque data block (in array format) to execute the call.
    # test
    ana_dir = "./test_2D/"
    img = sitk.ReadImage(ana_dir + "test_AD.jpg")
    # arr = tifffile.imread(ana_dir + "test_AD.tif")

    # Convert to greyscale image.
    img_gray = jpg_to_8bit_grayscale_advanced(img, ana_dir + "test_AD_gray.tif")

    arr = sitk.GetArrayFromImage(img_gray)
    thresh, cnt, mask, volume = outline(arr, "test_AD", ana_dir)

