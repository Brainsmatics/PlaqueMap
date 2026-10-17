'''

Plaque segmentation - 3D version

To use this script, call the "segmentation" function.
It requires the raw plaque data array as input.

'''


import cv2
import SimpleITK as sitk
import imutils
import numpy as np
import pandas as pd
import tifffile


def Outline_3D(compont_arr):
    im_outline = np.copy(compont_arr)
    i = 0
    for im in compont_arr:
        h, w = im.shape
        cnt_max, mask = Area_outline(im, h, w)
        if len(mask) == 0:
            mask = np.zeros((h, w))
        im_outline[i] = mask
        i = i + 1
    return im_outline


def Fuzzy(im_outline):
    mask_arr = sitk.GetImageFromArray(im_outline)
    sitk_src_gaus = sitk.DiscreteGaussianImageFilter()
    sitk_src_gaus.SetVariance(3)
    sitk_src_gaus.SetMaximumError(0.2)
    sitk_src_gaus = sitk_src_gaus.Execute(mask_arr)
    return sitk_src_gaus


def RemoveSmallConnectedCompont(sitk_maskimg, rate=0.5):
    cc = sitk.ConnectedComponent(sitk_maskimg)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.SetGlobalDefaultNumberOfThreads(8)
    stats.Execute(cc, sitk_maskimg)
    maxlabel = 0
    maxsize = 0
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if maxsize < size:
            maxlabel = l
            maxsize = size
    not_remove = []
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if size > maxsize * rate:
            not_remove.append(l)
    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    outmask[labelmaskimage != maxlabel] = 0
    for i in range(len(not_remove)):
        outmask[labelmaskimage == not_remove[i]] = 255
    outmask_sitk = sitk.GetImageFromArray(outmask)
    outmask_sitk.SetDirection(sitk_maskimg.GetDirection())
    outmask_sitk.SetSpacing(sitk_maskimg.GetSpacing())
    outmask_sitk.SetOrigin(sitk_maskimg.GetOrigin())
    return outmask_sitk


def GetConnectedCompont(remove):
    cc = sitk.ConnectedComponent(remove)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.SetGlobalDefaultNumberOfThreads(8)
    stats.Execute(cc, remove)
    maxlabel = 0
    maxsize = 0
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if maxsize < size:
            maxlabel = l
            maxsize = size
    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    outmask[labelmaskimage == maxlabel] = 255
    outmask[labelmaskimage != maxlabel] = 0
    outmasksitk = sitk.GetImageFromArray(outmask)
    outmasksitk.SetSpacing(remove.GetSpacing())
    outmasksitk.SetOrigin(remove.GetOrigin())
    outmasksitk.SetDirection(remove.GetDirection())

    # afteropen = sitk.BinaryMorphologicalOpening(outmasksitk!=0,[1,1,1])
    compont = sitk.BinaryFillhole(outmasksitk)
    return compont


def calGrayHist_3D_16(image, l, h, w):
    lows, rows, cols = l, h, w
    grayHist = np.zeros([65536])  # 8bit 256   16bit  65536
    for l in range(lows):
        for r in range(rows):
            for c in range(cols):
                grayHist[int(image[l, r, c])] += 1
    return grayHist


def safe_log10(x, epsilon=1e-10):
    x_safe = np.where(x > 0, x, epsilon)
    return np.log10(x_safe)



def threshEntroy_3D_16(image, l, h, w):
    lows, rows, cols = l, h, w
    grayHist = calGrayHist_3D_16(image, l, h, w)
    normgrayHist = grayHist / float(lows * rows * cols)
    zeroCumuMoment = np.zeros([65536], np.float32)
    for k in range(65536):
        if k == 0:
            zeroCumuMoment[k] = normgrayHist[k]
        else:
            zeroCumuMoment[k] = zeroCumuMoment[k - 1] + normgrayHist[k]
    entropy = np.zeros([65536], np.float32)

    for k in range(65536):
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

    ft = np.zeros([65536], np.float32)
    ft1, ft2 = 0., 0.
    totalEntropy = entropy[65535]
    for k in range(65535):
        maxfornt = np.max(normgrayHist[:k + 1])
        maxback = np.max(normgrayHist[k + 1:65536])
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
                ft2 = (1 - entropy[k] / totalEntropy) * (safe_log10(1 - zeroCumuMoment[k]) / safe_log10(maxback))
                # ft2 = (1 - entropy[k] / totalEntropy) * (np.log10(1 - zeroCumuMoment[k]) / np.log10(maxback))
        ft[k] = ft1 + ft2
    thresloc = np.where(ft == np.max(ft))
    if thresloc[0].size == 0:
        threshold = np.copy(image)
        threshold[threshold > 10] = 255
    else:
        thresh = thresloc[0][0]
        if thresh <= 10:
            thresh = 10
        threshold = np.copy(image)
        threshold[threshold > thresh] = 255
        threshold[threshold <= thresh] = 0
    return threshold.astype(np.uint8)



def calGrayHist_3D(image, l, h, w):
    lows, rows, cols = l, h, w
    grayHist = np.zeros([256])  # 8bit 256   16bit  65536
    for l in range(lows):
        for r in range(rows):
            for c in range(cols):
                grayHist[int(image[l, r, c])] += 1
    return grayHist



def threshEntroy_3D(image, l, h, w):
    lows, rows, cols = l, h, w
    grayHist = calGrayHist_3D(image, l, h, w)
    normgrayHist = grayHist / float(lows * rows * cols)
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
        threshold[threshold > 10] = 255
    else:
        thresh = thresloc[0][0]
        if thresh <= 10:
            thresh = 10
        threshold = np.copy(image)
        threshold[threshold > thresh] = 255
        threshold[threshold <= thresh] = 0
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


def segmentation(bbox_arr):
    l, h, w = bbox_arr.shape
    # 8-bit data utilise threshEntroy_3D; 16-bit data must be modified to threshEntroy_3D_16.
    threshImg_3D = threshEntroy_3D_16(bbox_arr, l, h, w)
    threshImg_3D_img = sitk.GetImageFromArray(threshImg_3D)
    remove = RemoveSmallConnectedCompont(threshImg_3D_img, rate=0.5)
    compont = GetConnectedCompont(remove)
    compont_arr = np.array(sitk.GetArrayFromImage(compont), dtype='uint8')
    volume = volume_single(compont_arr)
    im_outline = Outline_3D(compont_arr)
    gaus = Fuzzy(im_outline)
    gaus_arr = np.array(sitk.GetArrayFromImage(gaus), dtype='uint8')
    # ori_thresh, output mask, outline mask, after gaus, volume cal
    return threshImg_3D, compont_arr, im_outline, gaus_arr, volume


if __name__ == '__main__':
    # Input the plaque data block (in array format) to execute the call.
    # test
    ana_dir = "./test/"
    arr = tifffile.imread(ana_dir + "test_tif.tif")
    threshImg_3D, compont_arr, im_outline, gaus_arr, volume = segmentation(arr)       # Save separately according to requirements.
    tifffile.imsave(ana_dir + "test_seg.tif", compont_arr,  photometric='minisblack')
    tifffile.imsave(ana_dir + "test_seg-gaus.tif", gaus_arr,  photometric='minisblack')