'''

Analyze all plaques within the data block (default 512x512x512), including:

- Implement single-plaque foreground signal segmentation based on coordinates;
- Extract low-level features;
- Extract high-level features;
- Integrate extraction results.

Adjust usage according to provided application examples and data formats.

Usage Notes: The default segmentation method uses maximum entropy segmentation,
which can be adjusted to OTSU segmentation as needed.
The main function provides an optional data upsampling operation.
For 16-bit raw images, the segmentation function must call 'threshEntropy_3D_16'.

'''


import os
import cv2
import math
import SimpleITK as sitk
import pandas as pd
import tifffile
import imutils
from radiomics import featureextractor
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, color, data, filters, morphology, io
from scipy import interpolate, signal
import gc


base_dir = './extract/'
img_dir = './extract/'

# Configure settings based on actual user data.
data_ID = 195592
age = 1.5
index = 2.486728   # The ratio of the total volume of registration data to the total volume of actual data.

suffixes = {"thresh.tif", "compont.tif", "mask.tif", "gaus.tif"}
cls_ = {1:"diffuse", 2:"core", 3:"caa"}

# Determines the features to be computed. To extract different features, modify the feature_names and params.
feature_names = ['shape_VoxelVolume',
                 'shape_MeshVolume',
                 'shape_SurfaceArea',
                 'shape_SurfaceVolumeRatio',
                 'shape_Compactness1',
                 'shape_Compactness2',
                 'shape_Sphericity',
                 'shape_SphericalDisproportion',
                 'shape_Maximum3DDiameter',
                 'shape_Maximum2DDiameterSlice',
                 'shape_Maximum2DDiameterColumn',
                 'shape_Maximum2DDiameterRow',
                 'shape_MajorAxisLength',
                 'shape_MinorAxisLength',
                 'shape_LeastAxisLength',
                 'shape_Elongation',
                 'shape_Flatness',
                 'glcm_Autocorrelation',
                 'glcm_JointAverage',
                 'glcm_ClusterProminence',
                 'glcm_ClusterShade',
                 'glcm_ClusterTendency',
                 'glcm_Contrast',
                 'glcm_Correlation',
                 'glcm_JointEntropy',
                 'glcm_Imc1',
                 'glcm_MaximumProbability',
                 'glcm_SumEntropy',
                 'glcm_SumSquares',
                 'glcm_DifferenceAverage',
                 'glcm_DifferenceEntropy',
                 'glcm_DifferenceVariance',
                 'glcm_JointEnergy',
                 'glcm_Imc2',
                 'glcm_Idm',
                 'glcm_Idmn',
                 'glcm_Id',
                 'glcm_Idn',
                 'glcm_InverseVariance',
                 'glszm_GrayLevelVariance',
                 'glszm_ZoneVariance',
                 'glszm_SizeZoneNonUniformity',
                 'glszm_GrayLevelNonUniformity',
                 'glszm_SmallAreaHighGrayLevelEmphasis',
                 'glszm_LargeAreaLowGrayLevelEmphasis',
                 'glszm_LargeAreaHighGrayLevelEmphasis',
                 'glszm_ZonePercentage',
                 'glszm_ZoneEntropy',
                 'glszm_SmallAreaLowGrayLevelEmphasis',
                 'glszm_SmallAreaEmphasis',
                 'glszm_LargeAreaEmphasis',
                 'glszm_SizeZoneNonUniformityNormalized',
                 'glszm_GrayLevelNonUniformityNormalized',
                 'glszm_HighGrayLevelZoneEmphasis',
                 'glszm_LowGrayLevelZoneEmphasis',
                 'ngtdm_Coarseness',
                 'ngtdm_Contrast',
                 'ngtdm_Busyness',
                 'ngtdm_Complexity',
                 'ngtdm_Strength',
                 'gldm_SmallDependenceEmphasis',
                 'gldm_GrayLevelNonUniformity',
                 'gldm_LargeDependenceEmphasis',
                 'gldm_DependenceNonUniformity',
                 'gldm_DependenceNonUniformityNormalized',
                 'gldm_GrayLevelVariance',
                 'gldm_DependenceVariance',
                 'gldm_DependenceEntropy',
                 'gldm_LowGrayLevelEmphasis',
                 'gldm_HighGrayLevelEmphasis',
                 'gldm_SmallDependenceLowGrayLevelEmphasis',
                 'gldm_SmallDependenceHighGrayLevelEmphasis',
                 'gldm_LargeDependenceLowGrayLevelEmphasis',
                 'gldm_LargeDependenceHighGrayLevelEmphasis']

params = "./param/param.yaml"
extractor = featureextractor.RadiomicsFeatureExtractor(params)


def calGrayHist_3D_16(image, l, h, w):
    lows, rows, cols = l, h, w
    grayHist = np.zeros([65536])  # 8bit 256   16bit  65536
    for l in range(lows):
        for r in range(rows):
            for c in range(cols):
                grayHist[int(image[l, r, c])] += 1
                # gray = int(image[l,r,c])
                # if gray > 255:
                #     gray = 255
                # if gray < 0:
                #     gray = 0
                # grayHist[gray] += 1
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
    # return threshold, thresh, max(ft), entropy
    return threshold.astype(np.uint8)


def calGrayHist_3D(image, l, h, w):
    lows, rows, cols = l, h, w
    grayHist = np.zeros([256])  # 8bit 256   16bit  65536
    for l in range(lows):
        for r in range(rows):
            for c in range(cols):
                grayHist[int(image[l, r, c])] += 1
                # gray = int(image[l,r,c])
                # if gray > 255:
                #     gray = 255
                # if gray < 0:
                #     gray = 0
                # grayHist[gray] += 1
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
    # return threshold, thresh, max(ft), entropy
    return threshold


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
        threshold[threshold > 10] = 255
    else:
        thresh = thresloc[0][0]
        if thresh <= 10:
            thresh = 10
        threshold = np.copy(image)
        threshold[threshold > thresh] = 255
        threshold[threshold <= thresh] = 0
    # return threshold, thresh, max(ft), entropy
    return threshold



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


def Area_outline(image, height, width):
    # blurred = cv2.GaussianBlur(image,(3, 3),1,2)
    # edged = cv2.Canny(blurred ,1,2) # Only usable when the image values are 255
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) == 0:
        cnt_max = []
        mask = []
    else:
        cnt_max = max(cnts, key=cv2.contourArea)
        mask_zero = np.zeros((height, width))
        mask = cv2.fillPoly(mask_zero, [cnt_max], 255)
        # _,c = np.where(mask==255)
    # area_slice = len(c)
    return cnt_max, mask


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


def segmentation(bbox_arr):
    l, h, w = bbox_arr.shape
    threshImg_3D = threshEntroy_3D(bbox_arr, l, h, w)  # 16bit: threshEntroy_3D_16
    threshImg_3D_img = sitk.GetImageFromArray(threshImg_3D)
    remove = RemoveSmallConnectedCompont(threshImg_3D_img, rate=0.5)
    compont = GetConnectedCompont(remove)
    compont_arr = np.array(sitk.GetArrayFromImage(compont), dtype='uint8')
    volume = volume_single(compont_arr)
    im_outline = Outline_3D(compont_arr)
    gaus = Fuzzy(im_outline)
    gaus_arr = np.array(sitk.GetArrayFromImage(gaus), dtype='uint8')
    return threshImg_3D, compont_arr, im_outline, gaus_arr, volume



def connect(suffix, output_dir):
    import re
    regex = re.compile(r'\d+')
    file_names = next(os.walk(output_dir))[2]
    file_names.sort(key=lambda x: (x.split('.')[0]))
    img_seg = np.zeros((700, 700, 700)).astype('uint8')
    for file_name in file_names:
        if suffix not in file_name:
            continue
        img_path = os.path.join(output_dir, file_name)
        img = sitk.ReadImage(img_path)
        img_arr = sitk.GetArrayFromImage(img)

        n = int(max(regex.findall(file_name)))
        overlap = 28
        x_offset = int((n - 1) % 5)
        y_offset = int(((n - 1) % 25) / 5)
        z_offset = int((n - 1) / 25)
        xmin = int(x_offset) * (128 - overlap)
        ymin = int(y_offset) * (128 - overlap)
        zmin = int(z_offset) * (128 - overlap)
        img_seg[zmin:(zmin + 128), ymin:(ymin + 128), xmin:(xmin + 128)] += img_arr
    img_seg = img_seg[:512, :512, :512]
    return img_seg


def projection(image, analysis_dir):
    l, h, w = image.shape
    pro_z = np.zeros((h, w))
    pro_y = np.zeros((l, w))
    for i in range(h):
        for j in range(w):
            pro_z[i, j] = max(image[:, i, j])
    for m in range(l):
        for n in range(w):
            pro_y[m, n] = max(image[m, :, n])
    pro_z = np.array(pro_z, dtype='uint8')  # uint8
    pro_y = np.array(pro_y, dtype='uint8')
    pro_z_file = os.path.join(analysis_dir + "_proz.png")
    pro_y_file = os.path.join(analysis_dir + "_proy.png")
    cv2.imwrite(pro_z_file, pro_z)
    cv2.imwrite(pro_y_file, pro_y)
    return pro_z, pro_y


def min_box(image, analysis_dir, index):
    image = np.array(image, dtype='uint8')
    h, w = image.shape
    # image_thresh = threshEntroy(image)
    img_lens = h * w
    if (len(np.where(image == 255)[0]) == img_lens) or (len(np.where(image == 0)[0]) == img_lens):
        image_thresh = threshEntroy(image)
    else:
        thresh = filters.threshold_otsu(image)
        image_thresh = (image >= thresh) * 255.0
    image_thresh = np.array(image_thresh, dtype='uint8')
    cnt_max, mask = Area_outline(image_thresh, h, w)
    if len(cnt_max) == 0:
        rect = 0
        box = 0
    else:
        rect = cv2.minAreaRect(cnt_max)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # img = image
        img = image_thresh
        cv2.drawContours(img, [box], 0, (255, 0, 0), 1)
        if index == 1:
            label_file = os.path.join(analysis_dir + "_z.png")
        else:
            label_file = os.path.join(analysis_dir + "_y.png")
        cv2.imwrite(label_file, img)
    return rect, box


def remove_nan_inf_extended(original_list):
    clean = []
    for x in original_list:
        if isinstance(x, (int, np.integer)):
            clean.append(x)
        elif isinstance(x, (float, np.floating)):
            if not (np.isnan(x) or np.isinf(x)):
                clean.append(x)
        else:
            clean.append(x)
    return clean


def draw_gravity(image, analysis_dir):
    l, h, w = image.shape
    gravity_x = []
    gravity_y = []
    gravity_z = []
    for k in range(w):
        gravity_x.append(image[int(l / 2), int(h / 2), k])
    for j in range(h):
        gravity_y.append(image[int(l / 2), j, int(w / 2)])
    for i in range(l):
        gravity_z.append(image[i, int(h / 2), int(w / 2)])

    gravity_x = remove_nan_inf_extended(gravity_x)
    gravity_y = remove_nan_inf_extended(gravity_y)
    gravity_z = remove_nan_inf_extended(gravity_z)
    for m in range(3):
        if m == 0:
            y1 = gravity_x
            x1 = range(w)
            z1 = np.polyfit(x1, y1, 50)
            p1 = np.poly1d(z1)
            pred = p1(x1)
            for n in range(len(y1)):
                if y1[n] == 0:
                    y1[n] = 1
            # plt.plot(x1,y1,label='Horizontal line',linewidth=1,color='r',marker='o', markerfacecolor='blue',markersize=5)
            plt.clf()
            plt.plot(x1, y1, label='Gray value', marker='o', markerfacecolor='blue', markersize=5)
            plt.plot(x1, pred, label='Fit line', linewidth=2, color='r')  # np.log10(y1)
            plt.xlabel('Clows Number')
            plt.ylabel('Gravity_x value')  # Gravity_x value (log)
            plt.title('Gravity Value Graph')
            plt.legend()
            # save_path = os.path.join(analysis_dir+'_gray_x'+'.jpg')   # logx
            save_path = os.path.join(analysis_dir + '_fit_x' + '.jpg')
            plt.savefig(save_path)
            # plt.show()
        if m == 1:
            y1 = gravity_y
            x1 = range(h)
            z1 = np.polyfit(x1, y1, 50)
            p1 = np.poly1d(z1)
            pred = p1(x1)
            for n in range(len(y1)):
                if y1[n] == 0:
                    y1[n] = 1
            # plt.plot(x1,y1,label='Vertical line',linewidth=1,color='r',marker='o', markerfacecolor='blue',markersize=5)
            plt.clf()
            plt.plot(x1, y1, label='Gray value', marker='o', markerfacecolor='blue', markersize=5)
            plt.plot(x1, pred, label='Fit line', linewidth=2, color='r')  # np.log10(y1)
            plt.xlabel('Clows Number')
            plt.ylabel('Gravity_y value')
            plt.title('Gravity Value Graph')
            plt.legend()
            # save_path = os.path.join(analysis_dir+'_gray_y'+'.jpg')
            save_path = os.path.join(analysis_dir + '_fit_y' + '.jpg')
            plt.savefig(save_path)
            # plt.show()
        if m == 2:
            y1 = gravity_z
            x1 = range(l)
            z1 = np.polyfit(x1, y1, 50)
            p1 = np.poly1d(z1)
            pred = p1(x1)
            for n in range(len(y1)):
                if y1[n] == 0:
                    y1[n] = 1
            # plt.plot(x1,y1,label='Sagittal line',linewidth=1,color='r',marker='o', markerfacecolor='blue',markersize=5)
            plt.clf()
            plt.plot(x1, y1, label='Gray value', marker='o', markerfacecolor='blue', markersize=5)
            plt.plot(x1, pred, label='Fit line', linewidth=2, color='r')  # np.log10(y1)
            plt.xlabel('Clows Number')
            plt.ylabel('Gravity_z value')
            plt.title('Gravity Mean Value Graph')
            plt.legend()
            # save_path = os.path.join(analysis_dir+'_gray_z'+'.png')
            save_path = os.path.join(analysis_dir + '_fit_z' + '.jpg')
            plt.savefig(save_path)
            # plt.show()
    return gravity_x, gravity_y, gravity_z


def analysis_single(arr, analysis_dir):
    l, h, w = arr.shape
    pro_z, pro_y = projection(arr, analysis_dir)
    rect_z, box_z = min_box(pro_z, analysis_dir, 1)
    rect_y, box_y = min_box(pro_y, analysis_dir, 0)
    if (rect_z == 0) & (rect_y == 0):
        value_min = 0
        ratio = 0
    elif rect_y == 0:
        w1 = rect_z[1][0]
        h1 = rect_z[1][1]
        value1_max = max(w1, h1)
        value1_min = min(w1, h1)
        ratio = value1_max / value1_min
        value_min = value1_min
    elif rect_z == 0:
        w1 = rect_y[1][0]
        h1 = rect_y[1][1]
        value1_max = max(w1, h1)
        value1_min = min(w1, h1)
        ratio = value1_max / value1_min
        value_min = value1_min
    else:
        # max_y = abs(max(box_z[:,-1]))
        # max_z = abs(max(box_y[:,-1]))
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
    # Gravity calculation
    gravity_x, gravity_y, gravity_z = draw_gravity(arr, analysis_dir)

    # if ratio >= 2.2:
    #     if cls == 3:
    #         if ratio >= 2.4:
    #             conf = conf + 0.1
    #     else:
    #         cls = 3
    #         conf = conf - 0.1

    peak_x = []
    num_peak_x = signal.find_peaks(np.array(gravity_x))
    # plt.plot(range(w), gravity_x, 'r', label='polyfit values')
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

    peak_y = []
    num_peak_y = signal.find_peaks(np.array(gravity_y))
    # plt.plot(range(h), gravity_y, 'r', label='polyfit values')
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
    peak_z = []
    num_peak_z = signal.find_peaks(np.array(gravity_z))
    # plt.plot(range(l), gravity_z, 'r', label='polyfit values')
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

    return ratio, value_min, gravity_x, gravity_y, gravity_z, \
           max_peak_x1, max_peak_x2, max_peak_y1, max_peak_y2, max_peak_z1, max_peak_z2


def radiomics_feature(img_arr, label_arr):
    label_arr = label_arr / 255
    label_arr = label_arr.astype("uint8")
    label = sitk.GetImageFromArray(label_arr)
    img = sitk.GetImageFromArray(img_arr)
    result = extractor.execute(img, label)
    features = dict()
    for fn in feature_names:
        features[fn] = result['original_'+fn]
    return features

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



if __name__ == "__main__":
    out_dir = base_dir + "output/"
    if "output" not in os.listdir(base_dir):
        os.mkdir(out_dir)

    cut_dir = out_dir + "cut/"
    if "cut" not in os.listdir(out_dir):
        os.mkdir(cut_dir)

    df_unite = pd.DataFrame()
    img_list = next(os.walk(img_dir))[2]
    for img in img_list:
        img_suffix = img.split(".")[-1]
        if img_suffix != "tif":
            continue
        img_name = img.split(".")[0]

        img_path = base_dir + img
        img_arr = tifffile.imread(img_path)

        img_z, img_y, img_x = img_arr.shape

        box_path = cut_dir + img_name
        if img_name not in os.listdir(cut_dir):
            os.mkdir(box_path)

        ana_dir = box_path + "/analysis"
        if "analysis" not in os.listdir(box_path):
            os.mkdir(ana_dir)

        df = pd.read_csv(base_dir + img_name + "_label.csv")

        df["bbox_xmin"] = df["bbox_xmin"] - 2
        df["bbox_ymin"] = df["bbox_ymin"] - 2
        df["bbox_zmin"] = df["bbox_zmin"] - 2
        df["bbox_xmax"] = df["bbox_xmax"] + 2
        df["bbox_ymax"] = df["bbox_ymax"] + 2
        df["bbox_zmax"] = df["bbox_zmax"] + 2
        df["bbox_xmin"] = df['bbox_xmin'].apply(lambda x: 0 if x < 0 else x)
        df["bbox_ymin"] = df['bbox_ymin'].apply(lambda y: 0 if y < 0 else y)
        df["bbox_zmin"] = df['bbox_zmin'].apply(lambda z: 0 if z < 0 else z)
        df["bbox_xmax"] = df["bbox_xmax"].apply(lambda x: x if x <= img_x else img_x)
        df["bbox_ymax"] = df["bbox_ymax"].apply(lambda y: y if y <= img_y else img_y)
        df["bbox_zmax"] = df["bbox_zmax"].apply(lambda z: z if z <= img_z else img_z)

        # "ID", "data", "num", "age_month", "r"
        ratio_all = []
        value_all = []
        gravity_x_all = []
        gravity_y_all = []
        gravity_z_all = []
        volume_all = []
        all_features = []
        ID_all = []
        data_all = []
        num_all = []
        age_all = []
        r_all = []
        max_peak_x1_all = []
        max_peak_x2_all = []
        max_peak_y1_all = []
        max_peak_y2_all = []
        max_peak_z1_all = []
        max_peak_z2_all = []

        for i in range(df.shape[0]):
            p = df.iloc[i, :]
            xmin = int(p["bbox_xmin"])
            ymin = int(p["bbox_ymin"])
            zmin = int(p["bbox_zmin"])
            xmax = int(p["bbox_xmax"])
            ymax = int(p["bbox_ymax"])
            zmax = int(p["bbox_zmax"])
            num = int(p["ID"])
            ID = img_name + "_" + str(num)

            box_cut = img_arr[zmin:zmax, ymin:ymax, xmin:xmax]

            # If the z-axis resolution is 2 Î¼m, perform upsampling.
            '''
            vol_ori = sitk.GetImageFromArray(box_cut)
            vol_ori_spacing = vol_ori.GetSpacing()
            outspacing_0 = vol_ori_spacing[0]
            outspacing_1 = vol_ori_spacing[1]
            outspacing_2 = vol_ori_spacing[2] * 0.5
            outspacing = [outspacing_0, outspacing_1, outspacing_2]
            vol_resampled = resampleVolume(outspacing, vol_ori)
            box_cut = sitk.GetArrayFromImage(vol_resampled)
            '''

            threshImg_3D, compont_arr, im_outline, gaus_arr, volume = segmentation(box_cut)
            volume_regist = volume * index
            volume_all.append(volume_regist)

            Img_3D_path = box_path + "/" + str(num) + ".tif"
            tifffile.imsave(Img_3D_path, box_cut, photometric="minisblack")
            threshImg_3D_path = box_path + "/" + str(num) + "-thresh.tif"
            tifffile.imsave(threshImg_3D_path, threshImg_3D, photometric="minisblack")
            compont_3D_path = box_path + "/" + str(num) + "-compont.tif"
            tifffile.imsave(compont_3D_path, compont_arr, photometric="minisblack")
            mask_3D_path = box_path + "/" + str(num) + "-mask.tif"
            tifffile.imsave(mask_3D_path, im_outline, photometric="minisblack")
            gaus_3D_path = box_path + "/" + str(num) + "-gaus.tif"
            tifffile.imsave(gaus_3D_path, gaus_arr, photometric="minisblack")

            if len(list(zip(*np.where(compont_arr < 255)))) == 0:
                features = dict()
                for fn in feature_names:
                    features[fn] = 0
                # cls = 0
                # conf = 0
            else:
                aa = len(set(np.where(compont_arr == 255)[0]))
                bb = len(set(np.where(compont_arr == 255)[1]))
                cc = len(set(np.where(compont_arr == 255)[2]))
                if (aa == 1) or (bb == 1) or (cc == 1):
                    features = dict()
                    for fn in feature_names:
                        features[fn] = 0
                    # cls = 0
                    # conf = 0
                else:
                    features = radiomics_feature(box_cut, compont_arr)  # img is the cropped original image block; label is the result after connected-component analysis
            all_features.append(features)

            analysis_dir = ana_dir + "/" + str(num)
            ratio, value_min, gravity_x, gravity_y, gravity_z, \
            max_peak_x1, max_peak_x2, max_peak_y1, max_peak_y2, max_peak_z1, max_peak_z2 = analysis_single(box_cut,
                                                                                                           analysis_dir)
            ratio_all.append(ratio)
            value_all.append(value_min)
            gravity_x_all.append(max(gravity_x))
            gravity_y_all.append(max(gravity_y))
            gravity_z_all.append(max(gravity_z))
            # cls_all.append(cls)
            # conf_all.append(conf)
            ID_all.append(ID)
            data_all.append(data_ID)
            num_all.append(num)
            age_all.append(age)
            r_all.append(max([len(gravity_x), len(gravity_y), len(gravity_z)]))
            max_peak_x1_all.append(max_peak_x1)
            max_peak_x2_all.append(max_peak_x2)
            max_peak_y1_all.append(max_peak_y1)
            max_peak_y2_all.append(max_peak_y2)
            max_peak_z1_all.append(max_peak_z1)
            max_peak_z2_all.append(max_peak_z2)

            del features
            gc.collect()

        gravity_all = []
        for i in range(len(gravity_x_all)):
            gravity_all.append(max(gravity_x_all[i], gravity_y_all[i], gravity_z_all[i]))

        df_f = pd.DataFrame(all_features)
        df = pd.DataFrame()
        df["ID"] = ID_all
        df["data"] = data_all
        df["num"] = num_all
        df["age_month"] = age_all
        df["r"] = r_all
        df["max_gray"] = gravity_all
        df["value_min"] = value_all
        df["ratio"] = ratio_all
        df["volume_regist"] = volume_all
        df["max_peak_x1"] = max_peak_x1_all
        df["max_peak_x2"] = max_peak_x2_all
        df["max_peak_y1"] = max_peak_y1_all
        df["max_peak_y2"] = max_peak_y2_all
        df["max_peak_z1"] = max_peak_z1_all
        df["max_peak_z2"] = max_peak_z2_all

        df = df.reset_index(drop=True)
        df_f = df_f.reset_index(drop=True)
        df_all = pd.concat([df, df_f], axis=1)
        df_all.to_csv(ana_dir + "/features.csv")

        df_unite = pd.concat([df_unite, df_all], axis=0)

    df_unite.to_csv(cut_dir + "/features_all.csv")