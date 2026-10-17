'''

Analyse all 2D plaque images that have been cropped within the designated folder.

- Convert WSI color plaque images to grayscale images.;
- Extract low-level features;
- Extract high-level features;
- Integrate extraction results.

Adjust usage according to provided application examples and data formats.

Usage Notes: The default segmentation method uses maximum entropy segmentation,
which can be adjusted to OTSU segmentation as needed.

'''


from radiomics import featureextractor
import six, numpy as np
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import copy
from skimage import measure, color, data, filters
import imutils
from scipy import interpolate, signal

feature_names = ['shape2D_MeshSurface',
                 'shape2D_PixelSurface',
                 'shape2D_Perimeter',
                 'shape2D_PerimeterSurfaceRatio',
                 'shape2D_Sphericity',
                 'shape2D_SphericalDisproportion',
                 'shape2D_MaximumDiameter',
                 'shape2D_MajorAxisLength',
                 'shape2D_MinorAxisLength',
                 'shape2D_Elongation',
                 'glcm_Autocorrelation',
                 'glcm_JointAverage',
                 'glcm_ClusterProminence',
                 'glcm_ClusterShade',
                 'glcm_ClusterTendency',
                 'glcm_Contrast',
                 'glcm_Correlation',
                 'glcm_DifferenceAverage',
                 'glcm_DifferenceEntropy',
                 'glcm_DifferenceVariance',
                 'glcm_JointEnergy',
                 'glcm_JointEntropy',
                 'glcm_Imc1',
                 'glcm_Imc2',
                 'glcm_Idm',
                 'glcm_Idmn',
                 'glcm_Id',
                 'glcm_Idn',
                 'glcm_InverseVariance',
                 'glcm_MaximumProbability',
                 'glcm_SumEntropy',
                 'glcm_SumSquares',
                 'glszm_GrayLevelVariance',
                 'glszm_ZoneVariance',
                 'glszm_SizeZoneNonUniformity',
                 'glszm_GrayLevelNonUniformity',
                 'glszm_SmallAreaHighGrayLevelEmphasis',
                 'glszm_LargeAreaLowGrayLevelEmphasis',
                 'glszm_LargeAreaHighGrayLevelEmphasis',
                 'glszm_SmallAreaLowGrayLevelEmphasis',
                 'glszm_SmallAreaEmphasis',
                 'glszm_LargeAreaEmphasis',
                 'glszm_GrayLevelNonUniformityNormalized',
                 'glszm_SizeZoneNonUniformityNormalized',
                 'glszm_ZonePercentage',
                 'glszm_ZoneEntropy',
                 'glszm_LowGrayLevelZoneEmphasis',
                 'glszm_HighGrayLevelZoneEmphasis',
                 'ngtdm_Coarseness',
                 'ngtdm_Contrast',
                 'ngtdm_Busyness',
                 'ngtdm_Complexity',
                 'ngtdm_Strength',
                 'gldm_SmallDependenceEmphasis',
                 'gldm_LargeDependenceEmphasis',
                 'gldm_GrayLevelNonUniformity',
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


def calGrayHist(image):
    rows, cols = image.shape
    grayHist = np.zeros([256])  # 8bit 256
    for r in range(rows):
        for c in range(cols):
            grayHist[int(image[r, c])] += 1
    return grayHist



# Maximum entropy threshold
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



def Area_outline(image, height, width):
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


def getFiles(path, suffix):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path) for file in files if file.endswith(suffix)]


def min_box(cnt_max, image_thresh, analysis_dir, file_name):
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
        label_file = os.path.join(analysis_dir + file_name + "_z.png")
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


def draw_gravity(image, analysis_dir, file_name):
    h, w = image.shape
    gravity_x = []
    gravity_y = []

    for k in range(w):
        gravity_x.append(image[int(h / 2), k])
    for j in range(h):
        gravity_y.append(image[j, int(w / 2)])

    gravity_x = remove_nan_inf_extended(gravity_x)
    gravity_y = remove_nan_inf_extended(gravity_y)
    for m in range(2):
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
            save_path = os.path.join(analysis_dir + file_name + '_fit_x' + '.jpg')
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
            save_path = os.path.join(analysis_dir + file_name + '_fit_y' + '.jpg')
            plt.savefig(save_path)
            # plt.show()
    return gravity_x, gravity_y


def analysis_single(cnt_max, image_thresh, arr, analysis_dir, file_name):
    rect_z, box_z = min_box(cnt_max, image_thresh, analysis_dir, file_name)
    if rect_z == 0:
        value_min = 0
        ratio = 0
    else:
        # max_y = abs(max(box_z[:,-1]))
        # max_z = abs(max(box_y[:,-1]))
        w1 = rect_z[1][0]
        h1 = rect_z[1][1]
        value_max = max(w1, h1)
        value_min = min(w1, h1)
        if value_min == 0:
            ratio = 0
        else:
            ratio = value_max / value_min
    # gravity
    gravity_x, gravity_y = draw_gravity(arr, analysis_dir, file_name)

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

    return ratio, value_min, gravity_x, gravity_y, \
           max_peak_x1, max_peak_x2, max_peak_y1, max_peak_y2



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


def jpg_to_8bit_grayscale_advanced(img, output_path, normalize=True):
    try:
        color_array = sitk.GetArrayFromImage(img)
        if len(color_array.shape) == 3 and color_array.shape[2] == 3:
            print("rgb")
        else:
            print("no rgb")
        R = color_array[:, :, 0].astype(np.float32)
        G = color_array[:, :, 1].astype(np.float32)
        B = color_array[:, :, 2].astype(np.float32)
        gray_float = 0.299 * R + 0.587 * G + 0.114 * B
        gray_8bit = np.round(gray_float).astype(np.uint8)
        result_image = sitk.GetImageFromArray(gray_8bit)
        sitk.WriteImage(result_image, output_path)
        return result_image
    except Exception as e:
        print(f"error: {e}")
        return None


if __name__ == '__main__':
    ana_dir = './extract_2D/output/'
    img_dir = './extract_2D/'
    params = "./param/param_AD.yaml"
    all_tif = getFiles(img_dir, ".jpg")

    df_unite = pd.DataFrame()
    ratio_all = []
    value_all = []
    gravity_x_all = []
    gravity_y_all = []
    gravity_z_all = []
    volume_all = []
    all_features = []
    ID_all = []
    num_all = []
    r_all = []
    max_peak_x1_all = []
    max_peak_x2_all = []
    max_peak_y1_all = []
    max_peak_y2_all = []
    max_peak_z1_all = []
    max_peak_z2_all = []
    features = dict()
    for index, file in enumerate(all_tif):
        img_name = file.split('/')[-1].split('.')[0]
        img = sitk.ReadImage(file)
        img_gray = jpg_to_8bit_grayscale_advanced(img, ana_dir + img_name + "_gray.tif")

        arr = sitk.GetArrayFromImage(img_gray)
        thresh, cnt, mask, volume = outline(arr, img_name, ana_dir)

        if len(np.unique(mask)) == 1:
            continue
        if mask.ndim == 1:
            continue
        if len(mask) <= 4:
            continue
        if np.sum(mask == 255) <= 2:
            continue

        extractor = featureextractor.RadiomicsFeatureExtractor(params)

        label = mask / 255.0
        label = np.array(label, dtype='uint8')
        label = sitk.GetImageFromArray(label)
        result = extractor.execute(img_gray, label)
        for fn in feature_names:
            features[fn] = result['original_' + fn]
        all_features.append(copy.copy(features))

        ratio, value_min, gravity_x, gravity_y, \
        max_peak_x1, max_peak_x2, max_peak_y1, max_peak_y2 = analysis_single(cnt, thresh, arr, ana_dir, img_name)
        ratio_all.append(ratio)
        value_all.append(value_min)
        gravity_x_all.append(max(gravity_x))
        gravity_y_all.append(max(gravity_y))
        # cls_all.append(cls)
        # conf_all.append(conf)
        volume_all.append(volume)
        ID_all.append(img_name)
        # num_all.append(num)
        r_all.append(max([len(gravity_x), len(gravity_y)]))
        max_peak_x1_all.append(max_peak_x1)
        max_peak_x2_all.append(max_peak_x2)
        max_peak_y1_all.append(max_peak_y1)
        max_peak_y2_all.append(max_peak_y2)

    gravity_all = []
    for i in range(len(gravity_x_all)):
        gravity_all.append(max(gravity_x_all[i], gravity_y_all[i]))

    df_f = pd.DataFrame(all_features)
    df = pd.DataFrame()
    df["ID"] = ID_all
    # df["num"] = num_all
    df["r"] = r_all
    df["max_gray"] = gravity_all
    df["value_min"] = value_all
    df["ratio"] = ratio_all
    df["volume_regist"] = volume_all
    df["max_peak_x1"] = max_peak_x1_all
    df["max_peak_x2"] = max_peak_x2_all
    df["max_peak_y1"] = max_peak_y1_all
    df["max_peak_y2"] = max_peak_y2_all

    df = df.reset_index(drop=True)
    df_f = df_f.reset_index(drop=True)
    df_all = pd.concat([df, df_f], axis=1)
    df_all.to_csv(ana_dir + img_name + "_features.csv")
