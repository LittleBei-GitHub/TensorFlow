from __future__ import division, print_function, absolute_import
import pickle
import numpy as np
import selectivesearch
from PIL import Image
import os.path
import skimage


def pil_to_nparray(pil_image):
    """
        将图片转成数组格式
    """
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")


def resize_image(in_image, new_width, new_height, out_image=None,
                 resize_mode=Image.ANTIALIAS):
    """
        调整图片大小
    """
    img = in_image.resize((new_width, new_height), resize_mode)
    if out_image:
        img.save(out_image)
    return img


def if_intersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b):
    """
        IOU Part 1
        判断两个检测框是否相交，如果相交返回相交的面积
    """
    if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    else:
        return False
    if if_intersect:
        x_sorted_list = sorted([xmin_a, xmax_a, xmin_b, xmax_b])
        y_sorted_list = sorted([ymin_a, ymax_a, ymin_b, ymax_b])
        x_intersect_w = x_sorted_list[2] - x_sorted_list[1]
        y_intersect_h = y_sorted_list[2] - y_sorted_list[1]
        area_inter = x_intersect_w * y_intersect_h
        return area_inter


def IOU(ver1, vertice2):
    """
        IOU Part 2
        计算IOU
    """
    # vertices in four points
    # 构造四个候选框的四个顶点
    vertice1 = [ver1[0], ver1[1], ver1[0]+ver1[2], ver1[1]+ver1[3]]
    area_inter = if_intersection(vertice1[0], vertice1[2], vertice1[1], vertice1[3], vertice2[0], vertice2[2], vertice2[1], vertice2[3])
    # 如果相交计算iou
    if area_inter:
        area_1 = ver1[2] * ver1[3]
        area_2 = vertice2[4] * vertice2[5]
        iou = float(area_inter) / (area_1 + area_2 - area_inter)
        return iou
    return False


def clip_pic(img, rect):
    """
        裁剪图片
    """
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    x_1 = x + w
    y_1 = y + h
    return img[x:x_1, y:y_1, :], [x, y, x_1, y_1, w, h]

# Read in data and save data for Alexnet
def load_train_proposals(datafile, num_clss, threshold = 0.5, svm = False, save=False, save_path='dataset.pkl'):
    """
        生成用于finetuning或者训练svm的数据集
    """
    # 从文件中读取训练数据名字
    train_list = open(datafile, 'r')
    labels = []
    images = []
    for line in train_list:
        tmp = line.strip().split(' ')
        # 读取图片
        img = skimage.io.imread(tmp[0])
        print(tmp[0])
        # 使用selectivesearch选择候选区域
        img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
        candidates = set()
        # 对每个候选区域找到其对映的标签
        for r in regions:
	        # excluding same rectangle (with different segments)
            if r['rect'] in candidates:
                continue
            elif r['size'] < 220:
                continue
            # Ignore things contain 0 or not C contiguous array
            x, y, w, h = r['rect']
            if w == 0 or h == 0:
                continue
	        # resize to 224 * 224 for input
            proposal_img, proposal_vertice = clip_pic(img, r['rect'])
            # 删除空的图片
            if len(proposal_img) == 0:
                continue
            # Check if any 0-dimension exist
	        [a, b, c] = np.shape(proposal_img)
	        if a == 0 or b == 0 or c == 0:
		        continue
	        im = Image.fromarray(proposal_img)
	        resized_proposal_img = resize_image(im, 224, 224)
	        candidates.add(r['rect'])
	        img_float = pil_to_nparray(resized_proposal_img)
            images.append(img_float)

            # IOU
            ref_rect = tmp[2].split(',')
            ref_rect_int = [int(i) for i in ref_rect]
            iou_val = IOU(ref_rect_int, proposal_vertice)
            # labels, let 0 represent default class, which is background
            index = int(tmp[1])
            if svm == False:
            	label = np.zeros(num_clss+1)
            	if iou_val < threshold:
                    label[0] = 1
            	else:
                    label[index] = 1
            	labels.append(label)
            else:
                if iou_val < threshold:
                    labels.append(0)
                else:
                    labels.append(index)
    if save:
        pickle.dump((images, labels), open(save_path, 'wb'))
    return images, labels


def load_from_pkl(dataset_file):
    """
        从pickle文件中加载数据集
    """
    X, Y = pickle.load(open(dataset_file, 'rb'))
    return X, Y
