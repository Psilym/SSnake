import numpy as np
import cv2
import random

from torch import nn
import torch
from imgaug import augmenters as iaa
from scipy.ndimage import distance_transform_edt as distance


# use the code from ConerNet
def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    if b3 ** 2 - 4 * a3 * c3 < 0:
        r3 = min(r1, r2)
    else:
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=(1, 1), rho=0):
    if not isinstance(sigma, tuple):
        sigma = (sigma, sigma)
    sigma_x, sigma_y = sigma

    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]

    energy = (x * x) / (sigma_x * sigma_x) - 2 * rho * x * y / (sigma_x * sigma_y) + (y * y) / (sigma_y * sigma_y)
    h = np.exp(-energy / (2 * (1 - rho * rho)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def gaussian2D_poly(poly, w_sigma=1/6, rho=0,pad = 2):
    left, right = poly[:, 0].min(), poly[:, 0].max()
    bottom, top = poly[:, 1].min(), poly[:, 1].max()
    w = right - left + 1
    h = top - bottom + 1
    assert w > 0 and h > 0
    poly_can = poly.copy()
    poly_can[:,1] = poly[:,1] - bottom + pad
    poly_can[:,0] = poly[:,0] - left + pad
    # poly_can = np.round(poly_can).astype(np.int32)  # the poly is already list
    poly_can = poly_can[np.newaxis,:]


    mask_ = np.zeros((h+2*pad, w+2*pad), dtype=np.uint8)
    cv2.fillPoly(mask_, poly_can, 1)
    mask_ = mask_.astype(np.bool)
    dis = distance(mask_)
    dis = dis[pad:h+pad,pad:w+pad]
    dis_norm = dis / (dis.max()+0.0001)
    # dis_norm = dis
    # transform from distance to gaussian

    return dis_norm

def draw_gaussian_wh(gaus_map,wh_ct,th=0.5):
    bin_map = gaus_map>th
    w_ct,h_ct = wh_ct
    h_bin,w_bin = bin_map.shape
    wh_map = np.array((h_bin,w_bin,2),dtype=np.float)
    wh_map[...,0] = w_ct
    wh_map[...,1] = h_ct
    return wh_map


def draw_weight(ct_wgh,poly,box):
    output_h, output_w = ct_wgh.shape
    mask_ = np.zeros((output_h, output_w), dtype=np.float32)
    poly = np.round(poly).astype(np.int32)  # the poly is already list
    box = np.round(box).astype(np.int32)
    x_min, y_min, x_max, y_max = box
    weight = int((x_max-x_min)*((y_max-y_min)))
    poly = np.array([[x_min, y_min],
                     [x_max, y_min],
                     [x_max, y_max],
                     [x_min, y_max],], dtype=np.int32) # center position

    cv2.fillPoly(mask_, poly[np.newaxis,...], weight)
    if min(mask_.shape) > 0 and min(ct_wgh.shape) > 0:
        np.maximum(ct_wgh, mask_, out=ct_wgh)

    return ct_wgh

def norm(a):
    return (a - a.min())/(a.max()-a.min())

def draw_poly_gaussian(poly, heatmap, k=1):
    h_hm,w_hm = heatmap.shape
    if len(poly) == 0: # invalid poly
        return heatmap
    poly = poly[0]
    left, right = poly[:,0].min(),poly[:,0].max()
    bottom, top = poly[:,1].min(),poly[:,1].max()
    assert right<w_hm and top<h_hm
    w = right - left + 1
    h = top - bottom + 1
    assert w > 0 and h > 0
    pad = 2
    gaussian = gaussian2D_poly(poly, w_sigma=1/6, pad=pad)
    from PIL import Image
    def norm(a):
        min = 0
        eps = 0.001
        a = (a - min) / (a.max() - min + eps)
        a = np.clip(a,a_min=0,a_max=1)
        return a
    # Image.fromarray(norm(a)*255).show()
    masked_heatmap = heatmap[bottom:(top+1), left:(right+1)]
    masked_gaussian = gaussian
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    heatmap[bottom:top+1, left:right+1] = masked_heatmap

    return heatmap



def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    '''
        center:
        scale: 2d ndarry or list or int
        rot: angle degree (eg 180), normally set as 0
        output_size: tuple, eg [1024,1024]
        shift:
        inv: inverse flag

    '''
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    """pt: [n, 2]"""
    new_pt = np.dot(np.array(pt), t[:, :2].T) + t[:, 2]
    return new_pt


def get_border(border, size):
    i = 1
    while np.any(size - border // i <= border // i):
        i *= 2
    return border // i


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)


def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2


def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])


def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha


def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)


def color_aug(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)

def blur_aug(inp):
    if np.random.random() < 0.1:
        if np.random.random() < 0.8:
            inp = iaa.blur_gaussian_(inp, abs(np.clip(np.random.normal(0, 1.5), -3, 3)))
        else:
            inp = iaa.MotionBlur((3, 15), (-45, 45))(images=[inp])[0]


def gaussian_blur(image, sigma):
    from scipy import ndimage
    if image.ndim == 2:
        image[:, :] = ndimage.gaussian_filter(image[:, :], sigma, mode="mirror")
    else:
        nb_channels = image.shape[2]
        for channel in range(nb_channels):
            image[:, :, channel] = ndimage.gaussian_filter(image[:, :, channel], sigma, mode="mirror")


def inter_from_mask(pred, gt):
    pred = pred.astype(np.bool)
    gt = gt.astype(np.bool)
    intersection = np.logical_and(gt, pred).sum()
    return intersection


def draw_poly(mask, poly):
    cv2.fillPoly(mask, [poly], 255)
    return mask



def get_mask_img(img, poly):
    mask = np.zeros(img.shape[:2])[..., np.newaxis]
    cv2.fillPoly(mask, [np.round(poly['poly']).astype(int)], 1)
    poly_img = img * mask
    mask = mask[..., 0]
    return poly_img, mask



def get_gt_mask(img, poly):
    mask = np.zeros(img.shape[:2])[..., np.newaxis]
    for i in range(len(poly)):
        for j in range(len(poly[i])):
            cv2.fillPoly(mask, [np.round(poly[i][j]['poly']).astype(int)], 1)
    return mask



def truncated_normal(mean, sigma, low, high, data_rng=None):
    if data_rng is None:
        data_rng = np.random.RandomState()
    value = data_rng.normal(mean, sigma)
    return np.clip(value, low, high)


def _nms(heat, kernel=3):
    """heat: [b, c, h, w]"""
    pad = (kernel - 1) // 2

    # find the local minimum of heat within the neighborhood kernel x kernel
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def clip_to_image(bbox, h, w):
    bbox[..., :2] = torch.clamp(bbox[..., :2], min=0)
    bbox[..., 2] = torch.clamp(bbox[..., 2], max=w-1)
    bbox[..., 3] = torch.clamp(bbox[..., 3], max=h-1)
    return bbox


def get_area(bbox):
    area = (bbox[..., 2] - bbox[..., 0] + 1) * (bbox[..., 3] - bbox[..., 1] + 1)
    return area


def box_iou(box1, box2):
    """box1: [n, 4], box2: [m, 4]"""
    area1 = get_area(box1)
    area2 = get_area(box2)

    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt + 1).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou
