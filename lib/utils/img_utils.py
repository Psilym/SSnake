import torch
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2


def normalize99(Y, lower=1, upper=99):
    """ normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile """
    X = Y.copy()
    x01 = np.percentile(X, lower)
    x99 = np.percentile(X, upper)
    X = (X - x01) / (x99 - x01)
    return X


# modified to use sinebow color
def dx_to_circ(dP, transparency=False, mask=None):
    """ dP is 2 x Y x X => 'optic' flow representation

    Parameters
    -------------

    dP: 2xLyxLx array
        Flow field components [dy,dx]

    transparency: bool, default False
        magnitude of flow controls opacity, not lightness (clear background)

    mask: 2D array
        Multiplies each RGB component to suppress noise

    """

    dP = np.array(dP)
    mag = np.clip(normalize99(np.sqrt(np.sum(dP ** 2, axis=0))), 0, 1.)
    angles = np.arctan2(dP[1], dP[0]) + np.pi
    a = 2
    r = ((np.cos(angles) + 1) / a)
    g = ((np.cos(angles + 2 * np.pi / 3) + 1) / a)
    b = ((np.cos(angles + 4 * np.pi / 3) + 1) / a)

    if transparency:
        im = np.stack((r, g, b, mag), axis=-1)
    else:
        im = np.stack((r * mag, g * mag, b * mag), axis=-1)

    if mask is not None and transparency and dP.shape[0] < 3:
        im[:, :, -1] *= mask

    im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
    return im

def quiver_plot(vector_field, image=None, cmap='gray', figsize=None, qcmap='twilight', linewidth=.125, width=.19,
                alpha=.7):
    """Quiver plot.
    Plots a 2d vector field.
    Can be used to visualize local refinement tensor.
    Args:
        vector_field: Array[2, w, h].
        image: Array[h, w(, 3)].
        cmap: Image color map.
        figsize: Figure size.
        qcmap: Quiver color map. Consider seaborn's: `qcmap = ListedColormap(sns.color_palette("husl", 8).as_hex())`
        linewidth: Quiver line width.
        width: Quiver width.
        alpha: Quiver alpha.
    """
    angles = np.arctan2(vector_field[0], vector_field[1])
    X, Y = np.meshgrid(np.arange(0, vector_field.shape[2]), np.arange(0, vector_field.shape[1]))
    U, V = vector_field[0], vector_field[1]
    if figsize is not None:
        plt.figure(None, figsize)
    plt.quiver(X, Y, U, V, angles, width=width, alpha=alpha, linewidth=linewidth, angles='xy', units='xy', scale=1,
               cmap=qcmap)
    plt.quiver(X, Y, U, V, width=width, edgecolor='k', alpha=alpha, facecolor='None', linewidth=linewidth, angles='xy',
               units='xy', scale=1, cmap=qcmap)
    plt.grid(0)

def unnormalize_img(img, mean, std):
    """
    img: [3, h, w]
    """
    img = img.detach().cpu().clone()
    # img = img / 255.
    img *= torch.tensor(std).view(3, 1, 1)
    img += torch.tensor(mean).view(3, 1, 1)
    min_v = torch.min(img)
    img = (img - min_v) / (torch.max(img) - min_v)
    return img


def draw_seg_th(seg, num_cls=-1):
    """
    seg: [h, w]
    """
    r = seg.clone()
    g = seg.clone()
    b = seg.clone()
    num_cls = len(colors) if num_cls == -1 else num_cls
    seg_colors = 1 - colors[:, 0, 0]
    for l in range(num_cls):
        inds = (seg == l)
        r[inds] = int(seg_colors[l][0])
        g[inds] = int(seg_colors[l][1])
        b[inds] = int(seg_colors[l][2])
    seg = torch.stack([r, g, b], dim=0).float() / 255.
    return seg


def draw_seg_prob_th(seg_prob):
    """
    seg_prob: [num_cls, h, w]
    """
    num_cls = seg_prob.shape[0]
    seg = torch.argmax(seg_prob, dim=0).long()
    return draw_seg_th(seg, num_cls)


def draw_vertex_th(vertex):
    """
    vertex: [h, w]
    """
    min_ver = torch.min(vertex)
    max_ver = torch.max(vertex)
    vertex = (vertex - min_ver) / (max_ver - min_ver)
    vertex = cmap(vertex.detach().cpu().numpy())[..., :3]
    return torch.tensor(vertex).permute(2, 0, 1)


def visualize_coco_bbox(img, boxes):
    """
    img: [h, w, 3]
    boxes: [n, 4], [[x, y, x_max, y_max]]
    """
    _, ax = plt.subplots(1)
    ax.imshow(img)
    n = len(boxes)
    for ni in range(n):
        x, y, x_max, y_max = boxes[ni]
        ax.add_patch(patches.Polygon(xy=[[x, y], [x, y_max], [x_max, y_max], [x_max, y]], fill=False, linewidth=1, edgecolor='r'))
    plt.show()


def visualize_heatmap(img, hm):
    """
    img: [h, w, 3]
    hm: [c, h, w]
    """
    hm = np.max(hm, axis=0)
    h, w = hm.shape[:2]
    img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
    hm = np.array([255, 255, 255]) - (hm.reshape(h, w, 1) * colors[0]).astype(np.uint8)
    ratio = 0.5
    blend = (img * ratio + hm * (1 - ratio)).astype(np.uint8)
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax2.imshow(blend)
    plt.show()


def visualize_coco_img_mask(img, mask):
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax2.imshow(mask)
    plt.show()


def visualize_color_aug(orig_img, aug_img):
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(orig_img[:, :, [2, 1, 0]])
    ax2.imshow(aug_img[:, :, [2, 1, 0]])
    plt.show()


def visualize_coco_ann(coco, img, ann):
    plt.imshow(img)
    coco.showAnns(ann)
    plt.show()


def bgr_to_rgb(img):
    return img[:, :, [2, 1, 0]]

cmap = cm.get_cmap()
color_list = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.167, 0.000, 0.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000,
        0.50, 0.5, 0
    ]
).astype(np.float32)
colors = color_list.reshape((-1, 3)) * 255
colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
