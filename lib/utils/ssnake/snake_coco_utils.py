from lib.utils.ssnake.snake_cityscapes_utils import *

def transform_polys(polys, trans_output, output_h, output_w, use_handle_bp=True):
    new_polys = []
    for i in range(len(polys)):
        poly = polys[i] # [Np,2]
        poly = data_utils.affine_transform(poly, trans_output)
        if use_handle_bp:
            poly = handle_break_point(poly, 0, 0, lambda x, y: x < y)
            poly = handle_break_point(poly, 0, output_w, lambda x, y: x >= y)
            poly = handle_break_point(poly, 1, 0, lambda x, y: x < y)
            poly = handle_break_point(poly, 1, output_h, lambda x, y: x >= y)
        if len(poly) == 0:
            continue
        if len(np.unique(poly, axis=0)) <= 2:
            continue
        new_polys.append(poly)
    return new_polys

input_scale = np.array([512, 512])


def augment(img, split, _data_rng, _eig_val, _eig_vec, mean, std, polys):
    # resize input
    height, width = img.shape[0], img.shape[1]
    center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    scale = max(height, width)
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    # random crop and flip augmentation
    flipped_h = False
    flipped_v = False
    if split == 'train':
        scale = scale * np.random.uniform(0.6, 1.4)
        seed = np.random.randint(0, len(polys))
        index = np.random.randint(0, len(polys[seed][0]))
        x, y = polys[seed][0][index] # find an object point as center x, y
        center[0] = x
        border = scale[0] // 2 if scale[0] < width else width - scale[0] // 2
        center[0] = np.clip(center[0], a_min=border, a_max=width-border)
        center[1] = y
        border = scale[1] // 2 if scale[1] < height else height - scale[1] // 2
        center[1] = np.clip(center[1], a_min=border, a_max=height-border)

        # flip augmentation
        if np.random.random() < 0.5:
            flipped_h = True
            img = img[:, ::-1, :]
            center[0] = width - center[0] - 1
        # flip augmentation
        if np.random.random() < 0.5:
            flipped_v = True
            img = img[::-1, :,  :]
            center[1] = height - center[1] - 1

    input_w, input_h = input_scale
    if split != 'train':
        center = np.array([width // 2, height // 2])
        scale = np.array([width, height])
        x = 32
        input_w = (int(width / 1.) | (x - 1)) + 1
        input_h = (int(height / 1.) | (x - 1)) + 1
        scale = np.array([input_w, input_h])
        # input_w, input_h = (width + x - 1) // x * x, (height + x - 1) // x * x
        # input_w, input_h = int((width / 0.5 + x - 1) // x * x), int((height / 0.5 + x - 1) // x * x)
        # input_w, input_h = 512, 512

    trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h]) # trans_input is a transform matrix used for cv2.warpAffine
    inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

    # color augmentation
    orig_img = inp.copy()
    inp = (inp.astype(np.float32) / 255.)
    if split == 'train':
        data_utils.color_aug(_data_rng, inp, _eig_val, _eig_vec)
        # data_utils.blur_aug(inp)

    # normalize the image
    inp = (inp - mean) / std
    inp = inp.transpose(2, 0, 1)

    output_h, output_w = input_h // snake_config.down_ratio, input_w // snake_config.down_ratio
    trans_output = data_utils.get_affine_transform(center, scale, 0, [output_w, output_h]) # trans_output is a transform matrix used for cv2.warpAffine
    inp_out_hw = (input_h, input_w, output_h, output_w) #inp_out_hw
    # params: input_h,input_w are the size of input image; output_h/w are the size of output?
    return orig_img, inp, trans_input, trans_output, flipped_h, flipped_v, center, scale, inp_out_hw


def flip(img, polys):
    height, width = img.shape[0], img.shape[1]
    center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    scale = max(height, width)
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    flipped_h = np.random.random() < 0.5
    flipped_v = np.random.random() < 0.5
    if flipped_h:
        # img
        img = img[:, ::-1, :]
        # polys
        polys_ = []
        for poly in polys:
            poly[:, 0] = width - np.array(poly[:, 0]) - 1  # along x or w dimension
            polys_.append(poly.copy())
        polys = polys_
    if flipped_v:
        # img
        img = img[::-1, :, :]
        # polys
        polys_ = []
        for poly in polys:
            poly[:, 1] = height - np.array(poly[:, 1]) - 1  # along y or h dimension
            polys_.append(poly.copy())
        polys = polys_
    return img, polys


def random_scale(img, polys, scale_range=(0.7, 1.1), out_size=(512, 512), use_center_shift=False):
    height, width = img.shape[0], img.shape[1]
    center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    scale = max(height, width)
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_min, scale_max = scale_range
    scale = scale * np.random.uniform(scale_min, scale_max)
    seed = np.random.randint(0, len(polys))
    index = np.random.randint(0, len(polys[seed][0]))
    if use_center_shift:  # TO-DO
        centerx, centery = polys[seed][0][index]  # find an object point as center x, y
        center[0] = centerx
        border = scale[0] // 2 if scale[0] < width else width - scale[0] // 2
        center[0] = np.clip(center[0], a_min=border, a_max=width - border)
        center[1] = centery
        border = scale[1] // 2 if scale[1] < height else height - scale[1] // 2
        center[1] = np.clip(center[1], a_min=border, a_max=height - border)
        # trans_input is a transform matrix used for cv2.warpAffine
    out_w, out_h = out_size
    trans_in = data_utils.get_affine_transform(center, scale, 0, [out_w, out_h])
    # img
    img_out = cv2.warpAffine(img, trans_in, (out_w, out_h), flags=cv2.INTER_LINEAR)
    # others
    out_h2, out_w2 = out_h // snake_config.down_ratio, out_w // snake_config.down_ratio
    trans_out = data_utils.get_affine_transform(center, scale, 0, [out_w2, out_h2]) # trans_output is a transform matrix used for cv2.warpAffine
    in_out_hw = (out_h, out_w, out_h2, out_w2)
    # polys
    polys = transform_polys(polys, trans_out, out_h, out_w, use_handle_bp=False)
    return img_out, polys, trans_in, trans_out, in_out_hw, center, scale

def format_standardize(img,polys):
    img = (img.astype(np.float32) / 255.)
    polys = [poly[0] for poly in iter(polys)]
    return img, polys

def augment_standard(img, polys, scale_range = [0.7, 1.1], split='train'):
    """
    achieve data augmentation
    1. flip
    2. scale
    3. color_aug
    """
    if split == 'train':
        # standardize
        img, polys = format_standardize(img,polys)
        # flip augmentation
        img, polys = flip(img,polys)
        # scale
        scale_range = scale_range
        out_size = (512,512)
        img, polys, trans_in, trans_out, in_out_hw, center, scale = random_scale(img,polys,scale_range,out_size,use_center_shift=False)
        orig_img = img.copy()
        # color_aug
        data_utils.color_aug(snake_config.data_rng, img, snake_config.eig_val, snake_config.eig_vec)
        # normalize
        img = (img - snake_config.mean) / snake_config.std
        img = img.transpose(2, 0, 1)

    else:
        # standarize
        img, polys = format_standardize(img,polys)
        # scale
        scale_range = (1.0, 1.0)
        test_size = (512,512)
        img, polys, trans_in, trans_out, in_out_hw, center, scale = random_scale(img, polys, scale_range, test_size,
                                                                  use_center_shift=False)
        orig_img = img.copy()
        # normalize
        img = (img - snake_config.mean) / snake_config.std
        img = img.transpose(2, 0, 1)

    return img, polys, trans_in, trans_out, in_out_hw, orig_img, center, scale

def format_standardize_only_img_inference(img):
    img = (img.astype(np.float32) / 255.)
    return img

def random_scale_only_img_inference(img, scale_range=(0.7, 1.1), out_size=(512, 512)):
    height, width = img.shape[0], img.shape[1]
    center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    scale = max(height, width)
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_min, scale_max = scale_range
    scale = scale * np.random.uniform(scale_min, scale_max)
    out_w, out_h = out_size
    trans_in = data_utils.get_affine_transform(center, scale, 0, [out_w, out_h])
    # img
    img_out = cv2.warpAffine(img, trans_in, (out_w, out_h), flags=cv2.INTER_LINEAR)
    # others
    out_h2, out_w2 = out_h // snake_config.down_ratio, out_w // snake_config.down_ratio
    trans_out = data_utils.get_affine_transform(center, scale, 0, [out_w2, out_h2]) # trans_output is a transform matrix used for cv2.warpAffine
    in_out_hw = (out_h, out_w, out_h2, out_w2)

    return img_out, trans_in, trans_out, in_out_hw, center, scale


def augment_standard_only_img_inference(img):
    """
    achieve data augmentation only in inference (for data standardization)
    1. flip
    2. scale
    3. color_aug
    """
    # standarize
    img = format_standardize_only_img_inference(img)
    # scale
    scale_range = (1.0, 1.0)
    test_size = (512,512)
    img, trans_in, trans_out, in_out_hw, center, scale = random_scale_only_img_inference(img, scale_range, test_size)
    orig_img = img.copy()
    # normalize
    img = (img - snake_config.mean) / snake_config.std
    img = img.transpose(2, 0, 1)

    return img, trans_in, trans_out, in_out_hw, orig_img, center, scale
