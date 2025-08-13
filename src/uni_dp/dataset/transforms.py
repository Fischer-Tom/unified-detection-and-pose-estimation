import torch
from torchvision.transforms import v2 as tf
import numpy as np
import cv2

class ToTensor:
    def __init__(self):
        self.trans = tf.ToImage()

    def __call__(self, sample):
        sample["img"] = self.trans(sample["img"])
        if "K" in sample and not type(sample["K"]) == torch.Tensor:
            sample["K"] = torch.Tensor(sample["K"])
        return sample

class ToDtype:
    def __init__(self):
        self.trans = tf.ToDtype(torch.float32, scale=True)

    def __call__(self, sample):
        sample["img"] = self.trans(sample["img"])

        return sample

class Normalize:
    def __init__(self):
        self.trans = tf.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __call__(self, sample):
        sample["img"] = self.trans(sample["img"])
        return sample


class ColorJitter:
    def __init__(self):
        # the values are from secondpose
        self.trans = tf.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        )

    def __call__(self, sample):
        sample["img"] = self.trans(sample["img"])
        sample["img"] = np.array(sample["img"])

        return sample

class RandomRGBNoise:
    def __init__(self):
        pass

    def __call__(self, sample):
        sample["img"] = self._rgb_add_noise(sample["img"])

        return sample

    def _rgb_add_noise(self, img):
        rng = np.random
        # apply HSV augmentor
        if rng.rand() > 0:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.uint16)
            hsv_img[:, :, 1] = hsv_img[:, :, 1] * self._rand_range(rng, 1.25, 1.45)
            hsv_img[:, :, 2] = hsv_img[:, :, 2] * self._rand_range(rng, 1.15, 1.35)
            hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 255)
            hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2], 0, 255)
            img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2RGB)

        if rng.rand() > 0.8:  # sharpen
            kernel = -np.ones((3, 3))
            kernel[1, 1] = rng.rand() * 3 + 9
            kernel /= kernel.sum()
            img = cv2.filter2D(img, -1, kernel)

        if rng.rand() > 0.8:  # motion blur
            r_angle = int(rng.rand() * 360)
            r_len = int(rng.rand() * 15) + 1
            img = self._linear_motion_blur(img, r_angle, r_len)

        if rng.rand() > 0.8:
            if rng.rand() > 0.2:
                img = cv2.GaussianBlur(img, (3, 3), rng.rand())
            else:
                img = cv2.GaussianBlur(img, (5, 5), rng.rand())

        if rng.rand() > 0.2:
            img = self._gaussian_noise(rng, img, rng.randint(15))
        else:
            img = self._gaussian_noise(rng, img, rng.randint(25))

        if rng.rand() > 0.8:
            img = img + rng.normal(loc=0.0, scale=7.0, size=img.shape)

        return np.clip(img, 0, 255).astype(np.uint8)

    def _rand_range(self, rng, lo: float, hi: float) -> float:
        return float(rng.rand()) * (hi - lo) + lo

    def _gaussian_noise(self, rng, img, sigma):
        """add gaussian noise of given sigma to image"""
        img = img + rng.randn(*img.shape) * sigma
        img = np.clip(img, 0, 255).astype("uint8")
        return img

    def _linear_motion_blur(self, img, angle, length):
        """:param angle: in degree"""
        rad = np.deg2rad(angle)
        dx = np.cos(rad)
        dy = np.sin(rad)
        a = int(max(list(map(abs, (dx, dy)))) * length * 2)
        if a <= 0:
            return img
        kern = np.zeros((a, a))
        cx, cy = a // 2, a // 2
        dx, dy = list(map(int, (dx * length + cx, dy * length + cy)))
        cv2.line(kern, (cx, cy), (dx, dy), 1.0)
        s = kern.sum()
        if s == 0:
            kern[cx, cy] = 1.0
        else:
            kern /= s
        return cv2.filter2D(img, -1, kern)
    
def aug_bbox_DZI(bbox_xyxy, im_H, im_W,DZI_TYPE="uniform",DZI_SCALE_RATIO=0.25,DZI_SHIFT_RATIO=0.25,DZI_PAD_SCALE=1.5):
    """Used for DZI, the augmented box is a square (maybe enlarged)
    Args:
        bbox_xyxy (np.ndarray):
    Returns:
        center, scale
    """
    x1, y1, x2, y2 = bbox_xyxy.copy()
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bh = y2 - y1
    bw = x2 - x1
    if DZI_TYPE.lower() == "uniform":
        scale_ratio = 1 + DZI_SCALE_RATIO * (2 * np.random.random_sample() - 1)  # [1-0.25, 1+0.25]
        shift_ratio = DZI_SHIFT_RATIO * (2 * np.random.random_sample(2) - 1)  # [-0.25, 0.25]
        bbox_center = np.array([cx + bw * shift_ratio[0], cy + bh * shift_ratio[1]])  # (h/2, w/2)
        scale = max(y2 - y1, x2 - x1) * scale_ratio * DZI_PAD_SCALE
    elif DZI_TYPE.lower() == "roi10d":
        # shift (x1,y1), (x2,y2) by 15% in each direction
        _a = -0.15
        _b = 0.15
        x1 += bw * (np.random.rand() * (_b - _a) + _a)
        x2 += bw * (np.random.rand() * (_b - _a) + _a)
        y1 += bh * (np.random.rand() * (_b - _a) + _a)
        y2 += bh * (np.random.rand() * (_b - _a) + _a)
        x1 = min(max(x1, 0), im_W)
        x2 = min(max(x1, 0), im_W)
        y1 = min(max(y1, 0), im_H)
        y2 = min(max(y2, 0), im_H)
        bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
        scale = max(y2 - y1, x2 - x1) * DZI_PAD_SCALE
    elif DZI_TYPE.lower() == "truncnorm":
        raise NotImplementedError("DZI truncnorm not implemented yet.")
    else:
        bbox_center = np.array([cx, cy])  # (w/2, h/2)
        scale = max(y2 - y1, x2 - x1)
    scale = min(scale, max(im_H, im_W)) * 1.0
    return bbox_center, scale

def get_real_hw(bbox,  img_height, img_width):
    y1, x1, y2, x2 = bbox
    y1 = max(0,y1)
    x1 = max(0,x1)
    y2 = min(img_height, y2)
    x2 = min(img_width, x2)
    return x2-x1, y2-y1


def crop_resize_by_warp_affine(img, center, scale, output_size, rot=0, interpolation=cv2.INTER_LINEAR):
    """
    output_size: int or (w, h)
    NOTE: if img is (h,w,1), the output will be (h,w)
    """
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img, trans, (int(output_size[0]), int(output_size[1])), flags=interpolation)

    return dst_img



def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=False):
    """
    adapted from CenterNet: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
    center: ndarray: (cx, cy)
    scale: (w, h)
    rot: angle in deg
    output_size: int or (w, h)
    """
    if isinstance(center, (tuple, list)):
        center = np.array(center, dtype=np.float32)

    if isinstance(scale, (int, float)):
        scale = np.array([scale, scale], dtype=np.float32)

    if isinstance(output_size, (int, float)):
        output_size = (output_size, output_size)

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


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)