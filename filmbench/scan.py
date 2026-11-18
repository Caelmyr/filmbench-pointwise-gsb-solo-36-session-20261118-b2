from dataclasses import dataclass
from io import BytesIO
import warnings

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from image_lab.validation import Invalid


MAX_UPLOAD = 28 * 1024 * 1024
MAX_PIXELS = 6_500_000
MAX_SIDE = 6500
PREVIEW_SIDE = 1300
FORMATS = {'PNG': ('png', 'image/png'), 'JPEG': ('jpg', 'image/jpeg'),
           'TIFF': ('tif', 'image/tiff')}


@dataclass(frozen=True)
class Scan:
    codes: np.ndarray
    format: str

    @property
    def depth(self):
        return 8 * self.codes.dtype.itemsize

    @property
    def size(self):
        height, width = self.codes.shape[:2]
        return width, height

    @property
    def maximum(self):
        return (1 << self.depth) - 1

    def linear(self, encoding):
        values = self.codes.astype(np.float32) / self.maximum
        return srgb_to_linear(values) if encoding == 'srgb' else values


def decode(raw):
    if not raw or len(raw) > MAX_UPLOAD:
        raise Invalid('扫描文件为空或超过28MB')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as header:
                fmt = header.format
                width, height = header.size
                if fmt not in FORMATS:
                    raise Invalid('请上传 PNG、JPEG 或单页 TIFF')
                if not (16 <= min(width, height) and max(width, height) <= MAX_SIDE
                        and width * height <= MAX_PIXELS):
                    raise Invalid('扫描须至少16×16像素，长边不超过6500且不超过650万像素')
                if getattr(header, 'n_frames', 1) != 1:
                    raise Invalid('只接受一页扫描，先拆分多页文件')
                if header.mode in ('CMYK', 'LAB', 'HSV', 'F'):
                    raise Invalid('请先将扫描转换为 RGB 或灰度')
        matrix = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise Invalid('扫描尺寸超出处理上限') from error
    except (UnidentifiedImageError, OSError, TypeError, cv2.error) as error:
        raise Invalid('扫描文件无法读取或已损坏') from error
    if matrix is None or matrix.shape[:2] != (height, width) or matrix.dtype not in (np.uint8, np.uint16):
        raise Invalid('仅支持8或16位无符号 RGB 或灰度扫描')
    if matrix.ndim == 2:
        codes = cv2.cvtColor(matrix, cv2.COLOR_GRAY2RGB)
    elif matrix.shape[2] == 3:
        codes = cv2.cvtColor(matrix, cv2.COLOR_BGR2RGB)
    else:
        raise Invalid('扫描不能含有透明或额外通道')
    codes.flags.writeable = False
    return Scan(codes, fmt)


def srgb_to_linear(values):
    return np.where(values <= .04045, values / 12.92,
                    np.power((values + .055) / 1.055, 2.4))


def linear_to_srgb(values):
    clipped = np.clip(values, 0, 1)
    return np.where(clipped <= .0031308, 12.92 * clipped,
                    1.055 * np.power(clipped, 1 / 2.4) - .055)


def compact(array, limit=PREVIEW_SIDE):
    height, width = array.shape[:2]
    if max(height, width) <= limit:
        return array
    ratio = limit / max(height, width)
    return cv2.resize(array, (max(1, round(width * ratio)), max(1, round(height * ratio))),
                      interpolation=cv2.INTER_AREA)


def display_rgb(values, depth=8):
    high = (1 << depth) - 1
    dtype = np.uint8 if depth == 8 else np.uint16
    return np.rint(linear_to_srgb(values) * high).astype(dtype)


def png_rgb(rgb):
    success, data = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not success:
        raise Invalid('图像无法编码为 PNG')
    return data.tobytes()


def tiff_rgb(rgb):
    success, data = cv2.imencode('.tiff', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_TIFF_COMPRESSION, 5])
    if not success:
        raise Invalid('图像无法编码为 TIFF')
    return data.tobytes()
