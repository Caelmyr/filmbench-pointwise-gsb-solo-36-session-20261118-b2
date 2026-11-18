import cv2
import numpy as np


def bounds(settings, size):
    return settings['crop'] or [0, 0, *size]


def prepared(array, settings, size):
    x0, y0, x1, y1 = bounds(settings, size)
    window = np.rot90(array[y0:y1, x0:x1], -settings['turn'])
    return window[:, ::-1] if settings['mirror'] else window


def frame_size(settings, size):
    left, top, right, bottom = bounds(settings, size)
    width, height = right - left, bottom - top
    return [height, width] if settings['turn'] % 2 else [width, height]


def scan_to_frame(point, settings, size):
    """原片坐标（未转向、左上原点）换算成当前正片画格里的坐标。"""
    x, y = point
    x0, y0, x1, y1 = bounds(settings, size)
    lx, ly = x - x0, y - y0
    width, height = x1 - x0, y1 - y0
    turn = settings['turn']
    if turn == 0:
        fx, fy = lx, ly
    elif turn == 1:
        fx, fy = ly, width - 1 - lx
    elif turn == 2:
        fx, fy = width - 1 - lx, height - 1 - ly
    else:
        fx, fy = height - 1 - ly, lx
    if settings['mirror']:
        frame_width = height if turn % 2 else width
        fx = frame_width - 1 - fx
    return [fx, fy]


def frame_to_scan(point, settings, size):
    """当前正片画格里的点击位置换算回原片坐标。"""
    fx, fy = point
    x0, y0, x1, y1 = bounds(settings, size)
    width, height = x1 - x0, y1 - y0
    turn = settings['turn']
    if settings['mirror']:
        frame_width = height if turn % 2 else width
        fx = frame_width - 1 - fx
    if turn == 0:
        lx, ly = fx, fy
    elif turn == 1:
        lx, ly = width - 1 - fy, fx
    elif turn == 2:
        lx, ly = width - 1 - fx, height - 1 - fy
    else:
        lx, ly = fy, height - 1 - fx
    return [x0 + lx, y0 + ly]


def _masked_blur(values, weight, sigma):
    """只按 weight>0 的像素做加权高斯模糊，边界不会被零像素拉暗。"""
    kernel = max(3, int(round(sigma * 6)) | 1)
    weighted = values * weight[..., None]
    summed = cv2.GaussianBlur(weighted, (kernel, kernel), sigma)
    shared = cv2.GaussianBlur(weight, (kernel, kernel), sigma)
    return summed / np.maximum(shared[..., None], np.finfo(np.float32).tiny)


def _grain(values, weight, radius):
    """从干净区域里分离出接近零均值的细颗粒。"""
    sigma = max(1.2, radius * .28)
    low = _masked_blur(values, weight, sigma)
    grain = values - low
    grain -= cv2.GaussianBlur(grain * weight[..., None], (0, 0), max(2.0, radius * .9)) \
        / np.maximum(cv2.GaussianBlur(weight, (0, 0), max(2.0, radius * .9)),
                     np.finfo(np.float32).tiny)[..., None]
    return grain


def apply_retouch(positive, change, interior=None):
    if not change:
        return positive
    sx, sy = change['source']
    tx, ty = change['target']
    radius = change['radius']
    height, width = positive.shape[:2]

    # 修补窗口：以目标为中心、半径为界，绝不写窗口之外的任何像素。
    x0 = max(0, tx - radius)
    y0 = max(0, ty - radius)
    x1 = min(width, tx + radius + 1)
    y1 = min(height, ty + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return positive

    y, x = np.ogrid[y0:y1, x0:x1]
    distance = np.sqrt((x - tx) ** 2 + (y - ty) ** 2) / radius
    feather = np.clip((1 - distance) / .35, 0, 1).astype(np.float32)
    feather *= feather * (3 - 2 * feather)

    # 画格内部掩膜：贴着片边时，橙色片边像素不参与来源采样也不被修改。
    inside = np.ones((y1 - y0, x1 - x0), np.float32)
    if interior is not None:
        # 硬阈值，避免浮点零值经乘法后仍产生微小 alpha 而改动片边像素。
        inside = (interior[y0:y1, x0:x1] > .5).astype(np.float32)
    feather *= inside

    # 来源窗口与目标窗口同尺寸，越界部分通过掩膜剔除，避免把片边卷进来。
    dx, dy = sx - tx, sy - ty
    donor_x0, donor_y0 = x0 + dx, y0 + dy
    donor = np.zeros_like(positive[y0:y1, x0:x1])
    source_valid = np.zeros((y1 - y0, x1 - x0), np.float32)
    copy_x0 = max(0, donor_x0)
    copy_y0 = max(0, donor_y0)
    copy_x1 = min(width, donor_x0 + (x1 - x0))
    copy_y1 = min(height, donor_y0 + (y1 - y0))
    if copy_x0 < copy_x1 and copy_y0 < copy_y1:
        local_x0, local_y0 = copy_x0 - donor_x0, copy_y0 - donor_y0
        local_x1 = local_x0 + (copy_x1 - copy_x0)
        local_y1 = local_y0 + (copy_y1 - copy_y0)
        donor[local_y0:local_y1, local_x0:local_x1] = \
            positive[copy_y0:copy_y1, copy_x0:copy_x1]
        source_valid[local_y0:local_y1, local_x0:local_x1] = 1
        if interior is not None:
            source_valid[local_y0:local_y1, local_x0:local_x1] *= \
                interior[copy_y0:copy_y1, copy_x0:copy_x1]

    target = positive[y0:y1, x0:x1]
    # 目标低频只取灰尘周围的干净环带：核心里的灰尘不参与均值，
    # 这样亮暗渐变完全由周围像素延续，亮斑不会被糊成一团浅色底。
    ring = inside * np.clip((distance - .55) / .45, 0, 1)
    if ring.sum() < 8:  # 极小画格的退化情况：退回全部内部像素
        ring = inside
    target_low = _masked_blur(target, ring.astype(np.float32), max(2.0, radius * .75))

    # 来源只贡献细颗粒；来源不可用的像素退回纯目标低频，不引入片边。
    donor_weight = feather * source_valid
    grain = _grain(donor, source_valid, radius) * source_valid[..., None]
    donor_coverage = cv2.GaussianBlur(donor_weight, (0, 0), max(1.0, radius * .5))
    donor_coverage = np.clip(donor_coverage / np.maximum(feather, np.finfo(np.float32).tiny), 0, 1)
    healed = target_low + grain * donor_coverage[..., None]

    output = positive.copy()
    patch = output[y0:y1, x0:x1]
    patch *= 1 - feather[..., None]
    patch += healed * feather[..., None]
    return output
