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


def apply_retouch(positive, change):
    if not change:
        return positive
    sx, sy = change['source']
    tx, ty = change['target']
    radius = change['radius']
    height, width = positive.shape[:2]
    x0 = max(0, tx - radius, tx - sx)
    y0 = max(0, ty - radius, ty - sy)
    x1 = min(width, tx + radius + 1, tx + width - sx)
    y1 = min(height, ty + radius + 1, ty + height - sy)
    if x0 >= x1 or y0 >= y1:
        return positive
    y, x = np.ogrid[y0:y1, x0:x1]
    distance = np.sqrt((x - tx) ** 2 + (y - ty) ** 2) / radius
    alpha = np.clip((1 - distance) / .44, 0, 1).astype(np.float32)[..., None]
    donor = positive[y0 + sy - ty:y1 + sy - ty, x0 + sx - tx:x1 + sx - tx]
    output = positive.copy()
    target = output[y0:y1, x0:x1]
    target *= 1 - alpha
    target += donor * alpha
    return output
