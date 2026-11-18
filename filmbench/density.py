import numpy as np

from image_lab.validation import Invalid


CHANNEL_WEIGHTS = np.array([.2126, .7152, .0722], dtype=np.float32)
MIN_TRANSMISSION = .0001


def border_pixels(linear):
    height, width = linear.shape[:2]
    step = max(1, min(height, width) // 500)
    band = max(2, round(min(height, width) * .025))
    return np.concatenate((linear[:band:step, ::step].reshape(-1, 3),
                           linear[-band::step, ::step].reshape(-1, 3),
                           linear[band:-band:step, :band:step].reshape(-1, 3),
                           linear[band:-band:step, -band::step].reshape(-1, 3)))


def check_region(codes, selection):
    if not selection:
        return
    x0, y0, x1, y1 = selection
    patch = codes[y0:y1:max(1, (y1 - y0) // 300), x0:x1:max(1, (x1 - x0) // 300)]
    if np.any(np.median(patch, axis=(0, 1)) == 0):
        raise Invalid('片基选区至少有一个通道接近黑场，请选透光片边')


def sample_base(linear, codes, maximum, selection):
    if selection is not None:
        x0, y0, x1, y1 = selection
        patch = linear[y0:y1, x0:x1]
        stride = max(1, int(np.sqrt(patch.shape[0] * patch.shape[1] / 180000)))
        pixels = patch[::stride, ::stride].reshape(-1, 3)
        native = codes[y0:y1:stride, x0:x1:stride].reshape(-1, 3)
        clipped_fraction = float(np.mean(np.any((native == 0) | (native == maximum), axis=1)))
        mode = 'selection'
    else:
        border = border_pixels(linear)
        pool = border[np.all((border > MIN_TRANSMISSION) & (border < .9995), axis=1)]
        if len(pool) < max(8, len(border) // 50):
            pool = border
        level = pool @ CHANNEL_WEIGHTS
        pixels = pool[level >= np.quantile(level, .86)]
        clipped_fraction = float(np.mean(np.any((pixels <= 0) | (pixels >= 1), axis=1)))
        mode = 'edge_estimate'
    return {'transmission': np.median(pixels, axis=0).clip(MIN_TRANSMISSION, 1).astype(np.float32),
            'mode': mode, 'count': len(pixels), 'clipped_fraction': clipped_fraction}


def relative_density(linear, base):
    ratio = linear / base
    np.clip(ratio, MIN_TRANSMISSION, 1, out=ratio)
    np.log10(ratio, out=ratio)
    return -ratio


def develop(density, tone):
    positive = density + np.array([tone['red'], tone['green'], tone['blue']], dtype=np.float32)
    positive += .16 * tone['exposure'] - tone['black']
    positive /= tone['white'] - tone['black']
    positive -= .5
    positive *= tone['contrast']
    positive += .5
    np.clip(positive, 0, 1, out=positive)
    return positive


def density_view(density):
    return np.clip(density @ CHANNEL_WEIGHTS / 1.6, 0, 1)
