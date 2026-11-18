import numpy as np

from .density import CHANNEL_WEIGHTS


def histogram(values):
    rows = [np.histogram(channel, bins=64, range=(0, 1))[0].tolist()
            for channel in np.moveaxis(values, -1, 0)]
    return rows


def clipping(codes, limit):
    return np.any(codes == 0, axis=2), np.any(codes == limit, axis=2)


def overlay(linear, low, high):
    grey = (linear @ CHANNEL_WEIGHTS) * .53
    display = np.repeat(grey[..., None], 3, axis=2)
    display[low] = [.025, .36, .9]
    display[high] = [.94, .15, .022]
    display[low & high] = [.72, .067, .72]
    return display


def rounded(values):
    return np.round(values.astype(float), 5).tolist()


def summary(scan, base, source, density, positive):
    codes = scan.codes
    low, high = clipping(codes, scan.maximum)
    step = max(1, int(np.sqrt(source.shape[0] * source.shape[1] / 200000)))
    return {
        'base': {**base, 'transmission': rounded(base['transmission'])},
        'codes': {'depth': scan.depth, 'low': int(low.sum()), 'high': int(high.sum()),
                  'both': int((low & high).sum()), 'pixels': low.size},
        'source_histogram': histogram(codes[::step, ::step].astype(np.float32) / scan.maximum),
        'positive_histogram': histogram(positive[::step, ::step]),
        'density_midpoint': rounded(np.median(density[::step, ::step].reshape(-1, 3), axis=0)),
        'histogram_stride': step,
    }
