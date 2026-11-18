import cv2
import numpy as np

from .density import CHANNEL_WEIGHTS


# 与极限码值相差不到该比例的像素，视作可跨越的“截断边缘颗粒”
NEAR_LIMIT_TOLERANCE = 1 / 512
# 闭运算只允许跨过一两粒颗粒，真正有细节的窄缝不会被抹掉
BRIDGE_ITERATIONS = 2
# 小于该面积且最长边也不足该跨度的连通域按零星坏点忽略
SPECK_AREA = 16
SPECK_SPAN = 12
# 标注色：暗调表示零值截断，亮调表示满量程截断，色相区分通道
CLIP_COLORS = {
    ('R', 'low'): (.72, .10, .14),
    ('G', 'low'): (.16, .56, .20),
    ('B', 'low'): (.17, .27, .85),
    ('R', 'high'): (1.0, .42, .32),
    ('G', 'high'): (.50, .95, .36),
    ('B', 'high'): (.42, .62, 1.0),
}
CHANNELS = ('R', 'G', 'B')
POLARITIES = ('low', 'high')


def histogram(values):
    rows = [np.histogram(channel, bins=64, range=(0, 1))[0].tolist()
            for channel in np.moveaxis(values, -1, 0)]
    return rows


def rounded(values):
    return np.round(values.astype(float), 5).tolist()


def channel_masks(codes, limit):
    """每个通道分别给出零值、满量程和近极限颗粒掩膜。"""
    near_band = max(1, int(round(limit * NEAR_LIMIT_TOLERANCE)))
    exact_low = codes <= 0
    exact_high = codes >= limit
    near_low = codes <= near_band
    near_high = codes >= limit - near_band
    return exact_low, exact_high, near_low, near_high


def _components(mask, minimum_area=SPECK_AREA, minimum_span=SPECK_SPAN):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    pieces = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        # 零星坏点：又小又不成线；又细又连续的截断线靠跨度保留下来
        if area < minimum_area and max(width, height) < minimum_span:
            continue
        pieces.append((label, x, y, width, height, area))
    return labels, pieces


def clipping_analysis(codes, limit):
    """按通道、按截断类型拆分连通域，近极限颗粒只用于桥接同一处截断。"""
    exact_low, exact_high, near_low, near_high = channel_masks(codes, limit)
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    blocks = []
    frame_shape = codes.shape[:2]
    exact_maps = {(channel, polarity): np.zeros(frame_shape, bool)
                  for channel in CHANNELS for polarity in POLARITIES}
    kept_hard = np.zeros(frame_shape, bool)
    near_map = np.zeros(frame_shape, bool)
    label_matrix = np.zeros(frame_shape, np.int32)
    raw_low = np.zeros(frame_shape, bool)
    raw_high = np.zeros(frame_shape, bool)

    for channel_index, channel in enumerate(CHANNELS):
        raw_low |= exact_low[..., channel_index]
        raw_high |= exact_high[..., channel_index]
        for polarity, exact, other, near_polarity in (
                ('low', exact_low, exact_high, near_low),
                ('high', exact_high, exact_low, near_high)):
            hard = exact[..., channel_index]
            # 本极性的近极限颗粒；另一极性已硬截断的点不拿来桥接
            soft = near_polarity[..., channel_index] & ~hard & ~other[..., channel_index]
            # 近极限颗粒把只隔着一两粒颗粒的截断片连成同一处；
            # 没有贴上硬截断的近极限颗粒本身不会被标记
            bridged = cv2.morphologyEx(hard.astype(np.uint8), cv2.MORPH_CLOSE,
                                       bridge_kernel, iterations=BRIDGE_ITERATIONS)
            grown = cv2.dilate(bridged, bridge_kernel,
                               iterations=BRIDGE_ITERATIONS).astype(bool)
            joined = bridged.astype(bool) | (soft & grown)
            labels, pieces = _components(joined)
            kept = np.zeros(frame_shape, bool)
            for label, x, y, width, height, area in pieces:
                component = labels == label
                kept |= component
                thin_line = (min(width, height) <= 2 and max(width, height) >= SPECK_SPAN)
                blocks.append({
                    'id': len(blocks) + 1,
                    'channel': channel,
                    'kind': polarity,
                    'x': x, 'y': y, 'width': width, 'height': height,
                    'area': area,
                    'exact_pixels': int(np.count_nonzero(component & hard)),
                    'near_pixels': int(np.count_nonzero(component & soft)),
                    'thin_line': thin_line,
                })
                label_matrix[component] = len(blocks)
            exact_maps[(channel, polarity)] |= kept & hard
            kept_hard |= kept & hard
            near_map |= kept & soft & ~hard

    blocks.sort(key=lambda item: (item['y'], item['x'], item['channel'], item['kind']))
    # 排序后重新编号，并同步标签矩阵
    remapped = np.zeros_like(label_matrix)
    for new_id, block in enumerate(blocks, 1):
        old_id = block['id']
        remapped[label_matrix == old_id] = new_id
        block['id'] = new_id
    return {
        'blocks': blocks,
        'exact_maps': exact_maps,
        'near_map': near_map,
        'label_matrix': remapped,
        'raw_counts': {
            'low': int(np.count_nonzero(raw_low)),
            'high': int(np.count_nonzero(raw_high)),
            'both': int(np.count_nonzero(raw_low & raw_high)),
            'pixels': int(codes.shape[0] * codes.shape[1]),
            'ignored_specks': int(np.count_nonzero((raw_low | raw_high) & ~kept_hard)),
        },
        'tolerance': NEAR_LIMIT_TOLERANCE,
        'near_band_codes': max(1, int(round(limit * NEAR_LIMIT_TOLERANCE))),
    }


def frame_maps(analysis, settings, size):
    """所有掩膜和区块几何都走与画格完全相同的裁切、旋转、镜像变换。"""
    from .frame import prepared
    exact = {key: prepared(mask.astype(np.uint8), settings, size).astype(bool)
             for key, mask in analysis['exact_maps'].items()}
    near = prepared(analysis['near_map'].astype(np.uint8), settings, size).astype(bool)
    labels = prepared(analysis['label_matrix'], settings, size)
    framed_blocks = []
    for block in analysis['blocks']:
        ys, xs = np.where(labels == block['id'])
        if xs.size == 0:
            # 整个区块落在画格裁切框之外
            continue
        framed_blocks.append({
            **block,
            'x': int(xs.min()), 'y': int(ys.min()),
            'width': int(xs.max() - xs.min() + 1),
            'height': int(ys.max() - ys.min() + 1),
            'centroid_x': int(round(float(xs.mean()))),
            'centroid_y': int(round(float(ys.mean()))),
        })
    return exact, near, framed_blocks


def overlay(linear, exact_maps, near_map):
    """在画格灰度底上叠加截断标注；近极限桥用浅色，留出细节窄缝的可见度。"""
    grey = (linear @ CHANNEL_WEIGHTS) * .53
    display = np.repeat(grey[..., None], 3, axis=2)
    for key, mask in exact_maps.items():
        display[mask] = display[mask] * .18 + np.array(CLIP_COLORS[key]) * .82
    display[near_map] = display[near_map] * .55 + np.array((.86, .86, .82)) * .45
    return display


def summary(scan, base, source, density, positive, analysis=None):
    step = max(1, int(np.sqrt(source.shape[0] * source.shape[1] / 200000)))
    report = {
        'base': {**base, 'transmission': rounded(base['transmission'])},
        'codes': {'depth': scan.depth, 'pixels': scan.codes.shape[0] * scan.codes.shape[1]},
        'source_histogram': histogram(scan.codes[::step, ::step].astype(np.float32) / scan.maximum),
        'positive_histogram': histogram(positive[::step, ::step]),
        'density_midpoint': rounded(np.median(density[::step, ::step].reshape(-1, 3), axis=0)),
        'histogram_stride': step,
    }
    if analysis is not None:
        blocks = analysis['blocks']
        report['clipping'] = {
            **analysis['raw_counts'],
            'near_band_codes': analysis['near_band_codes'],
            'tolerance': analysis['tolerance'],
            'block_count': len(blocks),
            'blocks': [{'id': block['id'], 'channel': block['channel'], 'kind': block['kind'],
                        'x': block['x'], 'y': block['y'], 'width': block['width'],
                        'height': block['height'], 'area': block['area'],
                        'exact_pixels': block['exact_pixels'],
                        'near_pixels': block['near_pixels'],
                        'thin_line': block['thin_line']} for block in blocks],
        }
    return report
