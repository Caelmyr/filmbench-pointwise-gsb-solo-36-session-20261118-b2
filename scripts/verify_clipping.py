#!/usr/bin/env python3
"""截断区块分析的行为验证：坏点、细线、近极限桥接、细节窄缝、画格几何。"""
import sys

sys.path.insert(0, '/opt/pointwise-env/site-packages')
sys.path.insert(0, '/workspace')

import numpy as np

from filmbench.diagnostics import (BRIDGE_ITERATIONS, NEAR_LIMIT_TOLERANCE,
                                   clipping_analysis, frame_maps, summary)
from filmbench.scan import Scan
from filmbench.settings import initial_settings


def make_scan(height=120, width=160, limit=255):
    middle = limit // 2
    dtype = np.uint8 if limit == 255 else np.uint16
    return np.full((height, width, 3), middle, dtype)


def blocks_of(analysis, channel, kind):
    return [b for b in analysis['blocks'] if b['channel'] == channel and b['kind'] == kind]


def test_specks_ignored_but_lines_kept():
    codes = make_scan()
    # 四个互不相连的红通道零值坏点
    codes[10, 12, 0] = 0
    codes[20, 40, 0] = 0
    codes[90, 130, 0] = 0
    codes[70, 80, 0] = 0
    # 一条又细又连续的绿通道零值线（1px 宽，20px 长）
    codes[50, 30:50, 1] = 0
    analysis = clipping_analysis(codes, 255)
    red_low = blocks_of(analysis, 'R', 'low')
    green_low = blocks_of(analysis, 'G', 'low')
    assert red_low == [], f'零星坏点不应成块: {red_low}'
    assert len(green_low) == 1, '细截断线应保留'
    assert green_low[0]['thin_line'] is True
    assert green_low[0]['width'] >= 20
    assert analysis['raw_counts']['low'] >= 24
    scan = Scan(codes, 'PNG')
    linear = codes.astype(np.float32) / 255
    base = {'transmission': np.ones(3, np.float32), 'mode': 'edge_estimate',
            'count': 4, 'clipped_fraction': 0.0}
    report = summary(scan, base, linear, linear.copy(), linear.copy(), analysis)
    assert report['clipping']['ignored_specks'] >= 4
    print('PASS 零星坏点被忽略，又细又连续的截断线保留')


def test_near_limit_grains_merge():
    codes = make_scan()
    near_band = max(1, round(255 * NEAR_LIMIT_TOLERANCE))
    # 两片硬零值，中间隔 4 列近零值颗粒（在容差带内）
    codes[30:50, 20:40, 2] = 0
    codes[30:50, 44:64, 2] = 0
    codes[30:50, 40:44, 2] = np.arange(1, near_band + 1)[:4][None, :]
    analysis = clipping_analysis(codes, 255)
    blue_low = blocks_of(analysis, 'B', 'low')
    assert len(blue_low) == 1, f'近零值颗粒隔开的两片应合并: {len(blue_low)} 块'
    block = blue_low[0]
    assert block['near_pixels'] > 0, '区块应记录并入的近极限颗粒'
    assert block['x'] <= 20 and block['width'] >= 44
    print('PASS 隔着近极限颗粒的截断片认作同一处，并记录近极限颗粒数')


def test_detail_seam_kept_separate():
    codes = make_scan()
    # 两片满量程，中间 10 列是真正有细节的中间调窄缝
    codes[60:90, 20:50, 0] = 255
    codes[60:90, 60:90, 0] = 255
    codes[60:90, 50:60, 0] = np.linspace(60, 180, 10)[None, :]
    analysis = clipping_analysis(codes, 255)
    red_high = blocks_of(analysis, 'R', 'high')
    assert len(red_high) == 2, f'有细节的窄缝不能被抹掉: {len(red_high)} 块'
    left, right = sorted(red_high, key=lambda b: b['x'])
    assert left['x'] == 20 and right['x'] == 60
    # 窄缝本身不被标记
    exact, near, _ = frame_maps(analysis, initial_settings('srgb'), (160, 120))
    seam_masked = exact[('R', 'high')][60:90, 50:60].any()
    assert not seam_masked, '细节窄缝不应被标注覆盖'
    print(f'PASS 真正有细节的窄缝保留，两片截断各自独立（桥接半径 {BRIDGE_ITERATIONS} 轮）')


def test_channel_and_polarity_separation():
    codes = make_scan()
    codes[20:35, 20:35, 0] = 0          # 红零值
    codes[20:35, 60:75, 1] = 255        # 绿满量程
    codes[60:80, 20:40, 2] = 255        # 蓝满量程
    analysis = clipping_analysis(codes, 255)
    by_key = {(b['channel'], b['kind']) for b in analysis['blocks']}
    assert by_key == {('R', 'low'), ('G', 'high'), ('B', 'high')}, by_key
    print('PASS 区块按通道和零值/满量程分别标注')


def test_frame_geometry_follows_crop_rotation_mirror():
    codes = make_scan(200, 240)
    crop = [40, 30, 200, 170]  # 宽 160、高 140
    # 在原片坐标 (x=100, y=80) 附近放一块红满量程
    codes[80:100, 100:120, 0] = 255
    settings = initial_settings('srgb', crop=crop)
    settings['turn'] = 1
    settings['mirror'] = True
    analysis = clipping_analysis(codes, 255)
    exact, near, blocks = frame_maps(analysis, settings, (240, 200))
    assert len(blocks) == 1
    block = blocks[0]
    # 变换后画格尺寸：旋转后宽 140、高 160
    framed = exact[('R', 'high')]
    assert framed.shape == (160, 140), framed.shape
    # 区块掩膜与编号包围盒必须重合在同一处
    ys, xs = np.where(framed)
    assert block['x'] == xs.min() and block['y'] == ys.min()
    assert block['width'] == xs.max() - xs.min() + 1
    assert block['height'] == ys.max() - ys.min() + 1
    assert near.sum() == 0
    print('PASS 截断标注跟随裁切、旋转、镜像，编号包围盒与掩膜对齐')


def test_sixteen_bit_tolerance():
    limit = 65535
    codes = make_scan(120, 160, limit).astype(np.uint16)
    near_band = max(1, round(limit * NEAR_LIMIT_TOLERANCE))
    assert near_band == 128
    codes[30:50, 20:40, 1] = limit
    codes[30:50, 44:64, 1] = limit
    codes[30:50, 40:44, 1] = limit - np.array([12, 40, 90, 127])[None, :]
    analysis = clipping_analysis(codes, limit)
    green_high = blocks_of(analysis, 'G', 'high')
    assert len(green_high) == 1 and green_high[0]['near_pixels'] == 4 * 20
    # 超出容差带的颗粒不参与桥接；缝隙加宽到闭运算半径之外，必须分成两片
    codes[30:50, 40:48, 1] = limit - 200
    analysis = clipping_analysis(codes, limit)
    green_high = blocks_of(analysis, 'G', 'high')
    assert len(green_high) == 2
    print('PASS 16 位扫描按比例容差（±128 码值）桥接，超差不合并')


def test_blocks_outside_crop_are_skipped():
    codes = make_scan(200, 240)
    # 区块 A 在裁切框内；区块 B 完全在框外；区块 C 被裁切框切到一半
    codes[80:100, 80:100, 0] = 255
    codes[10:30, 10:30, 1] = 0
    codes[120:140, 180:220, 2] = 0
    settings = initial_settings('srgb', crop=[40, 40, 200, 160])
    analysis = clipping_analysis(codes, 255)
    exact, near, blocks = frame_maps(analysis, settings, (240, 200))
    ids = {b['id'] for b in blocks}
    assert len(blocks) == 2, f'框外区块应跳过: {[(b["channel"], b["x"], b["y"]) for b in blocks]}'
    partial = next(b for b in blocks if b['channel'] == 'B')
    # 部分入框的区块只保留框内部分（原宽 40，框右边界 x=199 切掉一半）
    assert partial['x'] + partial['width'] <= 160
    assert near.sum() >= 0
    print('PASS 裁切框外的区块被跳过，部分入框的区块按画格裁剪')


if __name__ == '__main__':
    test_specks_ignored_but_lines_kept()
    test_near_limit_grains_merge()
    test_detail_seam_kept_separate()
    test_channel_and_polarity_separation()
    test_frame_geometry_follows_crop_rotation_mirror()
    test_sixteen_bit_tolerance()
    test_blocks_outside_crop_are_skipped()
    print('\n全部验证通过')
