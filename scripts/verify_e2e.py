#!/usr/bin/env python3
"""端到端冒烟：演示工程的截断 metrics、预览图和画格几何。"""
import sys
import tempfile

sys.path.insert(0, '/opt/pointwise-env/site-packages')
sys.path.insert(0, '/workspace')

from flask import Flask

from filmbench.server import register


def main():
    with tempfile.TemporaryDirectory() as data_dir:
        app = Flask(__name__)
        register(app, data_dir)
        client = app.test_client()

        for key, expected_min_blocks in (('market', 5), ('harbor', 0)):
            created = client.post(f'/film/demos/{key}')
            assert created.status_code == 201, created.data
            project = created.get_json()
            pid = project['id']

            metrics = client.get(f'/film/projects/{pid}/metrics')
            assert metrics.status_code == 200
            report = metrics.get_json()
            clip = report['clipping']
            print(key, '截断区块:', clip['block_count'], '忽略坏点:', clip['ignored_specks'])
            for block in clip['blocks']:
                print('  #', block['id'], block['channel'], block['kind'],
                      block['width'], 'x', block['height'],
                      'area', block['area'], 'near', block['near_pixels'],
                      'thin' if block['thin_line'] else '')
            assert clip['block_count'] >= expected_min_blocks
            if key == 'market':
                # 细线区块存在且被标记 thin_line
                assert any(b['thin_line'] and b['height'] == 1 for b in clip['blocks'])
                # 近极限桥接的绿通道零值合并为一处
                green_low = [b for b in clip['blocks']
                             if b['channel'] == 'G' and b['kind'] == 'low']
                assert len(green_low) == 1 and green_low[0]['near_pixels'] > 0
                # 细节窄缝两侧的蓝满量程保持两个区块
                blue_high = [b for b in clip['blocks']
                             if b['channel'] == 'B' and b['kind'] == 'high']
                assert len(blue_high) == 2
                # 区块坐标都在画格范围内（frame_size 为 [宽, 高]）
                fw, fh = project['frame_size']
                for block in clip['blocks']:
                    assert 0 <= block['x'] and block['x'] + block['width'] <= fw
                    assert 0 <= block['y'] and block['y'] + block['height'] <= fh
            else:
                assert clip['block_count'] == 0

            preview = client.get(f'/film/projects/{pid}/preview/clipping.png')
            assert preview.status_code == 200 and preview.data[:8] == b'\x89PNG\r\n\x1a\n'
            # 正片与密度视图仍正常
            for view in ('positive', 'density', 'source', 'reference'):
                response = client.get(f'/film/projects/{pid}/preview/{view}.png')
                assert response.status_code == 200 and response.data[:8] == b'\x89PNG\r\n\x1a\n', view

        # 修改裁切/旋转后，区块坐标跟随画格变化
        project = client.post('/film/demos/market').get_json()
        pid = project['id']
        before = client.get(f'/film/projects/{pid}/metrics').get_json()['clipping']
        patch = client.patch(f'/film/projects/{pid}', json={
            'version': project['version'],
            'changes': {'turn': 1, 'crop': [60, 60, 900, 560]},
        })
        assert patch.status_code == 200, patch.data
        after = client.get(f'/film/projects/{pid}/metrics').get_json()['clipping']
        rotated = patch.get_json()
        fw, fh = rotated['frame_size']
        for block in after['blocks']:
            assert 0 <= block['x'] and block['x'] + block['width'] <= fw
            assert 0 <= block['y'] and block['y'] + block['height'] <= fh
        assert len(after['blocks']) <= len(before['blocks'])
        preview = client.get(f'/film/projects/{pid}/preview/clipping.png')
        assert preview.status_code == 200
        print('PASS 旋转+重新裁切后截断标注仍在画格内')
        print('\n端到端冒烟全部通过')


if __name__ == '__main__':
    main()
