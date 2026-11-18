#!/usr/bin/env python3
"""FilmBench 点修补的端到端验证：Flask 测试客户端 + 临时数据目录。"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, "/workspace")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from filmbench.server import register  # noqa: E402
from image_lab import create_app  # noqa: E402

failures = []


def check(name, condition, detail=""):
    print(("PASS" if condition else "FAIL"), name, detail)
    if not condition:
        failures.append(name)


with tempfile.TemporaryDirectory() as data_dir:
    app = create_app(Path(data_dir) / "preparation")
    register(app, Path(data_dir) / "negatives")
    client = app.test_client()

    created = client.post("/film/demos/market")
    check("创建演示工程", created.status_code == 201)
    project = created.get_json()
    pid = project["id"]
    version = project["version"]
    settings = project["settings"]
    fw, fh = project["frame_size"]

    # 正片视图里把灰尘目标选在画格内部、来源选在上方干净区域。
    change = {"source": [fw // 2, max(8, fh // 4)],
              "target": [fw // 2 + 4, fh // 2], "radius": 9}
    response = client.post(f"/film/projects/{pid}/retouch",
                           json={"version": version, "change": change})
    check("正片坐标修补端点 200", response.status_code == 200, response.data[:120].decode())
    updated = response.get_json()
    check("修订号递增", updated["version"] == version + 1)

    marks = updated["retouch_view"]
    check("返回的画格标记就是点击位置",
          marks["source"] == change["source"] and marks["target"] == change["target"]
          and marks["radius"] == 9)

    # 存档里的修补坐标必须落在原片裁切框内，且与画格点击位置映射一致。
    from filmbench.frame import frame_to_scan
    stored = updated["settings"]["retouch"]
    crop = settings["crop"]
    in_crop = all(crop[0] <= p[0] < crop[2] and crop[1] <= p[1] < crop[3]
                  for p in (stored["source"], stored["target"]))
    check("存档原片坐标落在裁切框内", in_crop)
    check("存档坐标与点击位置互逆",
          frame_to_scan(change["source"], updated["settings"], updated["size"]) == stored["source"]
          and frame_to_scan(change["target"], updated["settings"], updated["size"]) == stored["target"])

    # 预览与下载都能生成，且下载的仍是正片 PNG。
    preview = client.get(f"/film/projects/{pid}/preview/positive.png?v={updated['version']}")
    check("正片预览生成", preview.status_code == 200 and preview.data[:8] == b"\x89PNG\r\n\x1a\n")
    download = client.get(f"/film/projects/{pid}/export/positive.png")
    check("正片 PNG 下载", download.status_code == 200
          and download.headers["Content-Type"] == "image/png"
          and download.data[:8] == b"\x89PNG\r\n\x1a\n")
    png = cv2.imdecode(np.frombuffer(download.data, np.uint8), cv2.IMREAD_UNCHANGED)
    check("下载尺寸等于画格尺寸", png.shape[1] == fw and png.shape[0] == fh,
          f"{png.shape} vs {fh}x{fw}")

    before = cv2.imdecode(np.frombuffer(preview.data, np.uint8), cv2.IMREAD_UNCHANGED)
    check("预览与下载内容一致", np.array_equal(before, png))

    # 陈旧修订号必须被拒绝。
    stale = client.post(f"/film/projects/{pid}/retouch",
                        json={"version": version, "change": change})
    check("旧修订号冲突", stale.status_code == 409)

    # 越界画格坐标必须被拒绝。
    bad = client.post(f"/film/projects/{pid}/retouch",
                      json={"version": updated["version"],
                            "change": {"source": [0, 0], "target": [fw, fh], "radius": 9}})
    check("越界坐标被拒", bad.status_code == 400)

    # 缺字段必须被拒绝。
    missing = client.post(f"/film/projects/{pid}/retouch",
                          json={"version": updated["version"], "change": {"radius": 9}})
    check("缺字段被拒", missing.status_code == 400)

    # 移除修补后恢复原样。
    reset = client.patch(f"/film/projects/{pid}",
                         json={"version": updated["version"], "changes": {"retouch": None}})
    check("移除修补", reset.status_code == 200 and reset.get_json()["settings"]["retouch"] is None)

    # 旋转+镜像后再修补，验证坐标链路。
    turned = client.patch(f"/film/projects/{pid}",
                          json={"version": reset.get_json()["version"],
                                "changes": {"turn": 1, "mirror": True}})
    check("旋转镜像设置", turned.status_code == 200)
    tproject = turned.get_json()
    tfw, tfh = tproject["frame_size"]
    tchange = {"source": [12, 13], "target": [tfw - 14, tfh - 15], "radius": 7}
    tret = client.post(f"/film/projects/{pid}/retouch",
                       json={"version": tproject["version"], "change": tchange})
    check("旋转镜像后修补成功", tret.status_code == 200, tret.data[:120].decode())
    tmarks = tret.get_json()["retouch_view"]
    check("旋转镜像后画格标记保持点击位置",
          tmarks["source"] == tchange["source"] and tmarks["target"] == tchange["target"])
    tdl = client.get(f"/film/projects/{pid}/export/positive.png")
    tpng = cv2.imdecode(np.frombuffer(tdl.data, np.uint8), cv2.IMREAD_UNCHANGED)
    check("旋转镜像后下载尺寸正确", tpng.shape[1] == tfw and tpng.shape[0] == tfh)

    # 扫描原档下载不受修补影响。
    raw = client.get(f"/film/projects/{pid}/export/original")
    check("原档仍可下载", raw.status_code == 200)

print()
if failures:
    print("失败项:", failures)
    sys.exit(1)
print("全部通过")
