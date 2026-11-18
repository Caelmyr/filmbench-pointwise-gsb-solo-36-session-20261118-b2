#!/usr/bin/env python3
"""FilmBench 点修补的独立验证脚本，使用 python -I 运行。"""
import sys

sys.path.insert(0, "/opt/pointwise-env/site-packages")
import numpy as np  # noqa: E402

sys.path.insert(0, "/workspace")

from filmbench.frame import (  # noqa: E402
    apply_retouch, frame_size, frame_to_scan, scan_to_frame,
)

rng = np.random.default_rng(7)
H, W = 120, 160
CROP = [20, 15, W - 20, H - 15]
size = (W, H)
base_settings = {"crop": CROP, "turn": 0, "mirror": False}

# 带亮暗渐变的干净正片（线性值），叠一层细颗粒。
gradient = np.linspace(0.25, 0.75, H, dtype=np.float32)[:, None, None]
clean = np.repeat(np.repeat(gradient, W, axis=1), 3, axis=2)
clean += rng.normal(0, 0.008, clean.shape).astype(np.float32)

interior = np.zeros((H, W), np.uint8)
interior[CROP[1]:CROP[3], CROP[0]:CROP[2]] = 1

# 画格外填成橙色片边。
orange = np.array([0.95, 0.45, 0.12], np.float32)
positive = np.broadcast_to(orange, (H, W, 3)).copy()
inside_mask = interior.astype(bool)
positive[inside_mask] = clean[inside_mask]

TX, TY, R = 60, 80, 9
SX, SY = 90, 40
dusty = positive.copy()
# 在目标处放一团明显的亮灰尘。
yy, xx = np.ogrid[:H, :W]
spot = (xx - TX) ** 2 + (yy - TY) ** 2 <= 9
dusty[spot] = np.array([0.98, 0.97, 0.95], np.float32)

healed = apply_retouch(dusty, {"source": [SX, SY], "target": [TX, TY], "radius": R}, interior)

failures = []


def check(name, condition, detail=""):
    print(("PASS" if condition else "FAIL"), name, detail)
    if not condition:
        failures.append(name)


# 1. 修补半径外的像素一个都不能动（含片边）。
disk = (xx - TX) ** 2 + (yy - TY) ** 2 <= R ** 2
outside = ~disk
check("半径外像素完全不变", np.array_equal(healed[outside], dusty[outside]))

# 2. 橙色片边像素绝不被修改。
border = ~inside_mask
check("橙色片边未被带入或修改", np.allclose(healed[border], orange, atol=1e-6))

# 3. 目标区域的低频应贴合周围渐变，而不是来源的底色；
#    与“无灰尘干净图”在目标处的局部均值对比，残差应远小于旧克隆法的圆印。
target_region = disk & inside_mask
clean_local_mean = clean[target_region].mean(axis=0)
healed_local_mean = healed[target_region].mean(axis=0)
old_clone_mean = positive[
    (xx - SX) ** 2 + (yy - SY) ** 2 <= R ** 2
].mean(axis=0)
new_drift = np.abs(healed_local_mean - clean_local_mean).max()
old_drift = np.abs(old_clone_mean - clean_local_mean).max()
check("目标低频延续原有渐变（漂移远小于旧克隆）", new_drift < 0.04 and new_drift < old_drift * 0.5,
      f"new={new_drift:.4f} old={old_drift:.4f}")

# 4. 灰尘应被去掉：目标中心不再是接近 1 的亮斑。
check("灰尘亮斑已消除", healed[TY, TX].max() < 0.85,
      f"center={healed[TY, TX].round(3).tolist()}")

# 5. 来源的细颗粒被借用：修补区域内的高频能量应与干净颗粒同量级，
#    而不是被抹成平板（纯低频填充的方差会接近零）。
def high_freq(patch):
    import cv2
    low = cv2.GaussianBlur(patch, (0, 0), 2.5)
    return patch - low


grain_energy = np.var(high_freq(healed[target_region].reshape(-1, 3)))
clean_energy = np.var(high_freq(clean[target_region].reshape(-1, 3)))
check("借用了细颗粒（纹理能量与干净区域同量级）",
      grain_energy > clean_energy * 0.35,
      f"healed={grain_energy:.6f} clean={clean_energy:.6f}")

# 6. 贴着画格边的灰尘：目标圆被画格边界截断，也不能写入片边。
edge_change = {"source": [40, 40], "target": [CROP[0] + 2, CROP[1] + 2], "radius": R}
edge_healed = apply_retouch(dusty, edge_change, interior)
check("贴边修补不修改片边", np.allclose(edge_healed[border], orange, atol=1e-6))
edge_disk = (xx - edge_change["target"][0]) ** 2 + (yy - edge_change["target"][1]) ** 2 <= R ** 2
check("贴边修补只动圆内像素", np.array_equal(edge_healed[~edge_disk], dusty[~edge_disk]))

# 7. 来源圆本身压在片边上时，片边颜色不能进入画格内的修补结果。
edge_source = {"source": [CROP[2] + 3, CROP[3] + 3], "target": [70, 70], "radius": R}
es_healed = apply_retouch(dusty, edge_source, interior)
es_disk = (xx - 70) ** 2 + (yy - 70) ** 2 <= R ** 2
check("来源含片边时不泄漏橙色",
      not np.any(np.abs(es_healed[es_disk] - orange).max(axis=1) < 0.15))

# 8. 坐标映射：正片每个像素都能映回画格内原片点，且内部点往返一致；
#    边界在 90° 转向时允许一个像素的取整差，单独用前向映射核对。
mapping_ok = True
for turn in range(4):
    for mirror in (False, True):
        settings = {**base_settings, "turn": turn, "mirror": mirror}
        fw, fh = frame_size(settings, size)
        if (fw, fh) != (W - 40, H - 30) and (fw, fh) != (H - 30, W - 40):
            mapping_ok = False
        # 正片画格上的每个点映回的原片坐标必须落在裁切框内。
        for fy in range(fh):
            for fx in (0, fw // 2, fw - 1):
                sx0, sy0 = frame_to_scan([fx, fy], settings, size)
                if not (CROP[0] <= sx0 < CROP[2] and CROP[1] <= sy0 < CROP[3]):
                    mapping_ok = False
        # 内缩两像素的点不受边界取整影响，必须严格往返。
        for p in ((CROP[0] + 2, CROP[1] + 2), (CROP[2] - 3, CROP[3] - 3), (W // 2, H // 2)):
            fp = scan_to_frame(p, settings, size)
            if frame_to_scan(fp, settings, size) != list(p):
                mapping_ok = False
        # 前向映射应当是画格范围上的双射（无重复、无遗漏）。
        forward = set()
        for sy0 in range(CROP[1], CROP[3]):
            for sx0 in range(CROP[0], CROP[2]):
                forward.add(tuple(scan_to_frame([sx0, sy0], settings, size)))
        if len(forward) != fw * fh:
            mapping_ok = False
check("原片↔正片坐标映射完整且内部往返一致", mapping_ok)

# 9. None 修补直接返回原图。
check("空修补为恒等", apply_retouch(dusty, None, interior) is dusty)

print()
if failures:
    print("失败项:", failures)
    sys.exit(1)
print("全部通过")
