# %%
import os 
import os.path as osp

import pickle
from tqdm import tqdm
import numpy as np

from plyfile import PlyElement, PlyData

import utils

# %% [markdown]
# # read pointcloud ply data

# %%
ply_file = "/home/fredom/datasets/powergrid/out/ply_fps/towerline/sljx/001-010.ply"

# %%
with open(ply_file, "rb") as f:
	ply_data = PlyData.read(f)["vertex"].data[::10]

coords = np.stack([
	ply_data["x"], ply_data["y"], ply_data["z"]
], axis=1)

colors = np.stack([
	ply_data["red"], ply_data["green"], ply_data["blue"]
], axis=1)

print(coords.shape)

# %% [markdown]
# # first height filtration

# %%
# 执行基于高程信息的基础滤波。由于输电线路可能跨越不同地形，
# 直接对整个点云计算全局高程阈值可能不准确，因此采用分段处理策略。
ppos = 0
step = int(1e4)  # 每个批次的点数量
pivot_list = []  # 存储每个批次计算出的高程阈值（基准值）
stapt_list = []  # 存储每个批次的起始索引

num_points = len(coords)

# 第一步：将点云分成多个批次，并计算每个批次的高程基准值（如平均值）
while ppos < num_points:
	# 获取当前批次点的高程（Z坐标）并计算基准值（例如平均值）
	pivot = utils.height_hist_filter(coords[ppos:min(ppos + step, num_points), 2])
	pivot_list.append(pivot)
	stapt_list.append(ppos)
	ppos += step

# 第二步：基于每个批次的基准值，生成滤波掩码（布尔数组）
height_filter_mask = np.zeros((0,), dtype=bool)  # 初始化为空的布尔数组

# 遍历每个批次，生成该批次内点的高程是否大于基准值的布尔掩码
for stapt, pivot in tqdm(zip(stapt_list, pivot_list), desc="生成高程滤波掩码", total=len(pivot_list), ncols=100):
	# stapt + step 是当前批次的预期结束位置，min() 确保不超过总点数。
	batch_end = min(stapt + step, num_points)
	# 计算当前批次内每个点的高程是否大于该批次的基准值
	batch_mask = coords[stapt:batch_end, 2] > pivot
	# 将当前批次的掩码拼接到总掩码中
	height_filter_mask = np.concatenate((height_filter_mask, batch_mask))

# 第三步：使用生成的掩码从原始数据中筛选出符合条件的点
# 掩码 height_filter_mask 的长度必须与 coords 的行数完全一致
coords_hf = coords[height_filter_mask]
colors_hf = colors[height_filter_mask]

# 输出滤波后的点云
# utils.npy2ply(coords_hf, colors_hf, "./output/pipeline_height_filtered.ply", use_txt=False)

# %%
# 多进程处理子空间坐标几何特征计算
from concurrent.futures import ProcessPoolExecutor, as_completed

# line_feat_mat_ckpt_path = "./output/line_feat_mat.pkl"
# if os.path.exists(line_feat_mat_ckpt_path):
#     with open(line_feat_mat_ckpt_path, "rb") as f:
#         line_feat_mat = pickle.load(f)
# else:
num_workers = 4
batch_size = coords_hf.shape[0] // num_workers
line_feat_mat_list = [None] * num_workers
num_neighbour_list = [None] * num_workers
with ProcessPoolExecutor(max_workers=num_workers) as executor:
	futures_dict = {
		executor.submit(
			utils.eigval_radius,
			coords_hf[i * batch_size : (i + 1) * batch_size],
			7.5
		) : i for i in range(num_workers)
	}
	for future in as_completed(futures_dict):
		feat_mat, neig_lst = future.result()
		line_feat_mat_list[futures_dict[future]] = feat_mat
		num_neighbour_list[futures_dict[future]] = neig_lst
line_feat_mat = np.concatenate(line_feat_mat_list, axis=0)
num_neighbour = np.concatenate(num_neighbour_list, axis=0)
# os.makedirs(osp.dirname(line_feat_mat_ckpt_path), exist_ok=True)
# with open(line_feat_mat_ckpt_path, "wb") as f:
# 	pickle.dump(line_feat_mat, f)



# %%
import numpy as np
import open3d as o3d
from tqdm import tqdm

def knn_fill_features(
	points: np.ndarray,
	feat: np.ndarray,
	num_neigh: np.ndarray,
	valid_thr: int = 10,   # 半径搜索邻居数阈值，小于它就认为不可靠
	knn: int = 100,        # 先搜这么多近邻
	m: int = 10,           # 从近邻里取前 m 个有效点插值
	method: str = 'idw',   # idw 或 gaussian
	sigma: float | None = None,
	eps: float = 1e-6,
):
	n = len(points)
	assert feat.shape == (n, 3)
	assert num_neigh.shape == (n,)

	pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
	kdtree = o3d.geometry.KDTreeFlann(pcd)

	out = feat.copy()
	valid = num_neigh >= valid_thr

	for i in tqdm(range(n), desc='knn fill', ncols=100):
		if valid[i]:
			continue

		k, idx, d2 = kdtree.search_knn_vector_3d(points[i], knn)
		idx = np.asarray(idx)
		d2 = np.asarray(d2)

		# 去掉自己
		mask_not_self = idx != i
		idx = idx[mask_not_self]
		d2 = d2[mask_not_self]

		# 只用有效点
		mask_valid = valid[idx]
		idx_v = idx[mask_valid]
		d2_v = d2[mask_valid]

		if idx_v.size == 0:
			continue

		# 取最近的前 m 个有效点
		if idx_v.size > m:
			idx_v = idx_v[:m]
			d2_v = d2_v[:m]

		if method == 'gaussian':
			if sigma is None:
				# 用第 m 个邻居距离做一个自适应尺度
				sigma = np.sqrt(d2_v[-1] + eps)
			w = np.exp(-d2_v / (2.0 * sigma * sigma + eps))
		else:
			# 逆距离加权
			w = 1.0 / (np.sqrt(d2_v) + eps)

		w = w / (w.sum() + eps)
		out[i] = (out[idx_v] * w[:, None]).sum(axis=0)

	return out

# %%
# 因为使用了多进程分组处理，所以有可能原先分组不能整除刚好分给各个进程
# 因此处理后的掩码长度也许和原点云坐标数量不一，需要按照掩码的长度调整
coords_hf_eig = coords_hf[:line_feat_mat.shape[0]]
colors_hf_eig = colors_hf[:line_feat_mat.shape[0]]

line_feat_mat = knn_fill_features(coords_hf_eig, line_feat_mat, num_neighbour)

line_feat_max = np.max(line_feat_mat, axis=0)
line_feat_min = np.min(line_feat_mat, axis=0)
line_feat_rgb = ((line_feat_mat / line_feat_max) * 255.0).astype(np.int32)
print(line_feat_rgb.mean(axis=0))
utils.npy2ply(coords_hf_eig, line_feat_rgb, "./output/pipeline_height_filtered_eigval_vis.ply")

# %%
lin = line_feat_mat[:, 0]
pla = line_feat_mat[:, 1]
sca = line_feat_mat[:, 2]
valid = (num_neighbour >= 3)  # 至少保证协方差稳定，按您密度可调到 10-5
line_mask = (
	valid &
	(pla < 0.20) &
	(sca < 0.20)
)
# 获取电力线点云
coords_line = coords_hf_eig[line_mask]
colors_line = colors_hf_eig[line_mask]
utils.npy2ply(coords_line, line_feat_rgb[line_mask], "./output/pipeline_line.ply")


# %% [markdown]
# # tower extraction

# %%
line_excluded_mask = ~line_mask & valid
coords_line_excluded = coords_hf_eig[line_excluded_mask]
colors_line_excluded = colors_hf_eig[line_excluded_mask]
utils.npy2ply(coords_line_excluded, colors_line_excluded, "./output/pipeline_line_excluded.ply")

# %%
num_workers = 4
batch_size = coords_line_excluded.shape[0] // num_workers
tower_feat_mat_list = [None] * num_workers
from concurrent.futures import ProcessPoolExecutor, as_completed
with ProcessPoolExecutor(max_workers=num_workers) as executor:
	futures_dict = {
		executor.submit(
			utils.eigval_vertic_fast,
			coords_line_excluded[i * batch_size:(i + 1) * batch_size],
			7.5
		) : i for i in range(num_workers)
	}

	for future in as_completed(futures_dict):
		feat_mat, _ = future.result()
		tower_feat_mat_list[futures_dict[future]] = feat_mat
tower_feat_mat = np.concatenate(tower_feat_mat_list, axis=0)

# %%
coords_line_excluded = coords_line_excluded[:tower_feat_mat.shape[0]]
colors_line_excluded = colors_line_excluded[:tower_feat_mat.shape[0]]

tower_feat_max = np.max(tower_feat_mat, axis=0)
tower_feat_min = np.min(tower_feat_mat, axis=0)
tower_feat_rgb = (tower_feat_mat  / tower_feat_max * 255.0).astype(np.int32)[:, :3]
utils.npy2ply(coords_line_excluded, tower_feat_rgb, "./output/pipeline_line_excluded_vis.ply")

# %%
eps = 1e-12
tower_feat_mat = (tower_feat_mat - tower_feat_min) / (tower_feat_max - tower_feat_min + eps)
tower_feat_mat = np.clip(tower_feat_mat, 0.0, 1.0)

v = tower_feat_mat[:, 0]
l = tower_feat_mat[:, 1]
p = tower_feat_mat[:, 2]
h = tower_feat_mat[:, 3]

v_thr = np.quantile(v, 0.70)
l_thr = np.quantile(l, 0.60)
p_thr = np.quantile(p, 0.15)
h_thr = np.quantile(h, 0.00)
tower_mask = (v > v_thr) & (l > l_thr) & (p < p_thr) & (h > h_thr)

coords_tower = coords_line_excluded[tower_mask]
colors_tower = colors_line_excluded[tower_mask]
tower_feat_rgb = tower_feat_rgb[tower_mask]
utils.npy2ply(coords_tower, colors_tower, "./output/pipeline_tower.ply")
utils.npy2ply(coords_tower, tower_feat_rgb, "./output/pipeline_tower_vis.ply")

# %%
labels = np.array(utils.npy2o3d(coords_tower).cluster_dbscan(
	eps=10.0,
	min_points=75,
	print_progress=False
))

valid = labels >= 0
labels_v = labels[valid]
pts_v = coords_tower[valid]
tower_instanced_labels = pts_v
print(f"number of clustered towers: {labels.max() + 1}")

import numpy as np

def labels_to_rgb(labels: np.ndarray,
                  noise_color=(0, 0, 0),
                  seed: int = 0) -> np.ndarray:
    """
    labels: (N,) int array, DBSCAN labels, noise = -1
    return: (N, 3) uint8 RGB
    """
    labels = np.asarray(labels)
    n = labels.shape[0]
    rgb = np.empty((n, 3), dtype=np.uint8)
    rgb[:] = np.array(noise_color, dtype=np.uint8)

    uniq = np.unique(labels)
    uniq = uniq[uniq >= 0]  # ignore noise

    rng = np.random.default_rng(seed)

    # 给每个簇生成颜色
    # 用 label->color 映射，保证同一 label 同一颜色
    color_map = {}
    for lab in uniq:
        # 避免过暗颜色：限制在 [40, 255]
        c = rng.integers(40, 256, size=3, dtype=np.uint8)
        color_map[int(lab)] = c

    for lab, c in color_map.items():
        rgb[labels == lab] = c

    return rgb


rgb = labels_to_rgb(labels, noise_color=(0, 0, 0), seed=42)
utils.npy2ply(coords_tower, rgb, "./output/tower_clusters_vis.ply")

# %%
uniq, cnt = np.unique(labels_v, return_counts=True)

keep = []
size_thr = np.quantile(cnt, 0.01)  # 也可以直接写死如 500

for lab, c in zip(uniq, cnt):
	if c < size_thr:
		continue
	# print(lab)
	P = pts_v[labels_v == lab]
	x_range = P[:, 0].max() - P[:, 0].min()
	y_range = P[:, 1].max() - P[:, 1].min()
	z_range = P[:, 2].max() - P[:, 2].min()
	xy_range = max(x_range, y_range)

	# 按您数据尺度调：杆塔一般 z_range 明显大，且相对细长
	if z_range < 2.0:
		continue
	# if xy_range > 30.0:
	# 	continue
	keep.append(lab)
tower_denoised_mask = valid & np.isin(labels, np.array(keep))
print(f"denoised ratio: {tower_denoised_mask.astype(np.int32).sum() / len(tower_denoised_mask) * 100.0:.2f}%")
coords_tower_denoised = coords_tower[tower_denoised_mask]
colors_tower_denoised = colors_tower[tower_denoised_mask]
visual_tower_denoised = tower_feat_rgb[tower_denoised_mask]

utils.npy2ply(coords_tower_denoised, visual_tower_denoised, "./output/pipeline_tower_denoised_vis.ply")

tower_labels = uniq
coords_tower = coords_tower_denoised
tower_instanced_labels = labels[tower_denoised_mask]

# %% [markdown]
# # final composition

# %%
def position_instanced_radius(points: np.ndarray, poses: np.ndarray, radius: float):
    import open3d as o3d
    
    o3d_pcd = utils.npy2o3d(points)
    search_tree = o3d.geometry.KDTreeFlann(o3d_pcd)

    instanced_labels = np.zeros((len(points), ), dtype=bool)

    for query in poses:
        neighbour_num, neighbour_indicies, _ = search_tree.search_radius_vector_3d(query, radius)
        instanced_labels[neighbour_indicies] = True
    
    return instanced_labels

def position_instanced_vertical(points: np.ndarray, poses: np.ndarray, border: float):
    instanced_labels = np.zeros((len(points), ), dtype=np.int32)

    instanced_labels = np.zeros((len(points), ), dtype=bool)
    for query in poses:
        square_selection = (np.abs(points[:, 0] - query[0]) < border) & (np.abs(points[:, 1] - query[1]) < border)
        instanced_labels = instanced_labels | square_selection
    
    return square_selection



labels_line = position_instanced_radius(coords, coords_line, 2.0)
labels_tower = np.array([False] * len(coords))

for tower_idx in tower_labels:
    labels_tower = labels_tower | position_instanced_vertical(
        coords,
        np.array([coords_tower[tower_instanced_labels == tower_idx].mean(axis=0)]),
        10
    )

print(f"number of label in line:\t{labels_line.astype(np.int32).sum()}")
print(f"number of label in tower:\t{labels_tower.astype(np.int32).sum()}")


# give different colors to line and tower
colors_labeled = colors
colors_labeled[labels_line] = np.array([100, 255, 255])
colors_labeled[labels_tower] = np.array([255, 100, 100])

utils.npy2ply(coords, colors_labeled, "./output/pipeline_final.ply")



