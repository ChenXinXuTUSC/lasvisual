import numpy as np
import open3d as o3d

from tqdm import tqdm

def voxel_downsample(points: np.ndarray, voxel_size: float, use_avg: bool):
	'''
	Conduct a downsample with the given voxel size

	Params:
	-
	* points (np.ndarray) - original 3d coordinates array
	* voxel_size (float) - voxel size in meter
	* use_avg (bool) - whether to average points in the same voxel

	Returns:
	-
	* np.ndarray[n,3] - downsampled points
	* np.ndarray[1,n] - indices of element in the new array to form original array
	* np.ndarray[1,n] - indices of element in the original array to form new array
	* np.ndarray[1,n] - number of repeated times of element in the orignal array
	'''
	# in case that points contain colors
	points = points[:, :3]
	min_coord = np.min(points, axis=0)
	max_coord = np.max(points, axis=0)

	# calculate voxel number of each axis
	voxel_numaxis = (max_coord - min_coord) // voxel_size
	voxel_indices = (points - min_coord) // voxel_size
	voxel_centers = voxel_indices * voxel_size + voxel_size / 2.0

	# group the points by voxel indices
	# return_index:bool
	#   given a list of subscript which can be used to choose elements
	#   from the original array to form the unique array
	# return_inverse:bool
	#   given a list of subscript which can be used to choose elements
	#   from the unique array to form the original array
	# return_counts:bool
	#   return the number of repeated times in the original array.
	voxel_unique, idx_o2n, idx_n2o, unique_counts = np.unique(
		voxel_indices, axis=0,
		return_index=True,
		return_inverse=True,
		return_counts=True
	)

	voxel_points = np.zeros((len(voxel_unique), 3))

	if use_avg:
		np.add.at(voxel_points, idx_n2o, points)
		voxel_points /= unique_counts.reshape(-1, 1)
	else:
		voxel_points = voxel_centers[idx_o2n]
	
	return voxel_points, idx_o2n, idx_n2o, unique_counts

def get_average_pivot(data):
	avg_curr = np.mean(data)
	avg_next = (np.mean(data[data < avg_curr]) + np.mean(data[data > avg_curr])) / 2.0
	while True:
		if abs(avg_curr - avg_next) < 1e-2:
			break
		avg_curr = avg_next
		avg_next = (np.mean(data[data < avg_curr]) + np.mean(data[data > avg_curr])) / 2.0
	return avg_curr

def height_hist_filter(data, bin_size=0.5, elevation_idx=2):
	"""
	基于高程直方图寻找最佳分割点
	支持输入纯高程数组或三维坐标数组
	
	参数：
	data: 可以是高程数组（一维）或坐标数组（二维，n×3）
	bin_size: 直方图箱子大小（米）
	elevation_idx: 如果是坐标数组，高程所在的轴索引
	
	返回：高程阈值
	"""
	if len(data) == 0:
		return 0
	
	# 判断输入数据类型
	if data.ndim == 1:
		# 已经是高程数据
		elevation = data
	elif data.ndim == 2 and data.shape[1] >= 3:
		# 是坐标数据，提取高程
		elevation = data[:, elevation_idx]
	else:
		raise ValueError(f"输入数据维度不支持: {data.shape}，期望一维数组或二维数组(n×3)")
	
	# 计算高程范围
	min_h, max_h = np.min(elevation), np.max(elevation)
	
	# 创建直方图（确保至少有一个箱子）
	if max_h - min_h < bin_size:
		# 如果高程范围小于箱子大小，调整箱子大小
		bin_size = (max_h - min_h) / 10.0 if max_h > min_h else 0.1
	
	bins = np.arange(min_h, max_h + bin_size, bin_size)
	hist, bin_edges = np.histogram(elevation, bins=bins)
	
	# 如果直方图全为零（理论上不会发生）
	if np.sum(hist) == 0:
		return np.mean(elevation)
	
	# 寻找地面峰值（通常是最大的峰值）
	ground_peak_idx = np.argmax(hist)
	ground_height = (bin_edges[ground_peak_idx] + bin_edges[ground_peak_idx + 1]) / 2
	
	# 寻找第一个明显的谷底（地面和地物的分界）
	# 从地面峰值向高处搜索
	for i in range(ground_peak_idx + 1, len(hist) - 1):
		if hist[i] < hist[i-1] and hist[i] < hist[i+1] and hist[i] < 0.1 * hist[ground_peak_idx]:
			# 找到谷底，返回对应高程
			return (bin_edges[i] + bin_edges[i + 1]) / 2
	
	# 如果没有找到明显谷底，使用自适应阈值
	# 基于高程的标准差设置阈值
	elevation_std = np.std(elevation)
	return ground_height + max(2.0, elevation_std * 0.5)

def eigval_radius(points: np.ndarray, radius: float):
	n = len(points)
	assert n > 0

	pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
	search_tree = o3d.geometry.KDTreeFlann(pcd)

	eigval_list = np.zeros((n, 3), dtype=np.float32)
	neighbour_num_record = np.zeros((n,), dtype=np.int32)

	for i, query in enumerate(tqdm(points, desc='eigval progress', total=n, ncols=100)):
		k, idx, _ = search_tree.search_radius_vector_3d(query, radius)
		neighbour_num_record[i] = k - 1
		if k < 3:
			continue

		X = points[np.asarray(idx)]
		mu = X.mean(axis=0, keepdims=True)
		Y = X - mu
		eps = 1e-12
		scale = np.linalg.norm(Y, axis=1).max()
		if scale > eps:
			Y = Y / scale   # 归一化到单位球体
		C = (Y.T @ Y) / max((k - 1), 1)
		w = np.linalg.eigvalsh(C)[::-1] + 1e-12
		l1, l2, l3 = w

		assert w[0] >= w[1] and w[1] >= w[2]
		eps = 1e-6
		eigval_list[i, 0] = (l1 - l2) / l1
		eigval_list[i, 1] = (l2 - l3) / l1
		eigval_list[i, 2] = l3 / l1

	return eigval_list, neighbour_num_record

def my_sin(vec1: np.array, vec2: np.array):
	cross_product = np.cross(vec1, vec2)
	norm_cross_product = np.linalg.norm(cross_product)
	norm_vec1 = np.linalg.norm(vec1)
	norm_vec2 = np.linalg.norm(vec2)

	sin_val = norm_cross_product / (norm_vec1 * norm_vec2)

	return sin_val

def my_cos(vec1: np.array, vec2: np.array):
	dot_product = np.dot(vec1, vec2)
	norm_dot_product = np.linalg.norm(dot_product)
	norm_vec1 = np.linalg.norm(vec1)
	norm_vec2 = np.linalg.norm(vec2)

	cos_val = norm_dot_product / (norm_vec1 * norm_vec2)

	return cos_val

def eigval_vertic_fast(points: np.ndarray, border: float, k_min: int = 10):
	# 需要 scipy
	from scipy.spatial import cKDTree

	n = points.shape[0]
	assert n > 0

	global_max_h = float(np.max(points[:, 2]) + 1e-12)

	feat = np.zeros((n, 4), dtype=np.float32)
	neigh_cnt = np.zeros((n,), dtype=np.int32)

	# 只在 xy 上建树，实现正方形邻域：L∞ 半径 border
	tree = cKDTree(points[:, :2])

	ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)

	for i in tqdm(range(n), desc='eigval progress', total=n, ncols=100):
		idx = tree.query_ball_point(points[i, :2], r=border, p=np.inf)
		k = len(idx)
		neigh_cnt[i] = max(k - 1, 0)

		if k < k_min:
			continue

		X = points[np.asarray(idx, dtype=np.int64)]
		mu = X.mean(axis=0, keepdims=True)
		Y = X - mu

		# 可选：尺度归一化，减少不同密度尺度影响
		scale = np.linalg.norm(Y, axis=1).max()
		if scale > 1e-12:
			Y = Y / scale

		C = (Y.T @ Y) / max(k - 1, 1)

		# eigh 返回升序
		w, V = np.linalg.eigh(C)
		order = np.argsort(w)[::-1]
		w = w[order] + 1e-12
		V = V[:, order]

		l1, l2, l3 = w
		v1 = V[:, 0]  # 第一主轴

		verticality = float(abs(v1 @ ez))             # 0 到 1
		linearity = float((l1 - l2) / l1)             # 0 到 1
		planarity = float((l2 - l3) / l1)        
		height_norm = float(points[i, 2] / global_max_h) # 平面结构高

		feat[i, 0] = verticality
		feat[i, 1] = linearity
		feat[i, 2] = planarity
		feat[i, 3] = height_norm

	return feat, neigh_cnt

def pca_k(data: np.ndarray, k: int):
	'''
	compute the principle k components of the given data point.

	Params:
	-
	* data (np.ndarray[n, 3]) - xyz points
	* k (int) - number of principle components
	'''

	# centralized
	data = data - data.mean(axis=0)
	# cova = np.matmul(data.T, data) / data.shape[0]
	cova = np.cov(data, rowvar=False)

	eigvals, eigvecs = np.linalg.eig(cova)

	sorted_indices = np.argsort(eigvals)[::-1]
	eigvals = eigvals[sorted_indices]
	eigvecs = eigvecs[:, sorted_indices]
	
	return eigvals[:k], eigvecs[:, :k]

def cluster_instanced(points: np.ndarray, border_len: float, min_threshold: int):
	'''
	This function clusters the seperated points mainly based on vertical feature,
	with a pre-filtration test to avoid false positive points retained.

	Params:
	-
	* points (np.ndarray) - [n, 3] coordinates
	* border_len (float) - the length of the squre border
	* min_threshold (int) - minimum number of the selected points to be recognized as a cluster

	Returns:
	-
	* instanced_labels (np.array) - instantiation labels in [n, ] np.array
	'''
	instanced_labels = np.zeros((len(points), ), dtype=np.int32)
	instanced_counts = 1
	for idx, query in enumerate(points):
		if instanced_labels[idx] > 0:
			# skip the labeled point
			continue

		pre_square_selection_mask = \
			(abs(points[:, 0] - query[0]) < border_len / 15.0) & \
			(abs(points[:, 1] - query[1]) < border_len / 15.0)
		if pre_square_selection_mask.astype(np.int32).sum() < (min_threshold // 15):
			continue

		square_selection_mask = \
			(abs(points[:, 0] - query[0]) < border_len) & \
			(abs(points[:, 1] - query[1]) < border_len)
		if square_selection_mask.astype(np.int32).sum() >= min_threshold and instanced_labels[square_selection_mask].sum() == 0:
			instanced_labels[square_selection_mask] = instanced_counts
			instanced_counts += 1
	return instanced_labels

# some commonly used conversions
def npy2o3d(data: np.ndarray):
	'''
	convert numpy xyz coordinates to o3d points

	Params:
	-
	* data (np.ndarray) - original numpy coordinates

	Returns:
	-
	* open3d.geometry.PointCloud - converted coordinates
	'''
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(data)
	return pcd
