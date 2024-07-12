import numpy as np
import open3d as o3d

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

def variance(seq: list):
    n = len(seq)
    assert n > 0, "sequence length must be positive"
    mean = sum(seq) / n
    var = sum((x - mean)**2 for x in seq) / n
    return var

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

def eigval_radius(points:np.ndarray, be_range: tuple, radius: float):
    '''
    compute eigen values of all points given the neighbourhood
    radius.

    Params:
    -
    * points (np.ndarray) - [n, 3] np.ndarray coordinates.
    * be_range (tuple[int]) - (2) [start, end)
    * radius (float) - radius for computing  eigen  values  of
        each point.

    Returns:
    -
    * np.ndarray - [n, 3] eigen values of all points
    '''
    from tqdm import tqdm
    
    assert len(points) > 0
    assert len(be_range) == 2
    assert be_range[0] < be_range[1]
    
    # construct kd-tree to search
    pcd = npy2o3d(points)
    search_tree = o3d.geometry.KDTreeFlann(pcd)

    eigval_list = np.zeros((0, 3), dtype=np.float32)
    neighbour_num_record = []
    for query_idx in tqdm(range(be_range[0], be_range[1]), desc="eigval progress", total=be_range[1] - be_range[0], ncols=100):
        query = pcd.points[query_idx]
        neighbour_num, neighbour_indicies, _ = search_tree.search_radius_vector_3d(query, radius)
        neighbour_num_record.append(neighbour_num - 1)
        if neighbour_num < 3:
            eigval_list = np.concatenate((eigval_list, np.array([0.0, 0.0, 0.0])[np.newaxis, :]), axis=0)
            continue
        eigvals, eigvecs = pca_k(points[neighbour_indicies], 3)
        assert eigvals[0] >= eigvals[1] and eigvals[1] >= eigvals[2]
        eigval_list = np.concatenate((eigval_list, np.array([
            (eigvals[0] - eigvals[1]) / (eigvals[0] + 1e-9),
            (eigvals[1] - eigvals[2]) / (eigvals[1] + 1e-9),
            0
        ]).reshape(-1, 3)), axis=0)
    
    return eigval_list, neighbour_num_record

def eigval_vertic(points: np.ndarray, be_range: tuple, border: float):
    '''
    compute eigen values of all points given the neighbourhood
    radius.

    Params:
    -
    * points (np.ndarray) - [n, 3] np.ndarray coordinates.
    * be_range (tuple[int]) - (2) [start, end)
    * radius (float) - radius for computing  eigen  values  of
        each point.

    Returns:
    -
    * np.ndarray - [n, 3] eigen values of all points
    '''

    from tqdm import tqdm

    feat_matrix = np.zeros((0, 3), dtype=np.float32)
    neighbour_num_record = []

    global_max_height = np.max(points[:, 2])
    for query_idx in tqdm(range(be_range[0], be_range[1]), desc="eigval progress", total=be_range[1]-be_range[0], ncols=100):
        query = points[query_idx]
        square_neighbours_mask = (np.abs(points[:, 0] - query[0]) < border) & (np.abs(points[:, 1] - query[1]) < border)
        square_neighbours = points[square_neighbours_mask]
        square_neighbours_num = len(square_neighbours) - 1
        neighbour_num_record.append(len(square_neighbours))
        if square_neighbours_num < 3:
            feat_matrix = np.concatenate((feat_matrix, np.array([0.0, 0.0, 0.0])[np.newaxis, :]), axis=0)
            continue

        local_max_height = np.max(square_neighbours[:, 2])
        _, eigvecs = pca_k(square_neighbours, 3)
        feat = np.array([
            my_cos(eigvecs[:, 0], np.array([0, 0, 1])), # angle feat between eig1 and (0,0,1)
            my_cos(eigvecs[:, 1], np.array([0, 0, 1])), # angle feat between eig2 and (0,0,1)
            my_cos(eigvecs[:, 2], np.array([0, 0, 1]))  # angle feat between eig3 and (0,0,1)
        ]).reshape(-1, 3)
        feat_matrix = np.concatenate((feat_matrix, feat), axis=0)
    
    return feat_matrix, neighbour_num_record

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
