import numpy as np
import laspy
import plyfile

import os

def las2npy(las_file_path):
    las_file = laspy.read(las_file_path)
    # print("test info:")
    # print(type(las_file.X))
    # print(las_file.header.offsets)
    # print(las_file.header.scales)

    points = np.vstack((
        las_file.X * las_file.header.scales[0] + las_file.header.offsets[0],
        las_file.Y * las_file.header.scales[1] + las_file.header.offsets[1],
        las_file.Z * las_file.header.scales[2] + las_file.header.offsets[2]
    )).transpose()

    colors = None
    has_color = hasattr(las_file, "red") and hasattr(las_file, "green") and hasattr(las_file, "blue")
    if has_color:
        colors = np.vstack((
            las_file.red // 256,
            las_file.green // 256,
            las_file.blue // 256
        )).transpose()

    return points, colors

def npy2ply(points, colors, file_path, use_txt=False):
    if os.path.isdir(file_path):
        raise Exception("file path is a directory")
    vertex_data = None
    vertex_dtyp = None

    if colors is not None and len(points) == len(colors):
        vertex_data = [tuple(pt) + tuple(cl) for pt, cl in zip(points, colors)]
        vertex_dtyp = [
            ("x", "f8"),
            ("y", "f8"),
            ("z", "f8"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1")
        ]
    else:
        # remember the comma when creating a tuple that contains
        # only one element, or interpreter won't explain this to
        # tuple
        vertex_data = [tuple(pt) for pt in points]
        vertex_dtyp = [
            ("x", "f8"),
            ("y", "f8"),
            ("z", "f8")
        ]
    
    vertex_element = plyfile.PlyElement.describe(np.array(vertex_data, vertex_dtyp), "vertex")
    plyfile.PlyData([vertex_element], text=use_txt).write(file_path)
