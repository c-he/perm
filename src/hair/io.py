import os
import struct

import numpy as np


def load_hair(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1]
    assert ext == '.data', "only support loading hair data in .data format"

    with open(path, 'rb') as f:
        data = f.read()
    num_strands = struct.unpack('i', data[:4])[0]
    strands = []
    idx = 4
    for _ in range(num_strands):
        num_points = struct.unpack('i', data[idx:idx + 4])[0]
        points = struct.unpack('f' * num_points * 3, data[idx + 4:idx + 4 + 4 * num_points * 3])
        strands.append(list(points))
        idx = idx + 4 + 4 * num_points * 3
    strands = np.array(strands).reshape((num_strands, -1, 3))

    return strands


def save_hair(path: str, data: np.ndarray) -> None:
    ext = os.path.splitext(path)[1]
    assert ext in ['.data', '.obj'], "only support saving hair data in .data and .obj format"

    if ext == '.data':
        _save_hair_data(path, data)
    else:
        _save_hair_obj(path, data)


def _save_hair_data(path: str, data: np.ndarray) -> None:
    num_strands, num_points = data.shape[:2]
    with open(path, 'wb') as f:
        f.write(struct.pack('i', num_strands))
        for i in range(num_strands):
            f.write(struct.pack('i', num_points))
            f.write(struct.pack('f' * num_points * 3, *data[i].flatten().tolist()))


def _save_hair_obj(path: str, data: np.ndarray) -> None:
    verts = np.reshape(data, [-1, 3])
    edges = np.reshape(np.arange(data.shape[0] * data.shape[1]), [data.shape[0], data.shape[1]])
    edges = edges + 1
    lines = [None] * (verts.shape[0] + edges.shape[0] * (edges.shape[1] - 1))

    li = 0
    for v in verts:
        lines[li] = 'v ' + ' '.join(map(str, v)) + '\n'
        li += 1

    for e in edges:
        for i in range(len(e) - 1):
            lines[li] = 'l ' + ' '.join(map(str, e[i:i + 2])) + '\n'
            li += 1

    with open(path, 'w') as f:
        f.writelines(lines)
