"""Build point/color/adjacency buffers; validity comes from the caller."""
import numpy as np


def build_surface(camera_points, valid, rgb, origin, rotation, mesh_edge_limit):
    points = (camera_points[valid] - origin) @ rotation.T
    if not np.isfinite(points.astype('<f4')).all():
        raise ValueError('Measured points exceed finite float32 representation')
    indices = np.full(valid.shape, -1, dtype=np.int64)
    indices[valid] = np.arange(valid.sum())
    triangles = []
    for triplet in [(indices[:-1, :-1], indices[1:, :-1], indices[:-1, 1:]),
                    (indices[1:, 1:], indices[:-1, 1:], indices[1:, :-1])]:
        candidate = np.stack(triplet, axis=-1).reshape(-1, 3)
        candidate = candidate[(candidate >= 0).all(axis=1)]
        positions = points[candidate]
        edges = np.stack((positions[:, 0]-positions[:, 1], positions[:, 1]-positions[:, 2], positions[:, 2]-positions[:, 0]))
        triangles.append(candidate[(np.linalg.norm(edges, axis=-1) <= mesh_edge_limit).all(axis=0)])
    triangles = np.concatenate(triangles).astype('<u4')
    if not len(triangles):
        raise ValueError('No supported measured triangles survived the discontinuity gate')
    srgb = rgb[valid].astype(float) / 255
    linear = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)
    return points, linear, triangles
