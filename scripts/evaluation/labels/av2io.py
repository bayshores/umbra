"""Minimal read-only Argoverse 2 access for the labeling kit (development logs only).

Pixel convention everywhere in this kit: OpenCV / AV2 pinhole convention on the raw full-resolution
stereo_front_left image (2048 x 1550). Pixel (column i, row j) has its centre at (i, j); the boundary
between columns i-1 and i is at u = i - 0.5. The raw AV2 stereo JPEGs are already undistorted
(straight poles and lane lines at the image edges), so projection is plain pinhole, the same model the
app uses (prepare_av2.py passes zero distortion).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from common import ROOT, assert_development, full_log_id, require

CAMERA = 'stereo_front_left'
VEHICLE_WORDS = ('VEHICLE', 'TRUCK', 'BUS', 'TRAILER')


def quaternion_matrix(w, x, y, z):
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def se3(row):
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_matrix(row['qw'], row['qx'], row['qy'], row['qz'])
    matrix[:3, 3] = [row['tx_m'], row['ty_m'], row['tz_m']]
    return matrix


def apply(matrix, points):
    points = np.asarray(points, float)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def read_rows(path, columns=None):
    assert_development(path)
    table = feather.read_table(str(path), columns=columns)
    return table.to_pylist()


class Log:
    """Poses, calibration and annotations of one development log.

    `roots` are directories searched in order for each relative file (the shared data folder first,
    then kit download folders). Nothing outside development logs is accepted.
    """

    def __init__(self, log, roots):
        self.log = full_log_id(log)
        self.roots = [Path(r) for r in roots]
        for root in self.roots:
            assert_development(root)
        pose_rows = read_rows(self.find('city_SE3_egovehicle.feather'))
        self.poses = {int(r['timestamp_ns']): se3(r) for r in pose_rows}
        self.pose_times = np.array(sorted(self.poses))
        intr = {r['sensor_name']: r for r in read_rows(self.find('calibration/intrinsics.feather'))}[CAMERA]
        self.width, self.height = int(intr['width_px']), int(intr['height_px'])
        self.K = np.array([[intr['fx_px'], 0, intr['cx_px']], [0, intr['fy_px'], intr['cy_px']], [0, 0, 1.0]])
        extr = {r['sensor_name']: r for r in read_rows(self.find('calibration/egovehicle_SE3_sensor.feather'))}
        self.ego_from_camera = se3(extr[CAMERA])
        self._annotations = None

    def find(self, relative_path):
        for root in self.roots:
            candidate = root / relative_path
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f'{self.log}: {relative_path} not found in {[str(r) for r in self.roots]}')

    def pose(self, timestamp):
        timestamp = int(timestamp)
        require(timestamp in self.poses, f'No exact ego pose for {timestamp}')
        return self.poses[timestamp]

    def city_from_camera(self, timestamp):
        return self.pose(timestamp) @ self.ego_from_camera

    def camera_from_city(self, timestamp):
        return np.linalg.inv(self.city_from_camera(timestamp))

    def project(self, city_points, timestamp):
        """City points -> (u, v, depth) in the raw camera image at `timestamp` (pinhole, OpenCV convention)."""
        cam = apply(self.camera_from_city(timestamp), city_points)
        z = cam[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            u = self.K[0, 0] * cam[:, 0] / z + self.K[0, 2]
            v = self.K[1, 1] * cam[:, 1] / z + self.K[1, 2]
        return u, v, z

    def camera_center_city(self, timestamp):
        return self.city_from_camera(timestamp)[:3, 3]

    def sweep_city(self, timestamp):
        """One LiDAR sweep in city coordinates. AV2 sweeps are in the ego frame at the sweep timestamp."""
        rows = feather.read_table(str(self.find(f'sensors/lidar/{int(timestamp)}.feather')),
                                  columns=['x', 'y', 'z', 'intensity', 'laser_number']).to_pydict()
        ego = np.column_stack([np.asarray(rows[k], float) for k in ('x', 'y', 'z')])
        return {'xyz': apply(self.pose(timestamp), ego), 'intensity': np.asarray(rows['intensity'], float),
                'laser': np.asarray(rows['laser_number'], int), 'egoOrigin': self.pose(timestamp)[:3, 3]}

    @property
    def annotations(self):
        if self._annotations is None:
            rows = read_rows(self.find('annotations.feather'))
            for r in rows:
                box_pose = se3(r)
                city = self.pose(r['timestamp_ns']) @ box_pose
                r['city_center'] = city[:3, 3]
                r['city_rotation'] = city[:3, :3]
            self._annotations = rows
        return self._annotations

    def stationary_vehicles(self, max_spread_m=0.3, min_count=10):
        """Vehicle tracks whose annotated centres move less than `max_spread_m` (RMS) in the city frame.

        Used only to choose candidate cases and LiDAR cluster seeds, never as a reference shape.
        """
        tracks = {}
        for r in self.annotations:
            if any(word in r['category'] for word in VEHICLE_WORDS):
                tracks.setdefault(r['track_uuid'], []).append(r)
        out = []
        for track, rows in tracks.items():
            centers = np.array([r['city_center'] for r in rows])
            spread = float(np.linalg.norm(centers[:, :2].std(0)))
            if len(rows) >= min_count and spread <= max_spread_m:
                out.append({'track': track, 'category': rows[0]['category'], 'count': len(rows), 'spreadM': spread,
                            'meanCenter': centers.mean(0), 'timestamps': sorted(int(r['timestamp_ns']) for r in rows),
                            'rows': rows})
        return out


def default_roots(log):
    """Search order for a development log: shared data folder, then kit downloads for that log."""
    log = full_log_id(log)
    roots = [ROOT / '.local/data/av2/train' / log]
    downloads = ROOT / '.local/evaluation/pao-labels/downloads'
    if downloads.exists():
        roots += sorted(p for p in downloads.iterdir() if p.is_dir() and p.name.startswith(log[:8]))
    return roots
