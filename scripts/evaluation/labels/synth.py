"""Synthetic known-answer cases: car-like solids and crosswalk paint rendered into real DEVELOPMENT photos.

Geometry uses the real camera (intrinsics, pose) of the background frame and a ground plane fitted to
that frame's LiDAR. Every object is a union of convex solids, so its silhouette is the union of the
convex hulls of its projected vertices: the ground truth (extremes, top/bottom profile at any column,
paint near-edge row at any column) is exact. Images are rendered at `supersample` x resolution with a
z-buffer, box-filtered down, blurred, noised and JPEG-encoded like a camera frame.
Synthetic images are illustrative test material only, never evidence about a real scene.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull

from av2io import apply


# ----- solids (vehicle-local: x forward, y left, z up; z = 0 on the ground) -------------------------

def prism(plan_xy, z0, z1, top_plan_xy=None):
    """Convex prism / frustum from a convex CCW plan polygon at z0 (and optionally a different top at z1)."""
    bottom = np.column_stack([plan_xy, np.full(len(plan_xy), z0)])
    top_xy = plan_xy if top_plan_xy is None else top_plan_xy
    top = np.column_stack([top_xy, np.full(len(top_xy), z1)])
    vertices = np.vstack([bottom, top])
    n = len(plan_xy)
    faces = [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    faces.append(list(range(n))[::-1])          # bottom (outward -z)
    faces.append(list(range(n, 2 * n)))         # top (outward +z)
    return vertices, faces


def rect(x0, x1, y0, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def chamfered(x0, x1, y0, y1, c):
    return np.array([[x0 + c, y0], [x1 - c, y0], [x1, y0 + c], [x1, y1 - c], [x1 - c, y1], [x0 + c, y1], [x0, y1 - c], [x0, y0 + c]], float)


def cylinder_y(cx, cy, cz, radius, width, segments=20):
    """Wheel: cylinder with axis along y."""
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.column_stack([cx + radius * np.cos(angles), cz + radius * np.sin(angles)])
    left = np.column_stack([ring[:, 0], np.full(segments, cy + width / 2), ring[:, 1]])
    right = np.column_stack([ring[:, 0], np.full(segments, cy - width / 2), ring[:, 1]])
    vertices = np.vstack([right, left])
    faces = [[i, (i + 1) % segments, segments + (i + 1) % segments, segments + i] for i in range(segments)]
    faces.append(list(range(segments)))
    faces.append(list(range(segments, 2 * segments))[::-1])
    return vertices, faces


def vehicle(kind, length, width, height):
    """Convex parts of a simple vehicle. Returns [(name, vertices, faces, material)]."""
    L, W, H = length, width, height
    parts = []
    if kind == 'box':
        v, f = prism(rect(-L / 2, L / 2, -W / 2, W / 2), 0.0, H)
        return [('body', v, f, 'body')]
    sill = 0.22 if kind == 'sedan' else 0.30
    belt = 0.82 if kind == 'sedan' else 1.02
    v, f = prism(chamfered(-L / 2, L / 2, -W / 2, W / 2, 0.18), sill, belt)
    parts.append(('body', v, f, 'body'))
    if kind == 'sedan':
        bottom = rect(-L / 2 + 0.95, L / 2 - 1.25, -W / 2 + 0.10, W / 2 - 0.10)
        top = rect(-L / 2 + 1.45, L / 2 - 1.95, -W / 2 + 0.28, W / 2 - 0.28)
    else:
        bottom = rect(-L / 2 + 0.25, L / 2 - 1.05, -W / 2 + 0.08, W / 2 - 0.08)
        top = rect(-L / 2 + 0.35, L / 2 - 1.55, -W / 2 + 0.18, W / 2 - 0.18)
    v, f = prism(bottom, belt, H, top)
    parts.append(('cabin', v, f, 'cabin'))
    radius = 0.33 if kind == 'sedan' else 0.38
    for sx in (-1, 1):
        for sy in (-1, 1):
            v, f = cylinder_y(sx * (L / 2 - 0.85), sy * (W / 2 - 0.17), radius, radius, 0.23)
            parts.append((f'wheel{sx:+d}{sy:+d}', v, f, 'tire'))
    for sy in (-1, 1):
        mx = L / 2 - (1.30 if kind == 'sedan' else 1.10)
        y_lo, y_hi = (-W / 2 - 0.11, -W / 2 + 0.14) if sy < 0 else (W / 2 - 0.14, W / 2 + 0.11)
        v, f = prism(rect(mx - 0.10, mx + 0.06, y_lo, y_hi), belt + 0.06, belt + 0.20)
        parts.append((f'mirror{sy:+d}', v, f, 'body'))
    return parts


# ----- placement -------------------------------------------------------------------------------------

def fit_plane(points, iterations=400, threshold=0.04, seed=0):
    """RANSAC plane z = a x + b y + c over points (city), refined by least squares on inliers."""
    rng = np.random.default_rng(seed)
    best, best_count = None, -1
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        a = np.c_[sample[:, :2], np.ones(3)]
        try:
            coef = np.linalg.solve(a, sample[:, 2])
        except np.linalg.LinAlgError:
            continue
        resid = np.abs(points[:, :2] @ coef[:2] + coef[2] - points[:, 2])
        count = int((resid < threshold).sum())
        if count > best_count:
            best, best_count = coef, count
    inliers = np.abs(points[:, :2] @ best[:2] + best[2] - points[:, 2]) < threshold
    coef, *_ = np.linalg.lstsq(np.c_[points[inliers, :2], np.ones(inliers.sum())], points[inliers, 2], rcond=None)
    return coef, inliers


def plane_frame(coef, point_xy, yaw):
    """Pose (4x4) on plane z = a x + b y + c at point_xy with heading yaw (city, radians)."""
    a, b, c = coef
    normal = np.array([-a, -b, 1.0])
    normal /= np.linalg.norm(normal)
    heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    x = heading - normal * (heading @ normal)
    x /= np.linalg.norm(x)
    y = np.cross(normal, x)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([x, y, normal])
    pose[:3, 3] = [point_xy[0], point_xy[1], a * point_xy[0] + b * point_xy[1] + c]
    return pose


# ----- rasterisation ---------------------------------------------------------------------------------

def hull_polygon(uv):
    hull = ConvexHull(uv)
    return uv[hull.vertices]


def inside_convex(polygon, xs, ys):
    """Boolean mask of samples inside a convex polygon (either orientation)."""
    n = len(polygon)
    signs = []
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        signs.append((x1 - x0) * (ys - y0) - (y1 - y0) * (xs - x0))
    signs = np.array(signs)
    return (signs >= 0).all(0) | (signs <= 0).all(0)


def column_interval(polygon, u):
    """[v_min, v_max] where the vertical line at u crosses a convex polygon, or None."""
    vs = []
    n = len(polygon)
    for i in range(n):
        (u0, v0), (u1, v1) = polygon[i], polygon[(i + 1) % n]
        if (u0 - u) * (u1 - u) <= 0 and u0 != u1:
            t = (u - u0) / (u1 - u0)
            vs.append(v0 + t * (v1 - v0))
        elif u0 == u1 == u:
            vs += [v0, v1]
    return (min(vs), max(vs)) if vs else None


class Camera:
    def __init__(self, K, camera_from_city, width, height):
        self.K, self.T, self.width, self.height = K, camera_from_city, width, height

    def to_camera(self, city):
        return apply(self.T, city)

    def project_camera(self, cam):
        return np.column_stack([self.K[0, 0] * cam[:, 0] / cam[:, 2] + self.K[0, 2], self.K[1, 1] * cam[:, 1] / cam[:, 2] + self.K[1, 2]])


def render_solids(background, camera, parts_city, materials, supersample=8, margin=6, light=(0.3, -0.5, 0.8), seed=0, exposure=1.0):
    """Composite convex solids into `background` (HxWx3 uint8). Returns (image uint8, coverage float, region, polygons)."""
    rng = np.random.default_rng(seed)
    polygons, cam_parts = [], []
    for name, vertices, faces, material in parts_city:
        cam = camera.to_camera(vertices)
        if (cam[:, 2] <= 0.5).any():
            raise ValueError('Solid behind or too close to the camera')
        uv = camera.project_camera(cam)
        polygons.append(hull_polygon(uv))
        cam_parts.append((name, cam, uv, faces, material))
    allv = np.vstack(polygons)
    x0 = int(np.floor(allv[:, 0].min())) - margin
    y0 = int(np.floor(allv[:, 1].min())) - margin
    x1 = int(np.ceil(allv[:, 0].max())) + margin
    y1 = int(np.ceil(allv[:, 1].max())) + margin
    w, h = x1 - x0 + 1, y1 - y0 + 1
    s = supersample
    # hi-res sample centres in image coordinates (pixel i spans [i-0.5, i+0.5])
    xs = x0 - 0.5 + (np.arange(w * s) + 0.5) / s
    ys = y0 - 0.5 + (np.arange(h * s) + 0.5) / s
    gx, gy = np.meshgrid(xs, ys)
    depth = np.full(gx.shape, np.inf)
    shade = np.zeros(gx.shape)
    kinv = np.linalg.inv(camera.K)
    rays = np.stack([gx, gy, np.ones_like(gx)], -1) @ kinv.T
    light_cam = camera.T[:3, :3] @ np.array(light) / np.linalg.norm(light)
    up_cam = camera.T[:3, :3] @ np.array([0, 0, 1.0])
    for name, cam, uv, faces, material in cam_parts:
        mat = materials[material]
        centroid = cam.mean(0)
        for face in faces:
            pts = cam[face]
            normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            if np.linalg.norm(normal) < 1e-12:
                continue
            normal /= np.linalg.norm(normal)
            if normal @ (pts.mean(0) - centroid) < 0:  # orient outward (parts are convex)
                normal = -normal
            if normal @ pts[0] >= 0:  # facing away from the camera at the origin
                continue
            poly = uv[face]
            fx0, fx1 = poly[:, 0].min(), poly[:, 0].max()
            fy0, fy1 = poly[:, 1].min(), poly[:, 1].max()
            cx0 = max(0, int((fx0 - (x0 - 0.5)) * s) - 1)
            cx1 = min(w * s, int((fx1 - (x0 - 0.5)) * s) + 2)
            cy0 = max(0, int((fy0 - (y0 - 0.5)) * s) - 1)
            cy1 = min(h * s, int((fy1 - (y0 - 0.5)) * s) + 2)
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            sub = (slice(cy0, cy1), slice(cx0, cx1))
            inside = inside_convex(poly, gx[sub], gy[sub])
            d = rays[sub]
            t = (normal @ pts[0]) / (d @ normal)
            closer = inside & (t > 0) & (t < depth[sub])
            if not closer.any():
                continue
            lambert = max(0.0, float(normal @ light_cam))
            sky = 0.5 + 0.5 * float(normal @ up_cam)
            value = exposure * (mat['albedo'] * (mat['ambient'] + mat['diffuse'] * lambert) + mat.get('sky', 0.0) * sky)
            depth[sub][closer] = t[closer]
            shade[sub][closer] = value
    covered = np.isfinite(depth)
    coverage = covered.reshape(h, s, w, s).mean((1, 3))
    hi_bg = np.repeat(np.repeat(background[y0:y1 + 1, x0:x1 + 1].astype(float), s, 0), s, 1)
    texture = 1 + 0.03 * ndimage.gaussian_filter(rng.standard_normal(shade.shape), s * 0.8)
    hi = np.where(covered[..., None], (shade * texture)[..., None] * 255.0, hi_bg)
    low = hi.reshape(h, s, w, s, 3).mean((1, 3))
    out = background.astype(float).copy()
    out[y0:y1 + 1, x0:x1 + 1] = low
    return out, coverage, (x0, y0, w, h), polygons


def shadow(image, camera, footprint_city, strength=0.5, blur_px=1.5):
    """Darken the ground inside a projected footprint polygon (soft edge)."""
    uv = camera.project_camera(camera.to_camera(footprint_city))
    polygon = hull_polygon(uv)
    x0, y0 = int(np.floor(polygon[:, 0].min())) - 6, int(np.floor(polygon[:, 1].min())) - 6
    x1, y1 = int(np.ceil(polygon[:, 0].max())) + 6, int(np.ceil(polygon[:, 1].max())) + 6
    xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
    mask = ndimage.gaussian_filter(inside_convex(polygon, xs, ys).astype(float), blur_px)
    region = image[y0:y1 + 1, x0:x1 + 1]
    image[y0:y1 + 1, x0:x1 + 1] = region * (1 - strength * mask[..., None])
    return image


def paint(image, camera, quads_city, delta, supersample=8, wear=0.25, seed=0):
    """Add bright paint quads on the road. Returns (image, coverage map dict, region)."""
    rng = np.random.default_rng(seed)
    polys = [camera.project_camera(camera.to_camera(q)) for q in quads_city]
    allv = np.vstack(polys)
    x0, y0 = int(np.floor(allv[:, 0].min())) - 3, int(np.floor(allv[:, 1].min())) - 3
    x1, y1 = int(np.ceil(allv[:, 0].max())) + 3, int(np.ceil(allv[:, 1].max())) + 3
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(camera.width - 1, x1), min(camera.height - 1, y1)
    w, h, s = x1 - x0 + 1, y1 - y0 + 1, supersample
    xs = x0 - 0.5 + (np.arange(w * s) + 0.5) / s
    ys = y0 - 0.5 + (np.arange(h * s) + 0.5) / s
    gx, gy = np.meshgrid(xs, ys)
    hit = np.zeros(gx.shape, bool)
    for poly in polys:
        hit |= inside_convex(poly, gx, gy)
    coverage = hit.reshape(h, s, w, s).mean((1, 3))
    worn = np.clip(1 - wear * np.abs(ndimage.gaussian_filter(rng.standard_normal((h, w)), 1.2)) * 3, 0.55, 1.0)
    region = image[y0:y1 + 1, x0:x1 + 1]
    image[y0:y1 + 1, x0:x1 + 1] = region + (coverage * worn * delta)[..., None]
    return image, coverage, (x0, y0, w, h), polys


def camera_finish(image, mask, blur=0.5, noise=1.2, quality=92, seed=0):
    """Optical blur + sensor noise inside `mask` (edited pixels, dilated), then JPEG encode the whole frame."""
    rng = np.random.default_rng(seed)
    blurred = ndimage.gaussian_filter(image, (blur, blur, 0))
    noisy = blurred + rng.normal(0, noise, image.shape[:2])[..., None]
    m = ndimage.binary_dilation(mask, iterations=3)[..., None]
    out = np.where(m, noisy, image)
    gray = np.clip(out.mean(-1) + 0.5, 0, 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, -1)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format='JPEG', quality=quality, subsampling=0)
    return buffer.getvalue()


def silhouette_truth(polygons, columns):
    """Exact extremes and top/bottom boundary rows at the given pixel-centre columns."""
    allv = np.vstack(polygons)
    truth = {'L': float(allv[:, 0].min()), 'R': float(allv[:, 0].max()), 'T': float(allv[:, 1].min()), 'B': float(allv[:, 1].max())}
    profile = {}
    for c in columns:
        spans = [column_interval(p, float(c)) for p in polygons]
        spans = [sp for sp in spans if sp is not None]
        profile[int(c)] = None if not spans else (float(min(sp[0] for sp in spans)), float(max(sp[1] for sp in spans)))
    return truth, profile
