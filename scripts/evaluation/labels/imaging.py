"""Image rendering for the labeling kit: gridded overview crops, zoom panels and candidate overlays.

Rendering never shows LiDAR, AV2 boxes, the map or any app output. Only the raw photo, a grid, guide
lines placed by the task, and the single candidate edge or pixel under test.
Coordinates follow av2io: pixel (i, j) centre at (i, j); boundaries at half-integers.
"""
from __future__ import annotations

import io
import string

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

MAGENTA = (255, 0, 255)
YELLOW = (255, 214, 0)
CYAN = (0, 230, 255)
CANDIDATE_COLORS = [(255, 64, 64), (64, 200, 255), (255, 200, 0), (80, 255, 120), (255, 128, 255), (255, 255, 255)]


def font(size=14):
    return ImageFont.load_default(size=size)


def load_rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert('RGB'))


def luminance(rgb):
    rgb = np.asarray(rgb, float)
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def column_label(index):
    letters = string.ascii_uppercase
    label = ''
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        label = letters[rem] + label
    return label


def parse_column_label(text):
    value = 0
    for ch in text:
        if not ch.isalpha():
            raise ValueError(text)
        value = value * 26 + (ord(ch.upper()) - 64)
    return value - 1


def crop_padded(rgb, x0, y0, width, height, fill=0):
    """Crop [x0, x0+width) x [y0, y0+height) in pixel indices; outside the image is `fill`."""
    out = np.full((height, width, 3), fill, np.uint8)
    h, w = rgb.shape[:2]
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + width), min(h, y0 + height)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgb[sy0:sy1, sx0:sx1]
    return out


def enhance(rgb, low=1.0, high=99.0, gamma=0.7):
    """Local contrast stretch of one crop (percentiles over the crop), gamma lift for dark scenes."""
    lum = luminance(rgb)
    lo, hi = np.percentile(lum, [low, high])
    if hi - lo < 1e-6:
        return rgb.copy()
    scale = np.clip((np.asarray(rgb, float) - lo) / (hi - lo), 0, 1) ** gamma
    return (scale * 255 + 0.5).astype(np.uint8)


def upscale(rgb, zoom):
    return np.repeat(np.repeat(rgb, zoom, axis=0), zoom, axis=1)


def resize_to(rgb, width, height):
    return np.array(Image.fromarray(rgb).resize((width, height), Image.NEAREST))


def draw_pixel_grid(draw, width_px, height_px, zoom, strong_every=5, origin=(0, 0)):
    """Thin lines on every source-pixel boundary of a zoom panel (only drawn when zoom >= 6)."""
    if zoom < 6:
        return
    ox, oy = origin
    for i in range(width_px + 1):
        x = ox + i * zoom
        draw.line([(x, oy), (x, oy + height_px * zoom - 1)], fill=(0, 0, 0, 60 if i % strong_every else 120), width=1)
    for j in range(height_px + 1):
        y = oy + j * zoom
        draw.line([(ox, y), (ox + width_px * zoom - 1, y)], fill=(0, 0, 0, 60 if j % strong_every else 120), width=1)


def to_png(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG', optimize=False)
    return buffer.getvalue()


def caption(draw, xy, text, size=14):
    f = font(size)
    x, y = xy
    box = draw.textbbox((x, y), text, font=f)
    draw.rectangle([box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2], fill=(255, 255, 255, 235))
    draw.text((x, y), text, fill=(0, 0, 0, 255), font=f)


def render_overview(rgb, region, cell, max_width=820, guides=(), title='', ring=None):
    """Gridded overview of `region` = (x0, y0, width, height) in source pixels.

    Returns PNG bytes and the grid metadata. The grid cells are `cell` source pixels; columns are lettered
    A, B, ... from region x0 and rows numbered 1, 2, ... from region y0. The raw crop is on top and the
    contrast-enhanced crop below, both with the same grid. `guides` are vertical dashed lines at source
    columns (task-defined sampling columns, never answers).
    """
    x0, y0, width, height = region
    columns, rows = -(-width // cell), -(-height // cell)
    width, height = columns * cell, rows * cell
    raw = crop_padded(rgb, x0, y0, width, height)
    scale = max(1, int(max_width // width)) if width <= max_width else max_width / width
    if scale >= 1 and float(scale).is_integer():
        scale = int(scale)
        disp_w, disp_h = width * scale, height * scale
    else:
        disp_w, disp_h = int(round(width * scale)), int(round(height * scale))
    margin_left, margin_top, gap = 34, 26, 30
    canvas = Image.new('RGBA', (margin_left + disp_w + 8, margin_top + 2 * disp_h + gap + 8), (255, 255, 255, 255))
    overlay = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    f = font(13)
    for k, panel in enumerate((raw, enhance(raw))):
        top = margin_top + k * (disp_h + gap)
        canvas.paste(Image.fromarray(resize_to(panel, disp_w, disp_h)), (margin_left, top))
        for c in range(columns + 1):
            x = margin_left + c * cell * disp_w / width
            draw.line([(x, top), (x, top + disp_h)], fill=YELLOW + (150,), width=1)
            if c < columns:
                label = column_label(c)
                draw.text((x + cell * disp_w / width / 2 - 4 * len(label), top - 15), label, fill=(0, 0, 0, 255), font=f)
        for r in range(rows + 1):
            y = top + r * cell * disp_h / height
            draw.line([(margin_left, y), (margin_left + disp_w, y)], fill=YELLOW + (150,), width=1)
            if r < rows:
                draw.text((4, y + cell * disp_h / height / 2 - 7), str(r + 1), fill=(0, 0, 0, 255), font=f)
        for g in guides:
            gx = margin_left + (g - x0 + 0.5) * disp_w / width
            for yy in range(int(top), int(top + disp_h), 10):
                draw.line([(gx, yy), (gx, min(yy + 5, top + disp_h))], fill=CYAN + (255,), width=2)
        if ring is not None:
            rx = margin_left + (ring[0] - x0 + 0.5) * disp_w / width
            ry = top + (ring[1] - y0 + 0.5) * disp_h / height
            draw.ellipse([rx - 8, ry - 8, rx + 8, ry + 8], outline=CYAN + (255,), width=2)
        caption(draw, (margin_left + disp_w - 90, top + 4), 'RAW' if k == 0 else 'ENHANCED', 12)
    if title:
        caption(draw, (margin_left + 4, margin_top + 4), title, 12)
    image = Image.alpha_composite(canvas, overlay).convert('RGB')
    meta = {'region': [int(x0), int(y0), int(width), int(height)], 'cell': int(cell), 'columns': int(columns), 'rows': int(rows),
            'columnLabels': [column_label(0), column_label(columns - 1)], 'rowLabels': [1, rows]}
    return to_png(image), meta


def cell_bounds(meta, label):
    """'C4' -> source pixel index ranges (x_first, x_last, y_first, y_last) of that cell."""
    label = label.strip().upper()
    letters = ''.join(ch for ch in label if ch.isalpha())
    digits = ''.join(ch for ch in label if ch.isdigit())
    if not letters or not digits or letters + digits != label:
        raise ValueError(f'Cell must look like C4, got {label!r}')
    column, row = parse_column_label(letters), int(digits) - 1
    if not (0 <= column < meta['columns'] and 0 <= row < meta['rows']):
        raise ValueError(f'Cell {label} is outside the grid {meta["columnLabels"][0]}1..{meta["columnLabels"][1]}{meta["rowLabels"][1]}')
    x0, y0, cell = meta['region'][0], meta['region'][1], meta['cell']
    return x0 + column * cell, x0 + (column + 1) * cell - 1, y0 + row * cell, y0 + (row + 1) * cell - 1


def parse_box(meta, text):
    """'C4:H9' -> pixel index ranges (x_first, x_last, y_first, y_last) covering both corner cells."""
    parts = text.replace(' ', '').upper().split(':')
    if len(parts) != 2:
        raise ValueError('Box must look like C4:H9 (top-left cell : bottom-right cell)')
    a, b = cell_bounds(meta, parts[0]), cell_bounds(meta, parts[1])
    if b[0] < a[0] or b[2] < a[2]:
        raise ValueError('The second cell must be right of and below (or equal to) the first cell')
    return a[0], b[1], a[2], b[3]


def zoom_panel(rgb, window, zoom, enhanced):
    x0, y0, w, h = window
    crop = crop_padded(rgb, x0, y0, w, h)
    if enhanced:
        crop = enhance(crop)
    return Image.fromarray(upscale(crop, zoom)).convert('RGBA')


def render_question(rgb, context_region, window, zoom, marker, candidates=(), guides=(), context_width=360,
                    tick_on_pixel_boundary=False):
    """Three panels: CONTEXT (scaled region with the marker and a yellow zoom frame), RAW zoom, ENHANCED zoom.

    marker: {'type': 'vline'|'hline', 'at': boundary coordinate} or {'type': 'pixel', 'at': (i, j)}.
    candidates: [(number, axis, coordinate, fixed)] numbered ticks for a pick question (pixel scans).
    tick_on_pixel_boundary (protocol v2, coverage-design.md section 3): draw each tick on the pixel boundary nearest
    the candidate, floor(c) + 0.5, so a tick never sits inside a pixel. Only the drawing moves; brackets and snapping
    keep the sub-pixel candidate. False (protocol v1) draws at the sub-pixel coordinate as before.
    """
    cx0, cy0, cw, ch = context_region
    ctx = crop_padded(rgb, cx0, cy0, cw, ch)
    cscale = context_width / cw
    cdisp = (context_width, max(1, int(round(ch * cscale))))
    ctx_img = Image.fromarray(resize_to(enhance(ctx), *cdisp)).convert('RGBA')
    x0, y0, w, h = window
    raw = zoom_panel(rgb, window, zoom, False)
    enh = zoom_panel(rgb, window, zoom, True)
    zw, zh = w * zoom, h * zoom
    gap, top = 14, 24
    total_w = cdisp[0] + 2 * (zw + gap) + 4
    total_h = top + max(cdisp[1], zh) + 6
    canvas = Image.new('RGBA', (total_w, total_h), (255, 255, 255, 255))
    canvas.paste(ctx_img, (0, top))
    canvas.paste(raw, (cdisp[0] + gap, top))
    canvas.paste(enh, (cdisp[0] + gap + zw + gap, top))
    overlay = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # context markings
    def cmap(x, y):
        return (x - cx0 + 0.5) * cscale, top + (y - cy0 + 0.5) * cscale
    fx0, fy0 = cmap(x0 - 0.5, y0 - 0.5)
    fx1, fy1 = cmap(x0 + w - 0.5, y0 + h - 0.5)
    draw.rectangle([fx0, fy0, fx1, fy1], outline=YELLOW + (255,), width=2)
    for origin_x in (cdisp[0] + gap, cdisp[0] + gap + zw + gap):
        draw_pixel_grid(draw, w, h, zoom, origin=(origin_x, top))
    def zmap(origin_x, x, y):
        return origin_x + (x - x0 + 0.5) * zoom, top + (y - y0 + 0.5) * zoom
    for g in guides:
        gx, _ = cmap(g, 0)
        for yy in range(top, top + cdisp[1], 8):
            draw.line([(gx, yy), (gx, min(yy + 4, top + cdisp[1]))], fill=CYAN + (255,), width=1)
    if marker['type'] in ('column', 'row'):
        at = marker['at']
        for edge in (at - 0.5, at + 0.5):
            if marker['type'] == 'column':
                lx, _ = cmap(edge, 0)
                draw.line([(lx, top), (lx, top + cdisp[1])], fill=MAGENTA + (255,), width=1)
                for origin_x in (cdisp[0] + gap, cdisp[0] + gap + zw + gap):
                    zx, _ = zmap(origin_x, edge, 0)
                    draw.line([(zx, top), (zx, top + zh - 1)], fill=MAGENTA + (255,), width=1)
            else:
                _, ly = cmap(0, edge)
                draw.line([(0, ly), (cdisp[0], ly)], fill=MAGENTA + (255,), width=1)
                for origin_x in (cdisp[0] + gap, cdisp[0] + gap + zw + gap):
                    _, zy = zmap(origin_x, 0, edge)
                    draw.line([(origin_x, zy), (origin_x + zw - 1, zy)], fill=MAGENTA + (255,), width=1)
    elif marker['type'] in ('vline', 'hline'):
        at = marker['at']
        if marker['type'] == 'vline':
            lx, _ = cmap(at, 0)
            draw.line([(lx, top), (lx, top + cdisp[1])], fill=MAGENTA + (255,), width=2)
            for origin_x in (cdisp[0] + gap, cdisp[0] + gap + zw + gap):
                zx, _ = zmap(origin_x, at, 0)
                draw.line([(zx, top), (zx, top + zh - 1)], fill=MAGENTA + (255,), width=2)
        else:
            _, ly = cmap(0, at)
            draw.line([(0, ly), (cdisp[0], ly)], fill=MAGENTA + (255,), width=2)
            for origin_x in (cdisp[0] + gap, cdisp[0] + gap + zw + gap):
                _, zy = zmap(origin_x, 0, at)
                draw.line([(origin_x, zy), (origin_x + zw - 1, zy)], fill=MAGENTA + (255,), width=2)
    else:
        i, j = marker['at']
        px, py = cmap(i, j)
        r = 5
        draw.line([(px - 3 * r, py), (px - r, py)], fill=MAGENTA + (255,), width=2)
        draw.line([(px + r, py), (px + 3 * r, py)], fill=MAGENTA + (255,), width=2)
        draw.line([(px, py - 3 * r), (px, py - r)], fill=MAGENTA + (255,), width=2)
        draw.line([(px, py + r), (px, py + 3 * r)], fill=MAGENTA + (255,), width=2)
        for origin_x in (cdisp[0] + gap, cdisp[0] + gap + zw + gap):
            zx, zy = zmap(origin_x, i, j)
            half = zoom / 2
            draw.rectangle([zx - half - 2, zy - half - 2, zx + half + 1, zy + half + 1], outline=MAGENTA + (255,), width=2)
            arm = 2.5 * zoom
            draw.line([(zx - half - 2 - arm, zy), (zx - half - 3, zy)], fill=MAGENTA + (255,), width=2)
            draw.line([(zx + half + 2, zy), (zx + half + 2 + arm, zy)], fill=MAGENTA + (255,), width=2)
    for number, axis, coordinate, fixed in candidates:
        if tick_on_pixel_boundary:
            coordinate = float(np.floor(coordinate)) + 0.5
        color = CANDIDATE_COLORS[(number - 1) % len(CANDIDATE_COLORS)] + (255,)
        for origin_x in (cdisp[0] + gap, cdisp[0] + gap + zw + gap):
            if axis == 'y':
                # horizontal tick at boundary row `coordinate`, beside (not over) scan column `fixed`
                zx, zy = zmap(origin_x, fixed, coordinate)
                left0, left1 = zx - zoom * 5.5, zx - zoom * 1.5
                right0, right1 = zx + zoom * 1.5, zx + zoom * 5.5
                draw.line([(left0, zy), (left1, zy)], fill=color, width=2)
                draw.line([(right0, zy), (right1, zy)], fill=color, width=2)
                draw.text((right1 + 3, zy - 8), str(number), fill=color, font=font(14))
            else:
                zx, zy = zmap(origin_x, coordinate, fixed)
                draw.line([(zx, zy - zoom * 5.5), (zx, zy - zoom * 1.5)], fill=color, width=2)
                draw.line([(zx, zy + zoom * 1.5), (zx, zy + zoom * 5.5)], fill=color, width=2)
                draw.text((zx + 3, zy + zoom * 5.5), str(number), fill=color, font=font(14))
    caption(draw, (4, 4), 'CONTEXT (enhanced)', 12)
    caption(draw, (cdisp[0] + gap + 4, 4), f'RAW x{zoom}', 12)
    caption(draw, (cdisp[0] + gap + zw + gap + 4, 4), f'ENHANCED x{zoom}', 12)
    image = Image.alpha_composite(canvas, overlay).convert('RGB')
    return to_png(image)


def gradient_profile(rgb, axis, fixed, lo, hi, sigma=0.7, half_width=1):
    """Colour-gradient magnitude along a scanline, averaged over +-half_width perpendicular pixels.

    axis 'y': scan rows lo..hi in column `fixed`; axis 'x': scan columns lo..hi in row `fixed`.
    Returns (indices, magnitude, signed luminance derivative) with the derivative taken in the
    direction of increasing index.
    """
    rgb = np.asarray(rgb, float)
    pad = int(4 * sigma) + 3
    if axis == 'y':
        c0, c1 = int(fixed) - half_width, int(fixed) + half_width + 1
        r0, r1 = int(lo) - pad, int(hi) + pad + 1
        block = crop_padded(rgb.astype(np.uint8), c0, r0, c1 - c0, r1 - r0).astype(float)
        deriv = ndimage.gaussian_filter1d(block, sigma, axis=0, order=1)
        smooth = ndimage.gaussian_filter1d(deriv, sigma, axis=1)
        magnitude = np.sqrt((smooth ** 2).sum(-1)).mean(1)
        signed = luminance(smooth).mean(1)
    else:
        r0, r1 = int(fixed) - half_width, int(fixed) + half_width + 1
        c0, c1 = int(lo) - pad, int(hi) + pad + 1
        block = crop_padded(rgb.astype(np.uint8), c0, r0, c1 - c0, r1 - r0).astype(float)
        deriv = ndimage.gaussian_filter1d(block, sigma, axis=1, order=1)
        smooth = ndimage.gaussian_filter1d(deriv, sigma, axis=0)
        magnitude = np.sqrt((smooth ** 2).sum(-1)).mean(0)
        signed = luminance(smooth).mean(0)
    indices = np.arange(int(lo) - pad, int(hi) + pad + 1)
    keep = (indices >= lo) & (indices <= hi)
    return indices[keep], magnitude[keep], signed[keep]
