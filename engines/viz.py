"""Draw bounding boxes on a page image so we can see layout understanding."""
from __future__ import annotations

from typing import Any, Sequence

from PIL import Image, ImageDraw


def _flatten_bbox(bbox: Any, img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    """Return (x0, y0, x1, y1) in pixel coords, or None if we can't parse it."""
    if bbox is None:
        return None
    try:
        # 4-point polygon: [[x, y], [x, y], [x, y], [x, y]]
        if isinstance(bbox, Sequence) and len(bbox) == 4 and isinstance(bbox[0], Sequence) and len(bbox[0]) == 2:
            xs = [float(p[0]) for p in bbox]
            ys = [float(p[1]) for p in bbox]
            return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        # Flat (x0, y0, x1, y1)
        if isinstance(bbox, Sequence) and len(bbox) == 4 and not isinstance(bbox[0], Sequence):
            x0, y0, x1, y1 = (float(v) for v in bbox)
            # docTR returns normalized geometry: ((xmin, ymin), (xmax, ymax))? Sometimes 0..1
            if max(x0, y0, x1, y1) <= 1.5:
                return int(x0 * img_w), int(y0 * img_h), int(x1 * img_w), int(y1 * img_h)
            return int(x0), int(y0), int(x1), int(y1)
        # docTR geometry: ((xmin, ymin), (xmax, ymax)) normalized
        if isinstance(bbox, Sequence) and len(bbox) == 2 and isinstance(bbox[0], Sequence) and len(bbox[0]) == 2:
            (x0, y0), (x1, y1) = bbox
            if max(x0, y0, x1, y1) <= 1.5:
                return int(x0 * img_w), int(y0 * img_h), int(x1 * img_w), int(y1 * img_h)
            return int(x0), int(y0), int(x1), int(y1)
    except Exception:
        return None
    return None


def draw_boxes(page: Image.Image, lines: list[dict], color: str = "#ff2d55") -> Image.Image:
    img = page.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for line in lines:
        box = _flatten_bbox(line.get("bbox"), w, h)
        if box is None:
            continue
        draw.rectangle(box, outline=color, width=2)
    return img
