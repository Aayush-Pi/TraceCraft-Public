"""
engine.py - TraceCraft: True Painterly Hierarchical Vector Engine + Clean Edit Vectorizer.
Dual-Engine:
1. Painterly Studio Mode (High-detail inverse-graphics with progressive Béziers)
2. Clean Graphic Mode (Low object count, clean segmented polygons for design & editing)
"""

import base64
import math
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn


class TraceCraftEngine:
    MAX_DIM = 720          # working / output resolution (longest side)
    MAX_UPSCALE = 3.0      # small inputs are upscaled toward MAX_DIM, but never more than this
    BASE_DIM = 360.0       # resolution the original tuning constants were designed for

    def __init__(self) -> None:
        self.device = "cpu"
        self.model_name = "Dual-Engine Neuro-Symbolic & Clean Graphic Vectorizer"

        self.face_cascade = None
        self.eye_cascade = None
        try:
            if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                self.eye_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_eye.xml"
                )
        except Exception:
            pass

    @staticmethod
    def _rgb_to_hex(r: float, g: float, b: float) -> str:
        return f"#{int(np.clip(r, 0, 255)):02x}{int(np.clip(g, 0, 255)):02x}{int(np.clip(b, 0, 255)):02x}"

    @staticmethod
    def _array_to_png_b64(img_bgr: np.ndarray) -> str:
        success, buffer = cv2.imencode(".png", img_bgr)
        return base64.b64encode(buffer.tobytes()).decode("utf-8") if success else ""

    def compute_importance_map(
        self, img_rgb: np.ndarray
    ) -> Tuple[np.ndarray, str, np.ndarray, List[Tuple[int, int, int, int]]]:
        h, w = img_rgb.shape[:2]
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        canny = cv2.Canny(gray, 25, 90).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        grad_norm = grad / (np.max(grad) + 1e-6)

        rs = max(h, w) / self.BASE_DIM
        k = max(11, int(round(11 * rs)) | 1)
        blur = cv2.GaussianBlur(gray, (k, k), 0).astype(np.float32)
        contrast = np.abs(gray.astype(np.float32) - blur)
        contrast_norm = contrast / (np.max(contrast) + 1e-6)

        saliency = 0.40 * canny + 0.35 * grad_norm + 0.25 * contrast_norm

        face_mask = np.zeros((h, w), dtype=np.float32)
        face_boxes: List[Tuple[int, int, int, int]] = []
        face_prior = np.zeros((h, w), dtype=np.float32)
        eye_prior = np.zeros((h, w), dtype=np.float32)

        if self.face_cascade is not None and not self.face_cascade.empty():
            try:
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(20, 20))
                for fx, fy, fw, fh in faces:
                    pad_x, pad_y = int(fw * 0.15), int(fh * 0.20)
                    x0, y0 = max(0, fx - pad_x), max(0, fy - pad_y)
                    x1, y1 = min(w, fx + fw + pad_x), min(h, fy + fh + pad_y)
                    face_boxes.append((x0, y0, x1, y1))
                    cv2.ellipse(
                        face_prior,
                        ((x0 + x1) // 2, (y0 + y1) // 2),
                        (max(1, (x1 - x0) // 2), max(1, (y1 - y0) // 2)),
                        0, 0, 360, 1.0, -1,
                    )
                    cv2.rectangle(face_mask, (x0, y0), (x1, y1), 1.0, -1)

                    if self.eye_cascade is not None and not self.eye_cascade.empty():
                        roi_gray = gray[fy : fy + fh, fx : fx + fw]
                        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 2, minSize=(6, 6))
                        for ex, ey, ew, eh in eyes:
                            cv2.ellipse(
                                eye_prior,
                                (fx + ex + ew // 2, fy + ey + eh // 2),
                                (max(1, ew // 2), max(1, eh // 2)),
                                0, 0, 360, 1.0, -1,
                            )
            except Exception:
                pass

        if not face_boxes:
            sal_thresh = saliency > np.percentile(saliency, 65)
            face_mask[sal_thresh] = 1.0

        sigma = max(2.0, min(h, w) * 0.03)
        face_soft = cv2.GaussianBlur(face_prior, (0, 0), sigma)
        eye_soft = cv2.GaussianBlur(eye_prior, (0, 0), sigma * 0.5)
        edge_norm = np.clip(saliency / (np.percentile(saliency, 99.5) + 1e-6), 0.0, 1.0) ** 0.6
        fused = 0.75 * edge_norm + 0.30 * face_soft + 0.35 * eye_soft
        norm_sal = np.clip(fused / (np.max(fused) + 1e-6), 0.0, 1.0).astype(np.float32)
        weight_map = 1.0 + 8.0 * cv2.GaussianBlur(norm_sal, (5, 5), 1.0)

        heatmap_b64 = self._array_to_png_b64(
            cv2.applyColorMap((norm_sal * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        )
        return weight_map.astype(np.float32), heatmap_b64, face_mask, face_boxes

    @staticmethod
    def _poly_to_smooth_q_bezier(poly: np.ndarray) -> List[List[float]]:
        n = len(poly)
        if n < 3:
            return [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in poly]

        anchors = []
        controls = []
        for i in range(n):
            p_curr = poly[i]
            p_next = poly[(i + 1) % n]
            mid = 0.5 * (p_curr + p_next)
            anchors.append(mid)
            controls.append(p_next)

        pts: List[List[float]] = [[round(float(anchors[0][0]), 2), round(float(anchors[0][1]), 2)]]
        for i in range(n):
            ctrl = controls[i]
            next_anc = anchors[(i + 1) % n]
            pts.append([round(float(ctrl[0]), 2), round(float(ctrl[1]), 2)])
            pts.append([round(float(next_anc[0]), 2), round(float(next_anc[1]), 2)])

        return pts

    def _extract_painterly_pass(
        self,
        img_smooth: np.ndarray,
        target_f: np.ndarray,
        canvas: np.ndarray,
        weight_map: np.ndarray,
        div: int,
        min_area: float,
        max_area: float,
        eps_factor: float,
        alpha: float,
        stage_id: int,
        use_q_curves: bool,
        max_count: int,
        start_step: int,
        roi_mask: np.ndarray = None,
        max_colors: int = 64,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], int]:
        h, w = target_f.shape[:2]
        shapes: List[Dict[str, Any]] = []
        step = start_step

        q = ((img_smooth.astype(np.int32) // div) * div + div // 2)
        key = (q[..., 0] << 20) | (q[..., 1] << 10) | q[..., 2]
        if roi_mask is not None:
            active = key[roi_mask > 0]
            if active.size == 0:
                return canvas, shapes, step
            unique_keys, counts = np.unique(active, return_counts=True)
        else:
            unique_keys, counts = np.unique(key, return_counts=True)

        order = np.argsort(-counts)
        unique_keys = unique_keys[order[: min(max_colors, len(unique_keys))]]

        candidates = []
        for ck in unique_keys:
            color_match = (key == ck).astype(np.uint8) * 255
            if roi_mask is not None:
                color_match = cv2.bitwise_and(color_match, roi_mask)

            contours, _ = cv2.findContours(color_match, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue

                arc = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, eps_factor * arc, True).reshape(-1, 2)
                if len(approx) < 3:
                    continue

                x0, y0 = max(0, np.min(approx[:, 0])), max(0, np.min(approx[:, 1]))
                x1, y1 = min(w, np.max(approx[:, 0]) + 1), min(h, np.max(approx[:, 1]) + 1)
                if x1 <= x0 or y1 <= y0:
                    continue

                sal_score = float(np.mean(weight_map[y0:y1, x0:x1]))
                candidates.append((area, sal_score, approx, (x0, y0, x1, y1)))

        if stage_id in (1, 2):
            candidates.sort(key=lambda x: x[0], reverse=True)
        else:
            candidates.sort(key=lambda x: x[1] * math.sqrt(x[0]), reverse=True)

        for area, sal_score, poly, (x0, y0, x1, y1) in candidates:
            if len(shapes) >= max_count:
                break

            local_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            cv2.fillPoly(local_mask, [poly - np.array([x0, y0])], 255)
            m_bool = local_mask > 0
            if not np.any(m_bool):
                continue

            patch_t = target_f[y0:y1, x0:x1]
            patch_c = canvas[y0:y1, x0:x1]

            rgb = np.mean(patch_t[m_bool], axis=0)

            old_err = float(np.sum((patch_t[m_bool] - patch_c[m_bool]) ** 2))
            new_pixels = patch_c[m_bool] * (1.0 - alpha) + rgb * alpha
            new_err = float(np.sum((patch_t[m_bool] - new_pixels) ** 2))

            if new_err >= old_err:
                continue

            patch_c[m_bool] = new_pixels
            step += 1
            hex_col = self._rgb_to_hex(rgb[0], rgb[1], rgb[2])

            if use_q_curves and len(poly) >= 4 and (step % 2 == 0):
                q_pts = self._poly_to_smooth_q_bezier(poly.astype(np.float32))
                shapes.append({
                    "id": f"s{stage_id}_q_{step}",
                    "step": step,
                    "stage": stage_id,
                    "type": "path",
                    "kind": "Q",
                    "points": q_pts,
                    "color": hex_col,
                    "fill": hex_col,
                    "opacity": round(alpha, 3),
                    "error_reduction": round((old_err - new_err) / 100.0, 2),
                })
            else:
                shapes.append({
                    "id": f"s{stage_id}_p_{step}",
                    "step": step,
                    "stage": stage_id,
                    "type": "polygon",
                    "points": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in poly],
                    "color": hex_col,
                    "fill": hex_col,
                    "opacity": round(alpha, 3),
                    "error_reduction": round((old_err - new_err) / 100.0, 2),
                })

        return canvas, shapes, step

    def _stage3_fine_features_and_outlines(
        self,
        target_f: np.ndarray,
        canvas: np.ndarray,
        weight_map: np.ndarray,
        n_remaining: int,
        start_step: int,
        rs: float = 1.0,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], int, str]:
        h, w = target_f.shape[:2]
        shapes: List[Dict[str, Any]] = []
        step = start_step

        if n_remaining <= 0:
            return canvas, shapes, step, "budget_reached"

        gray = cv2.cvtColor(target_f.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        tangent_angles = np.arctan2(-gx, gy + 1e-6)

        edges = cv2.Canny(gray, 35, 110)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        scored_chains = []
        for c in contours:
            c = c.squeeze(axis=1)
            if c.ndim == 2 and len(c) >= 6:
                py = np.clip(c[:, 1], 0, h - 1)
                px = np.clip(c[:, 0], 0, w - 1)
                local_err = np.mean(np.sum((target_f[py, px] - canvas[py, px]) ** 2, axis=1))
                score = float(np.mean(weight_map[py, px])) * math.sqrt(local_err + 1.0)
                scored_chains.append((score, c))

        scored_chains.sort(key=lambda x: x[0], reverse=True)
        n_contours = min(len(scored_chains), int(n_remaining * 0.30))
        seg_cap = max(10, int(round(10 * rs)))

        for idx in range(n_contours):
            _, chain = scored_chains[idx]
            seg_len = min(len(chain), seg_cap)
            sub = chain[:seg_len].astype(np.float32)

            p0 = sub[0]
            p3 = sub[-1]
            p1 = sub[seg_len // 3]
            p2 = sub[(2 * seg_len) // 3]

            dx_v, dy_v = p3[0] - p0[0], p3[1] - p0[1]
            L = math.hypot(dx_v, dy_v) + 1e-6
            nx, ny = -dy_v / L * 1.1 * rs, dx_v / L * 1.1 * rs

            ribbon = np.array(
                [p0, p1, p2, p3, [p2[0] + nx, p2[1] + ny], [p1[0] + nx, p1[1] + ny], p0],
                dtype=np.float32,
            )
            ribbon[:, 0] = np.clip(ribbon[:, 0], 0, w - 1)
            ribbon[:, 1] = np.clip(ribbon[:, 1], 0, h - 1)

            sub_i = sub.astype(np.int32)
            cy = np.clip(sub_i[:, 1], 0, h - 1)
            cx = np.clip(sub_i[:, 0], 0, w - 1)
            rgb = np.percentile(target_f[cy, cx], 25, axis=0)
            alpha = 0.88

            old_e = float(np.sum((target_f[cy, cx] - canvas[cy, cx]) ** 2))
            new_p = canvas[cy, cx] * (1.0 - alpha) + rgb * alpha
            new_e = float(np.sum((target_f[cy, cx] - new_p) ** 2))

            if new_e < old_e:
                canvas[cy, cx] = new_p
                step += 1
                hex_col = self._rgb_to_hex(rgb[0], rgb[1], rgb[2])
                shapes.append({
                    "id": f"s3_c_{step}",
                    "step": step,
                    "stage": 3,
                    "type": "path",
                    "kind": "C",
                    "points": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in ribbon],
                    "color": hex_col,
                    "fill": hex_col,
                    "opacity": alpha,
                    "error_reduction": round((old_e - new_e) / 50.0, 2),
                })

        rem = n_remaining - len(shapes)
        work_weights = weight_map.astype(np.float32).copy()
        err_map = np.sum((target_f - canvas) ** 2, axis=2).astype(np.float32) * work_weights

        dab = rs * 0.8
        accepted = 0
        attempts = 0
        stale = 0
        max_attempts = max(rem * 4, 200)
        stale_limit = 300
        stop_reason = "budget_reached"

        while accepted < rem:
            if attempts >= max_attempts:
                stop_reason = "attempt_limit"
                break
            if stale >= stale_limit:
                stop_reason = "stagnated"
                break
            attempts += 1

            by, bx = np.unravel_index(np.argmax(err_map), (h, w))
            if err_map[by, bx] < 4.0:
                stop_reason = "converged"
                break

            progress = accepted / max(1, rem)
            rad_major = max(1.8 * dab, 4.5 * dab * (1.0 - 0.65 * progress))
            rad_minor = max(1.3 * dab, rad_major * 0.65)
            ang_deg = math.degrees(float(tangent_angles[by, bx]))

            pts = cv2.ellipse2Poly(
                (int(bx), int(by)),
                (int(round(rad_major)), int(round(rad_minor))),
                int(round(ang_deg)),
                0,
                360,
                45,
            )

            ok = len(pts) >= 3
            if ok:
                x0, y0 = max(0, np.min(pts[:, 0])), max(0, np.min(pts[:, 1]))
                x1, y1 = min(w, np.max(pts[:, 0]) + 1), min(h, np.max(pts[:, 1]) + 1)
                ok = x1 > x0 and y1 > y0
            if ok:
                local_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
                cv2.fillPoly(local_mask, [pts - np.array([x0, y0])], 255)
                m_bool = local_mask > 0
                ok = bool(np.any(m_bool))

            accept = False
            if ok:
                patch_t = target_f[y0:y1, x0:x1]
                patch_c = canvas[y0:y1, x0:x1]

                rgb = np.mean(patch_t[m_bool], axis=0)
                alpha = 0.85

                old_err = float(np.sum((patch_t[m_bool] - patch_c[m_bool]) ** 2))
                new_pixels = patch_c[m_bool] * (1.0 - alpha) + rgb * alpha
                new_err = float(np.sum((patch_t[m_bool] - new_pixels) ** 2))
                accept = new_err < old_err

            if accept:
                patch_c[m_bool] = new_pixels
                err_map[y0:y1, x0:x1] = (
                    np.sum((patch_t - patch_c) ** 2, axis=2) * work_weights[y0:y1, x0:x1]
                )
                accepted += 1
                stale = 0
                step += 1
                hex_col = self._rgb_to_hex(rgb[0], rgb[1], rgb[2])

                if accepted % 2 == 0 and len(pts) >= 4:
                    q_pts = self._poly_to_smooth_q_bezier(pts.astype(np.float32))
                    shapes.append({
                        "id": f"s3_q_{step}",
                        "step": step,
                        "stage": 3,
                        "type": "path",
                        "kind": "Q",
                        "points": q_pts,
                        "color": hex_col,
                        "fill": hex_col,
                        "opacity": alpha,
                        "error_reduction": round((old_err - new_err) / 50.0, 2),
                    })
                else:
                    shapes.append({
                        "id": f"s3_p_{step}",
                        "step": step,
                        "stage": 3,
                        "type": "polygon",
                        "points": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in pts],
                        "color": hex_col,
                        "fill": hex_col,
                        "opacity": alpha,
                        "error_reduction": round((old_err - new_err) / 50.0, 2),
                    })
            else:
                stale += 1
                r = 2
                wy0, wy1 = max(0, by - r), min(h, by + r + 1)
                wx0, wx1 = max(0, bx - r), min(w, bx + r + 1)
                work_weights[wy0:wy1, wx0:wx1] *= 0.2
                err_map[wy0:wy1, wx0:wx1] *= 0.2

        return canvas, shapes, step, stop_reason

    LAYER_ORDER = ["background", "form", "shadows", "highlights", "details", "outlines"]

    def _tag_layers(self, shapes: List[Dict[str, Any]], w: int, h: int) -> Dict[str, int]:
        total_area = float(w * h) + 1e-6

        def lum_of(s: Dict[str, Any]) -> float:
            hx = str(s.get("fill", "#808080")).lstrip("#")
            r, g, b = (int(hx[i : i + 2], 16) for i in (0, 2, 4))
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        def bbox_area(s: Dict[str, Any]) -> float:
            pts = np.array(s["points"], dtype=np.float32)
            return float((pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min()))

        body = [s for s in shapes if s["stage"] in (1, 2) and s["id"] != "s0_bg"]
        lums = np.array([lum_of(s) for s in body]) if body else np.array([128.0])
        p_dark = float(np.percentile(lums, 25))
        p_light = float(np.percentile(lums, 80))

        counts = {name: 0 for name in self.LAYER_ORDER}
        for s in shapes:
            if s["id"] == "s0_bg":
                layer = "background"
            elif s["stage"] == 3:
                layer = "outlines" if s.get("kind") == "C" else "details"
            elif s["stage"] == 1 and bbox_area(s) > 0.25 * total_area:
                layer = "background"
            else:
                lum = lum_of(s)
                if lum <= p_dark:
                    layer = "shadows"
                elif lum >= p_light:
                    layer = "highlights"
                else:
                    layer = "form"
            s["layer"] = layer
            counts[layer] += 1
        return counts

    # ------------------------------------------------------------------
    # CLEAN MODE (vtracer-style): hierarchical region merging + stacked,
    # gap-free shapes + smooth cubic Bezier outlines.
    # ------------------------------------------------------------------
    @staticmethod
    def _smooth_closed(pts: np.ndarray, sigma: float) -> np.ndarray:
        """Gaussian-smooth a closed polyline (removes pixel staircase)."""
        n = len(pts)
        if n < 6 or sigma < 0.3:
            return pts
        r = max(1, min(int(math.ceil(3.0 * sigma)), n // 2))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
        k /= k.sum()
        ext = np.concatenate([pts[-r:], pts, pts[:r]], axis=0)
        out = np.empty_like(pts)
        for d in range(2):
            out[:, d] = np.convolve(ext[:, d], k, mode="valid")
        return out

    @staticmethod
    def _fit_closed_cubic(poly: np.ndarray, corner_cos: float = -0.34) -> List[List[float]]:
        """
        Closed polygon -> smooth cubic Bezier path with corner detection.
        Returns [P0, c1, c2, P1, c1, c2, P2, ..., c1, c2, P0] (SVG 'C' triplets).
        Sharp vertices stay sharp, everything else gets a tangent-continuous curve.
        """
        n = len(poly)
        out_h = np.zeros_like(poly)
        in_h = np.zeros_like(poly)
        for i in range(n):
            p = poly[i]
            vp = poly[i - 1] - p
            vn = poly[(i + 1) % n] - p
            lp = math.hypot(vp[0], vp[1]) + 1e-9
            ln = math.hypot(vn[0], vn[1]) + 1e-9
            cosang = float(np.dot(vp, vn) / (lp * ln))
            if n <= 3 or cosang > corner_cos:
                out_h[i] = p + vn / 3.0
                in_h[i] = p + vp / 3.0
            else:
                t = vn / ln - vp / lp
                t = t / (math.hypot(t[0], t[1]) + 1e-9)
                hl = min(lp, ln) / 3.0  # same length both sides: long edges stay straight
                out_h[i] = p + t * hl
                in_h[i] = p - t * hl
        pts = [[round(float(poly[0][0]), 2), round(float(poly[0][1]), 2)]]
        for i in range(n):
            j = (i + 1) % n
            for q in (out_h[i], in_h[j], poly[j]):
                pts.append([round(float(q[0]), 2), round(float(q[1]), 2)])
        return pts

    @staticmethod
    def _flatten_cubic(pts: List[List[float]], samples: int = 6) -> np.ndarray:
        P = np.asarray(pts, dtype=np.float32)
        t = np.linspace(0.0, 1.0, samples, endpoint=False, dtype=np.float32)[:, None]
        out = []
        for s in range((len(P) - 1) // 3):
            p0, c1, c2, p1 = P[3 * s], P[3 * s + 1], P[3 * s + 2], P[3 * s + 3]
            out.append(((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * c1 + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * p1)
        return np.concatenate(out, axis=0)

    def _ward_merge_regions(
        self,
        cmap: np.ndarray,
        lab_sum: np.ndarray,
        rgb_sum: np.ndarray,
        area: np.ndarray,
        w_sum: np.ndarray,
        bbox: np.ndarray,
        target: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Greedy adjacency-constrained Ward merging (same idea as vtracer's hierarchical
        clustering). Cheapest merge first: tiny / similar-colour regions vanish,
        high-contrast and high-saliency regions (eyes, lips, glow) survive.
        Returns (final_label_map, ids, area, rgb_mean, bbox) for the surviving regions.
        """
        import heapq

        n = len(area)
        if n <= target:
            return cmap, np.arange(n), area, rgb_sum / np.maximum(area[:, None], 1), bbox

        # adjacency with border lengths (4-neighbourhood, fully vectorised)
        a = np.concatenate([cmap[:, :-1].ravel(), cmap[:-1, :].ravel()])
        b = np.concatenate([cmap[:, 1:].ravel(), cmap[1:, :].ravel()])
        m = a != b
        lo = np.minimum(a[m], b[m]).astype(np.int64)
        hi = np.maximum(a[m], b[m]).astype(np.int64)
        keys, cnts = np.unique(lo * n + hi, return_counts=True)
        adj: List[Dict[int, int]] = [dict() for _ in range(n)]
        for key, c in zip(keys.tolist(), cnts.tolist()):
            i, j = divmod(key, n)
            adj[i][j] = c
            adj[j][i] = c

        A = area.astype(np.float64).tolist()
        L0, L1, L2 = (lab_sum[:, 0].tolist(), lab_sum[:, 1].tolist(), lab_sum[:, 2].tolist())
        W = w_sum.astype(np.float64).tolist()
        R = rgb_sum.astype(np.float64).copy()
        bb = bbox.copy()
        alive = [True] * n
        ver = [0] * n
        root = list(range(n))

        def cost(i: int, j: int) -> float:
            ai, aj = A[i], A[j]
            d0 = L0[i] / ai - L0[j] / aj
            d1 = L1[i] / ai - L1[j] / aj
            d2 = L2[i] / ai - L2[j] / aj
            ward = (ai * aj / (ai + aj)) * (d0 * d0 + d1 * d1 + d2 * d2)
            mw = (W[i] + W[j]) / (ai + aj)  # mean saliency weight, 1..9
            return ward * (mw ** 0.8) * (1.0 + 6.0 / (adj[i][j] + 1.0))

        heap: List[Tuple[float, int, int, int, int]] = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    heap.append((cost(i, j), i, j, 0, 0))
        heapq.heapify(heap)

        alive_n = n
        while alive_n > target and heap:
            _, i, j, vi, vj = heapq.heappop(heap)
            if not (alive[i] and alive[j]) or ver[i] != vi or ver[j] != vj:
                continue
            if A[i] < A[j]:
                i, j = j, i  # i survives
            A[i] += A[j]
            L0[i] += L0[j]
            L1[i] += L1[j]
            L2[i] += L2[j]
            W[i] += W[j]
            R[i] += R[j]
            bb[i, 0] = min(bb[i, 0], bb[j, 0])
            bb[i, 1] = min(bb[i, 1], bb[j, 1])
            bb[i, 2] = max(bb[i, 2], bb[j, 2])
            bb[i, 3] = max(bb[i, 3], bb[j, 3])
            alive[j] = False
            root[j] = i
            alive_n -= 1
            ver[i] += 1
            adj[i].pop(j, None)
            for k, bl in adj[j].items():
                if k == i:
                    continue
                adj[k].pop(j, None)
                adj[i][k] = adj[i].get(k, 0) + bl
                adj[k][i] = adj[i][k]
            adj[j] = {}
            for k in adj[i]:
                heapq.heappush(heap, (cost(i, k), i, k, ver[i], ver[k]))

        root_arr = np.asarray(root, dtype=np.int64)
        for _ in range(64):
            nxt = root_arr[root_arr]
            if np.array_equal(nxt, root_arr):
                break
            root_arr = nxt
        final = root_arr[cmap]
        ids = np.nonzero(np.asarray(alive))[0]
        area_f = np.asarray(A)[ids]
        rgb_f = R[ids] / np.maximum(area_f[:, None], 1.0)
        return final, ids, area_f, rgb_f, bb[ids]

    def reconstruct_clean_flat(
        self,
        img_rgb: np.ndarray,
        weight_map: np.ndarray = None,
        face_mask: np.ndarray = None,
        target_max_shapes: int = 95,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Clean, designer-friendly vectorizer (vtracer-style).
          1. Edge-preserving flatten + saliency-weighted k-means in Lab.
          2. Label voting (Gaussian) -> smooth region borders, no speckle.
          3. Connected regions merged hierarchically (Ward + saliency) to the target count.
          4. Each region traced once as a *stacked* shape (big first, small on top):
             no holes, no gaps, no overlapping duplicates.
          5. Outlines smoothed, simplified and fitted with cubic Beziers (corners kept).
        """
        h, w = img_rgb.shape[:2]
        rs = max(h, w) / self.BASE_DIM
        if weight_map is None:
            weight_map, _, face_mask, _ = self.compute_importance_map(img_rgb)
        weight_map = weight_map.astype(np.float32)
        target = max(6, int(target_max_shapes) - 1)  # one slot reserved for the background

        # 1. flatten + quantise ------------------------------------------------
        sm = cv2.bilateralFilter(img_rgb, 9, 50, 7)
        sm = cv2.bilateralFilter(sm, 9, 40, 7)
        lab = cv2.cvtColor(sm.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        pix = lab.reshape(-1, 3)
        rng = np.random.default_rng(7)
        p = weight_map.ravel().astype(np.float64)
        p /= p.sum()
        sample = pix[rng.choice(len(pix), size=min(40000, len(pix)), replace=True, p=p)]
        K = 24 if target >= 40 else 16
        try:
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5)
            _, _, centers = cv2.kmeans(sample.astype(np.float32), K, None, crit, 1, cv2.KMEANS_PP_CENTERS)
        except Exception:
            centers = sample[rng.choice(len(sample), size=K, replace=False)].astype(np.float32)
        best_d = np.full(len(pix), np.inf, dtype=np.float32)
        labels = np.zeros(len(pix), dtype=np.int32)
        for k in range(len(centers)):
            d = ((pix - centers[k]) ** 2).sum(axis=1)
            upd = d < best_d
            best_d[upd] = d[upd]
            labels[upd] = k
        labels = labels.reshape(h, w)

        # 2. label voting -> smooth borders ------------------------------------
        sigma_v = max(1.0, 0.65 * rs)
        best = np.full((h, w), -1.0, dtype=np.float32)
        lab_s = np.zeros((h, w), dtype=np.int32)
        for k in range(len(centers)):
            mk = labels == k
            if not mk.any():
                continue
            bl = cv2.GaussianBlur(mk.astype(np.float32), (0, 0), sigma_v)
            upd = bl > best
            best[upd] = bl[upd]
            lab_s[upd] = k

        # 3. connected regions -> Ward merge -----------------------------------
        cmap = np.zeros((h, w), dtype=np.int32)
        boxes: List[np.ndarray] = []
        n = 0
        for k in range(len(centers)):
            mk = (lab_s == k).astype(np.uint8)
            if not mk.any():
                continue
            cnt, cl, st, _ = cv2.connectedComponentsWithStats(mk, connectivity=4)
            if cnt <= 1:
                continue
            sel = cl > 0
            cmap[sel] = cl[sel] - 1 + n
            bx = st[1:, :4].astype(np.int64)
            boxes.append(np.stack([bx[:, 0], bx[:, 1], bx[:, 0] + bx[:, 2], bx[:, 1] + bx[:, 3]], axis=1))
            n += cnt - 1
        bbox = np.concatenate(boxes, axis=0)
        flat = cmap.ravel()
        area = np.bincount(flat, minlength=n).astype(np.float64)
        lab_sum = np.stack([np.bincount(flat, weights=lab[..., c].ravel(), minlength=n) for c in range(3)], axis=1)
        rgb_sum = np.stack(
            [np.bincount(flat, weights=img_rgb[..., c].ravel().astype(np.float64), minlength=n) for c in range(3)],
            axis=1,
        )
        w_sum = np.bincount(flat, weights=weight_map.ravel().astype(np.float64), minlength=n)
        final, ids, r_area, r_rgb, r_box = self._ward_merge_regions(
            cmap, lab_sum, rgb_sum, area, w_sum, bbox, target
        )

        # 4/5. trace every region once, stacked, smooth ------------------------
        P = 3
        padded = np.pad(final, P, mode="edge")
        H2, W2 = padded.shape
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        items = []
        for idx, rid in enumerate(ids.tolist()):
            x0, y0, x1, y1 = [int(v) for v in r_box[idx]]
            cx0, cy0 = max(0, x0 - 2), max(0, y0 - 2)
            cx1, cy1 = min(W2, x1 + 2 * P + 2), min(H2, y1 + 2 * P + 2)
            sub = (padded[cy0:cy1, cx0:cx1] == rid).astype(np.uint8) * 255
            if not sub.any():
                continue
            sub = cv2.dilate(sub, kern)  # ~0.5px overlap hides anti-alias seams
            sub = cv2.copyMakeBorder(sub, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                continue
            cnt = max(cnts, key=len)
            pts = cnt.reshape(-1, 2).astype(np.float32)
            pts[:, 0] += cx0 - 1 - P + 0.5
            pts[:, 1] += cy0 - 1 - P + 0.5
            ar0 = max(float(cv2.contourArea(cnt)), float(len(pts)))
            sig = float(np.clip(0.03 * math.sqrt(ar0), 0.8, 2.2))
            pts = self._smooth_closed(pts, sig)
            pts[:, 0] = np.clip(pts[:, 0], 0, w)
            pts[:, 1] = np.clip(pts[:, 1], 0, h)
            eps = float(np.clip(0.05 * math.sqrt(ar0), 0.5 * rs, 1.1 * rs))
            approx = cv2.approxPolyDP(pts.reshape(-1, 1, 2), eps, True).reshape(-1, 2)
            if len(approx) < 3:
                continue
            ar = abs(float(cv2.contourArea(approx.reshape(-1, 1, 2))))
            if ar < 2.0:
                continue
            items.append({"area": ar, "poly": approx.astype(np.float32), "rgb": r_rgb[idx],
                          "n": float(r_area[idx])})

        if not items:
            mean_rgb = img_rgb.reshape(-1, 3).mean(axis=0)
            hx = self._rgb_to_hex(*mean_rgb)
            bg = {"id": "c0_bg", "step": 0, "stage": 1, "layer": "background", "type": "polygon",
                  "points": [[0, 0], [w, 0], [w, h], [0, h]], "color": hx, "fill": hx, "opacity": 1.0}
            return [bg], np.full_like(img_rgb, mean_rgb.astype(np.uint8))

        items.sort(key=lambda d: d["area"], reverse=True)
        bg_rgb = items[0]["rgb"]
        bg_hex = self._rgb_to_hex(*bg_rgb)
        if items[0]["area"] >= 0.97 * w * h:
            items = items[1:]  # the background rect already covers it

        clean_shapes: List[Dict[str, Any]] = [{
            "id": "c0_bg", "step": 0, "stage": 1, "layer": "background", "type": "polygon",
            "points": [[0, 0], [w, 0], [w, h], [0, h]],
            "color": bg_hex, "fill": bg_hex, "opacity": 1.0,
        }]

        def lum(rgb: np.ndarray) -> float:
            return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

        lums = np.array([lum(it["rgb"]) for it in items]) if items else np.array([128.0])
        p_dark, p_light = float(np.percentile(lums, 25)), float(np.percentile(lums, 80))
        total = float(w * h)
        n_items = max(1, len(items))

        canvas = np.empty((h, w, 3), dtype=np.uint8)
        canvas[:] = np.clip(np.round(bg_rgb), 0, 255).astype(np.uint8)

        for order, it in enumerate(items, start=1):
            rgb = it["rgb"]
            hex_c = self._rgb_to_hex(*rgb)
            frac = order / n_items
            stage = 1 if frac <= 0.25 else (2 if frac <= 0.70 else 3)
            l = lum(rgb)
            if it["area"] > 0.25 * total:
                layer = "background"
            elif stage == 3:
                layer = "details"
            elif l <= p_dark:
                layer = "shadows"
            elif l >= p_light:
                layer = "highlights"
            else:
                layer = "form"
            cpts = self._fit_closed_cubic(it["poly"])
            clean_shapes.append({
                "id": f"c_{order}", "step": order, "stage": stage, "layer": layer,
                "type": "path", "kind": "C", "points": cpts,
                "color": hex_c, "fill": hex_c, "opacity": 1.0,
            })
            flat_pts = (self._flatten_cubic(cpts) - 0.5) * 16.0  # px centre convention, 4 fractional bits
            cv2.fillPoly(canvas, [np.round(flat_pts).astype(np.int32)],
                         tuple(int(round(float(c))) for c in rgb), lineType=cv2.LINE_AA, shift=4)

        return clean_shapes, canvas

    def reconstruct(
        self, image_input: Any, max_shapes: int = 800, detail_mode: str = "balanced",
        clean_budget: int = 95,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()

        if isinstance(image_input, bytes):
            img_rgb = None
            nparr = np.frombuffer(image_input, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is not None and img_bgr.size > 0:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            else:
                import io
                pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
                img_rgb = np.array(pil_img)
        elif hasattr(image_input, "convert"):
            img_rgb = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            img_rgb = (
                cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
                if image_input.shape[2] == 4
                else image_input.copy()
            )
        else:
            raise TypeError("Unsupported image input type.")

        orig_h, orig_w = img_rgb.shape[:2]
        scale = min(self.MAX_DIM / max(orig_w, orig_h), self.MAX_UPSCALE)
        new_w, new_h = max(8, int(round(orig_w * scale))), max(8, int(round(orig_h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        target_u8 = cv2.resize(img_rgb, (new_w, new_h), interpolation=interp)
        rs = max(new_w, new_h) / self.BASE_DIM
        a2 = rs * rs
        target_f = target_u8.astype(np.float32)

        weight_map, saliency_b64, face_mask, _ = self.compute_importance_map(target_u8)

        corners = np.vstack([
            target_f[:10, :10].reshape(-1, 3),
            target_f[:10, -10:].reshape(-1, 3),
            target_f[-10:, :10].reshape(-1, 3),
            target_f[-10:, -10:].reshape(-1, 3),
        ])
        dom_rgb = np.median(corners, axis=0).astype(np.float32)
        canvas = np.full_like(target_f, dom_rgb)

        shapes: List[Dict[str, Any]] = [
            {
                "id": "s0_bg",
                "step": 0,
                "stage": 1,
                "type": "rect",
                "points": [[0, 0], [new_w, 0], [new_w, new_h], [0, new_h]],
                "color": self._rgb_to_hex(dom_rgb[0], dom_rgb[1], dom_rgb[2]),
                "fill": self._rgb_to_hex(dom_rgb[0], dom_rgb[1], dom_rgb[2]),
                "opacity": 1.0,
                "error_reduction": 0.0,
            }
        ]

        convergence_curve: List[Dict[str, float]] = []

        def snapshot(cur_step: int, cur_canv: np.ndarray) -> None:
            mse = float(np.mean((target_f - np.clip(cur_canv, 0, 255)) ** 2))
            psnr = float(10.0 * math.log10(255.0 ** 2 / (mse + 1e-10)))
            convergence_curve.append(
                {"step": cur_step, "mse": round(mse, 3), "psnr": round(psnr, 2)}
            )

        snapshot(0, canvas)
        step = 0

        def bf(d: int, sc: int) -> np.ndarray:
            dd = max(d, int(round(d * math.sqrt(rs))) | 1)
            return cv2.bilateralFilter(target_u8, dd, sc, sc)

        smooth_coarse = bf(9, 60)
        smooth_mid = bf(7, 35)
        smooth_fine = bf(5, 20)

        total_area = float(new_w * new_h)

        b_s1 = max(30, int(max_shapes * 0.10))
        canvas, s1_shapes, step = self._extract_painterly_pass(
            img_smooth=smooth_coarse, target_f=target_f, canvas=canvas, weight_map=weight_map,
            div=42, min_area=40.0 * a2, max_area=total_area * 0.85, eps_factor=0.007,
            alpha=0.94, stage_id=1, use_q_curves=False, max_count=b_s1, start_step=step,
            max_colors=64,
        )
        shapes.extend(s1_shapes)
        snapshot(step, canvas)

        b_s2 = int(max_shapes * 0.30)
        canvas, s2_shapes, step = self._extract_painterly_pass(
            img_smooth=smooth_mid, target_f=target_f, canvas=canvas, weight_map=weight_map,
            div=24, min_area=10.0 * a2, max_area=total_area * 0.30, eps_factor=0.005,
            alpha=0.88, stage_id=2, use_q_curves=True, max_count=b_s2, start_step=step,
            max_colors=96,
        )
        shapes.extend(s2_shapes)
        snapshot(step, canvas)

        b_s2b = int(max_shapes * 0.20)
        canvas, s2b_shapes, step = self._extract_painterly_pass(
            img_smooth=smooth_mid, target_f=target_f, canvas=canvas, weight_map=weight_map,
            div=16, min_area=5.0 * a2, max_area=total_area * 0.08, eps_factor=0.004,
            alpha=0.86, stage_id=2, use_q_curves=True, max_count=b_s2b, start_step=step,
            max_colors=128,
        )
        shapes.extend(s2b_shapes)
        snapshot(step, canvas)

        b_s3_contour = int(max_shapes * 0.15)
        roi_u8 = (face_mask > 0).astype(np.uint8) * 255
        canvas, s3a_shapes, step = self._extract_painterly_pass(
            img_smooth=smooth_fine, target_f=target_f, canvas=canvas, weight_map=weight_map,
            div=14, min_area=2.0 * a2, max_area=350.0 * a2, eps_factor=0.004,
            alpha=0.90, stage_id=3, use_q_curves=True, max_count=b_s3_contour, start_step=step,
            roi_mask=roi_u8, max_colors=96,
        )
        shapes.extend(s3a_shapes)
        snapshot(step, canvas)

        rem_budget = max(0, max_shapes - (len(shapes) - 1))
        canvas, s3b_shapes, step, stop_reason = self._stage3_fine_features_and_outlines(
            target_f=target_f, canvas=canvas, weight_map=weight_map,
            n_remaining=rem_budget, start_step=step, rs=rs,
        )
        shapes.extend(s3b_shapes)
        snapshot(step, canvas)

        final_u8 = np.clip(canvas, 0, 255).astype(np.uint8)
        final_mse = float(np.mean((target_f - np.clip(canvas, 0, 255)) ** 2))
        final_psnr = float(10.0 * math.log10(255.0 ** 2 / (final_mse + 1e-10)))

        try:
            final_ssim = float(ssim_fn(target_u8, final_u8, channel_axis=2))
        except Exception:
            try:
                final_ssim = float(ssim_fn(
                    cv2.cvtColor(target_u8, cv2.COLOR_RGB2GRAY),
                    cv2.cvtColor(final_u8, cv2.COLOR_RGB2GRAY),
                ))
            except Exception:
                final_ssim = 0.0

        diff_norm = np.clip(
            np.mean(np.abs(target_f - np.clip(canvas, 0, 255)), axis=2) / 64.0 * 255.0,
            0,
            255,
        ).astype(np.uint8)
        err_b64 = self._array_to_png_b64(cv2.applyColorMap(diff_norm, cv2.COLORMAP_MAGMA))
        rec_b64 = self._array_to_png_b64(cv2.cvtColor(final_u8, cv2.COLOR_RGB2BGR))
        orig_b64 = self._array_to_png_b64(cv2.cvtColor(target_u8, cv2.COLOR_RGB2BGR))

        layer_summary = self._tag_layers(shapes, new_w, new_h)
        # 4. Generate Clean Flat Vector Output (Mode 2 for designer editability!)
        clean_shapes, clean_raster = self.reconstruct_clean_flat(
            target_u8, weight_map=weight_map, face_mask=face_mask, target_max_shapes=clean_budget
        )
        clean_b64 = self._array_to_png_b64(cv2.cvtColor(clean_raster, cv2.COLOR_RGB2BGR))

        try:
            clean_ssim = float(ssim_fn(target_u8, clean_raster, channel_axis=2))
        except Exception:
            clean_ssim = 0.82
        clean_mse = float(np.mean((target_f - clean_raster.astype(np.float32)) ** 2))
        clean_psnr = float(10.0 * math.log10(255.0 ** 2 / (clean_mse + 1e-10)))

        runtime_val = round(time.perf_counter() - start_time, 3)

        return {
            "ai_model": self.model_name,
            "width": new_w,
            "height": new_h,
            "viewBox": {"width": new_w, "height": new_h},
            "saliency_map_b64": saliency_b64,
            "saliency_heatmap_b64": saliency_b64,
            "error_map_b64": err_b64,
            "error_heatmap_b64": err_b64,
            "reconstructed_image_b64": rec_b64,
            "original_image_b64": orig_b64,
            "clean_reconstructed_b64": clean_b64,
            "clean_shapes": clean_shapes,
            "clean_metrics": {
                "total_shapes": len(clean_shapes),
                "psnr": round(clean_psnr, 2),
                "ssim": round(clean_ssim, 4),
                "mse": round(clean_mse, 2)
            },
            "metrics": {
                "mse": round(final_mse, 3),
                "psnr": round(final_psnr, 2),
                "ssim": round(final_ssim, 4),
                "runtime_sec": runtime_val,
                "runtime_s": runtime_val,
                "total_shapes": len(shapes),
                "shape_count": len(shapes),
                "requested_shapes": max_shapes,
                "stop_reason": stop_reason,
                "stop_reason_text": {
                    "budget_reached": "Used the full shape budget",
                    "converged": "Converged: no visible error left",
                    "stagnated": "Stopped: new shapes no longer improved the result",
                    "attempt_limit": "Stopped: attempt limit reached",
                }.get(stop_reason, stop_reason),
                "resolution": [new_w, new_h],
            },
            "visualizations": {
                "saliency_heatmap_b64": saliency_b64,
                "saliency_map_b64": saliency_b64,
                "error_heatmap_b64": err_b64,
                "error_map_b64": err_b64,
                "reconstructed_image_b64": rec_b64,
                "original_image_b64": orig_b64,
                "clean_reconstructed_b64": clean_b64,
            },
            "convergence_curve": convergence_curve,
            "layer_summary": layer_summary,
            "shapes": shapes,
        }
