from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
from collections import deque
import math

# =======================
# CONFIG
# =======================
VIDEO_IN   = r"video path"
VIDEO_OUT  = r"output"
MODEL_PATH = r"c:\Users\PC\Documents\parking\model1\weights\last.pt"

CONF_THRESH = 0.65
IOU_NMS1    = 0.45
IOU_NMS2    = 0.35

USE_BILATERAL = False
GAUSS_KSIZE   = 5
GAUSS_SIGMA   = 0


PANEL_W      = 360
PANEL_BG     = (0, 0, 0)
FG_COLOR     = (255, 255, 255)
SEPARATOR    = (200, 200, 200)

FONT         = cv2.FONT_HERSHEY_SIMPLEX
TITLE_SCALE  = 0.8
TEXT_SCALE   = 0.6
SMALL_SCALE  = 0.5
THICK        = 1


PAD          = 16
SEC_GAP      = 14
LINE_GAP     = 10
MINIMAP_H    = 180


TABLE_ROW_H          = 18
TABLE_COL_MIN_W      = 130
TABLE_PAGE_SECONDS   = 2.0
TABLE_MAX_COLS       = 3


GRID_ROWS            = 5
GRID_COLS            = 10
TRAIL_LEN            = 12
HEATMAP_W, HEATMAP_H = 160, 100
HEAT_DECAY           = 0.92
HEAT_DOT             = 0.75
HEAT_BLUR            = (7, 7)

# --- 3D plane styling ---
PERSPECTIVE_TOP_INSET = 0.22
PLANE_MARGIN          = 8
PLANE_SHADOW_OFFSET   = (8, 6)
PLANE_SHADOW_ALPHA    = 0.35
PILLAR_MAX_H          = 18
PILLAR_BASE_R         = 2
PILLAR_SIDE_DARKEN    = 0.45


COLOR_CAR  = (0, 0, 255)
COLOR_FREE = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
COLOR_DETR = (255, 255, 0)  

# ========= Anti-flicker knobs =========
IOU_MATCH         = 0.4
START_CONF        = 0.50
KEEP_CONF         = 0.30
SMOOTH_POS_ALPHA  = 0.45
SMOOTH_SIZE_ALPHA = 0.35
MIN_CONFIRM_FR    = 3
COAST_FRAMES_S    = 0.6
FADE_WHEN_COAST   = True

# ---------- ROI drawing config ----------
ROI_COLOR = (0, 255, 255)   
ROI_THICK = 3
ROI_MIN_SIZE = 12

# ---------- Aerial YOLO (replaces DETR for ROI vehicle detection) ----------
AERIAL_MODEL_PATH = r"c:\Users\PC\Documents\multi-vehicles\Exp_Sample\weights\last.pt" 
AERIAL_ACCEPT_IDS = {0, 1, 2, 3}  
AERIAL_CONF_THR = 0.60
AERIAL_IOU_THR  = 0.45

MIN_ROI_COVER   = 0.20

# =======================
# HELPERS
# =======================
def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter + 1e-6
    return inter / union

def nms_extra(dets, iou_thr=0.35):
    if not dets: return dets
    dets = sorted(dets, key=lambda d: d["conf"], reverse=True)
    keep = []
    for d in dets:
        if all(iou_xyxy(d["xyxy"], k["xyxy"]) < iou_thr for k in keep):
            keep.append(d)
    return keep

def xyxy_to_cxcywh(b):
    x1,y1,x2,y2 = b
    w = max(1, x2 - x1); h = max(1, y2 - y1)
    cx = x1 + w/2; cy = y1 + h/2
    return np.array([cx,cy,w,h], dtype=np.float32)

def draw_box_with_fill(img, x1, y1, x2, y2, color, alpha=0.28, border=2):
    x1, y1 = int(x1), int(y1)
    x2, y2 = int(x2), int(y2)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, border)

def cxcywh_to_xyxy(v):
    cx,cy,w,h = v
    x1 = int(round(cx - w/2)); y1 = int(round(cy - h/2))
    x2 = int(round(cx + w/2)); y2 = int(round(cy + h/2))
    return np.array([x1,y1,x2,y2], dtype=np.int32)

def lerp(a,b,t): return a*(1-t)+b*t

def draw_centered_label_in_box(img, x1, y1, x2, y2, text, color):
    pad = 4
    box_w = max(1, x2 - x1); box_h = max(1, y2 - y1)
    thickness = 1; base_scale = 0.6
    (tw1, th1), _ = cv2.getTextSize(text, FONT, 1.0, thickness)
    scale = min(base_scale, (box_w - 2*pad)/max(1, tw1), (box_h - 2*pad)/max(1, th1))
    scale = max(0.35, min(scale, 1.0))
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
    bg_x1 = max(x1, cx - tw // 2 - pad); bg_y1 = max(y1, cy - th // 2 - pad)
    bg_x2 = min(x2, cx + tw // 2 + pad); bg_y2 = min(y2, cy + th // 2 + pad)
    overlay = img.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    text_x = max(x1 + pad, cx - tw // 2); text_x = min(text_x, x2 - pad - tw)
    text_y = min(bg_y2 - pad, max(bg_y1 + pad + th, cy + th // 2))
    cv2.putText(img, text, (text_x, text_y), FONT, scale, TEXT_COLOR, thickness, cv2.LINE_AA)

# ------- geometry + dotted line -------
def draw_dotted_line(img, p1, p2, color=(255,255,255), gap=7, radius=2):
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    dist = np.linalg.norm(p2 - p1)
    if dist < 1: return
    steps = max(1, int(dist // gap))
    t = np.linspace(0, 1, steps)
    for u in t:
        p = (p1*(1-u) + p2*u).astype(int)
        cv2.circle(img, tuple(p), radius, color, -1, lineType=cv2.LINE_AA)

def box_center(xyxy):
    x1,y1,x2,y2 = xyxy
    return (int((x1+x2)/2), int((y1+y2)/2))

def rect_intersection_area(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    return max(0, x2-x1) * max(0, y2-y1)

def frac_inside_any_roi(box, rois):
    if not rois: return 0.0
    bx1,by1,bx2,by2 = box
    b_area = max(1, (bx2-bx1)*(by2-by1))
    inter = 0
    for rx1,ry1,rx2,ry2 in rois:
        inter += rect_intersection_area((bx1,by1,bx2,by2), (rx1,ry1,rx2,ry2))
    return inter / float(b_area)

def point_in_rect(pt, rect):
    x,y = pt
    x1,y1,x2,y2 = rect
    return (x1 <= x <= x2) and (y1 <= y <= y2)

def center_inside_any_roi(box, rois):
    cx, cy = box_center(box)
    return any(point_in_rect((cx, cy), r) for r in rois) if rois else False

# --------- ROI helpers ----------
class ROISelector:
    def __init__(self, first_frame):
        self.orig = first_frame.copy()
        self.view = first_frame.copy()
        self.drawing = False
        self.start_pt = None
        self.rois = []   

    def _mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.view = self.orig.copy()
            for (x1,y1,x2,y2) in self.rois:
                cv2.rectangle(self.view, (x1,y1), (x2,y2), ROI_COLOR, ROI_THICK)
            cv2.rectangle(self.view, self.start_pt, (x, y), ROI_COLOR, ROI_THICK)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            x1, y1 = self.start_pt
            x2, y2 = x, y
            if abs(x2-x1) >= ROI_MIN_SIZE and abs(y2-y1) >= ROI_MIN_SIZE:
                x1, x2 = sorted([x1, x2]); y1, y2 = sorted([y1, y2])
                self.rois.append((x1, y1, x2, y2))
            self.view = self.orig.copy()
            for (rx1,ry1,rx2,ry2) in self.rois:
                cv2.rectangle(self.view, (rx1,ry1), (rx2,ry2), ROI_COLOR, ROI_THICK)

def select_rois_interactive(frame):
    sel = ROISelector(frame)
    win = "Draw ROIs: LMB=add | Z=undo | R=reset | Enter=done | Q/ESC=cancel"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, sel._mouse)
    while True:
        view = sel.view.copy()
        cv2.rectangle(view, (0,0), (view.shape[1], 26), (0,0,0), -1)
        cv2.putText(view, "LMB-drag: add  |  Z: undo  R: reset  Enter: done  Q/ESC: cancel",
                    (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.imshow(win, view)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10):   # Enter
            break
        elif k in (ord('q'), 27):  # q or ESC
            sel.rois = []
            break
        elif k == ord('z') and sel.rois:
            sel.rois.pop()
        elif k == ord('r'):
            sel.rois = []
    cv2.destroyWindow(win)
    return sel.rois

# --------- Anti-flicker tracker ----------
class SimpleTracker:
    def __init__(self, fps, iou_match=IOU_MATCH):
        self.next_id = 1
        self.tracks = {}
        self.iou_match = iou_match
        self.min_confirm = MIN_CONFIRM_FR
        self.coast_max = max(1, int(COAST_FRAMES_S * fps))

    def _match(self, dets):
        assigned = set()
        id_updates = {}
        dets_sorted = sorted(enumerate(dets), key=lambda x: x[1]["conf"], reverse=True)
        for di, d in dets_sorted:
            best_iou, best_id = 0.0, None
            for tid, tr in self.tracks.items():
                # only match same class
                if tr["cls_name"] != d["name"]:
                    continue
                iou = iou_xyxy(tr["xyxy_smooth"], d["xyxy"])
                if iou > best_iou and iou >= self.iou_match and tid not in assigned:
                    best_iou, best_id = iou, tid
            if best_id is not None:
                id_updates[di] = best_id
                assigned.add(best_id)
        return id_updates

    def _smooth_update(self, tr, new_xyxy):
        prev = tr["cxcywh"]
        cur  = xyxy_to_cxcywh(new_xyxy).astype(np.float32)
        a_pos = SMOOTH_POS_ALPHA
        a_sz  = SMOOTH_SIZE_ALPHA
        smoothed = np.array([
            lerp(prev[0], cur[0], a_pos),
            lerp(prev[1], cur[1], a_pos),
            lerp(prev[2], cur[2], a_sz),
            lerp(prev[3], cur[3], a_sz),
        ], dtype=np.float32)
        tr["cxcywh"] = smoothed
        tr["xyxy_smooth"] = cxcywh_to_xyxy(smoothed)

    def step(self, dets, frame_time):
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["last_seen"] += 1

        keep = []
        for d in dets:
            if d["conf"] >= KEEP_CONF or d["conf"] >= START_CONF:
                keep.append(d)

        mapping = self._match(keep)
        out = []
        for i, d in enumerate(keep):
            if i in mapping:
                tid = mapping[i]
                tr = self.tracks[tid]
                tr["xyxy_raw"] = d["xyxy"]
                tr["last_seen"] = 0
                tr["hit_count"] += 1
                tr["seconds"] += frame_time
                self._smooth_update(tr, d["xyxy"])
            else:
                if d["conf"] < START_CONF:
                    continue
                tid = self.next_id; self.next_id += 1
                cxcywh = xyxy_to_cxcywh(d["xyxy"])
                self.tracks[tid] = {
                    "xyxy_raw": d["xyxy"],
                    "xyxy_smooth": d["xyxy"],
                    "cxcywh": cxcywh.astype(np.float32),
                    "last_seen": 0,
                    "hit_count": 1,
                    "seconds": frame_time,
                    "cls_name": d["name"],
                    "hist": deque(maxlen=TRAIL_LEN)
                }
            tr = self.tracks[tid]
            sm = tr["xyxy_smooth"]
            cx = 0.5 * (sm[0] + sm[2]); cy = 0.5 * (sm[1] + sm[3])
            tr["hist"].append((cx, cy))

            d["track_id"] = tid
            d["xyxy"] = tr["xyxy_smooth"]
            out.append(d)

        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["last_seen"] > self.coast_max:
                del self.tracks[tid]

        return out

    def times_summary(self):
        items = [(tid, tr["seconds"]) for tid, tr in self.tracks.items()]
        items.sort(key=lambda x: x[0])
        return items

    def render_tracks(self, frame, color_by_name):
        for tid, tr in self.tracks.items():
            confirmed = tr["hit_count"] >= self.min_confirm
            if not confirmed and tr["last_seen"] == 0:
                continue
            x1,y1,x2,y2 = tr["xyxy_smooth"].tolist()
            color = color_by_name.get(tr["cls_name"], (255,255,255))
            if tr["last_seen"] > 0 and FADE_WHEN_COAST:
                fade = max(0.3, 1.0 - tr["last_seen"] / float(self.coast_max))
                color = (int(color[0]*fade), int(color[1]*fade), int(color[2]*fade))
            draw_box_with_fill(frame, x1, y1, x2, y2, color, alpha=0.17, border=2)
            draw_centered_label_in_box(frame, x1, y1, x2, y2, f"{tr['cls_name']}{tid}", color)

ROI_MARGIN      = 6        
MIN_ROI_COVER   = 0.15     

def expand_rect(r, m):
    x1, y1, x2, y2 = r
    return (x1 - m, y1 - m, x2 + m, y2 + m)

def point_in_any_roi(pt, rois):
    return any(point_in_rect(pt, expand_rect(r, ROI_MARGIN)) for r in rois) if rois else False

def frac_inside_any_roi_expanded(box, rois):
    if not rois: return 0.0
    bx1,by1,bx2,by2 = box
    b_area = max(1, (bx2-bx1)*(by2-by1))
    inter = 0
    for r in rois:
        rx1,ry1,rx2,ry2 = expand_rect(r, ROI_MARGIN)
        inter += rect_intersection_area((bx1,by1,bx2,by2), (rx1,ry1,rx2,ry2))
    return inter / float(b_area)

def box_inside_roi(box, rois, cover_thr=MIN_ROI_COVER):
    """Accept if enough overlap OR center is inside (with a small margin)."""
    if not rois: 
        return False
    cover = frac_inside_any_roi_expanded(box, rois)
    if cover >= cover_thr:
        return True
    
    cx, cy = box_center(box)
    return point_in_any_roi((cx, cy), rois)


def draw_timer_table(panel, origin, size, items, now_s,
                     row_h=TABLE_ROW_H, col_min_w=TABLE_COL_MIN_W,
                     max_cols=TABLE_MAX_COLS, page_seconds=TABLE_PAGE_SECONDS):
    x0, y0 = origin
    w, h   = size
    if h <= 0 or w <= 0:
        return

    HEADER_H = 22
    inner_h  = max(0, h - HEADER_H)
    if inner_h <= 0:
        return

    cv2.putText(panel, "Car timers (s):", (x0, y0 + 16),
                FONT, TEXT_SCALE, FG_COLOR, THICK, cv2.LINE_AA)

    rows_fit = max(1, inner_h // row_h)
    cols_fit = max(1, min(max_cols, w // col_min_w))
    per_page = rows_fit * cols_fit
    if per_page <= 0:
        return

    page = int(now_s // page_seconds)
    total_pages = max(1, math.ceil(len(items) / per_page))
    page %= total_pages
    start = page * per_page
    end   = min(len(items), start + per_page)
    page_items = items[start:end]

    col_w = w // cols_fit
    y_text = y0 + HEADER_H

    for c in range(1, cols_fit):
        x = x0 + c * col_w
        cv2.line(panel, (x, y_text), (x, y_text + rows_fit * row_h),
                 (240, 240, 240), 1, cv2.LINE_AA)

    for idx, (tid, secs) in enumerate(page_items):
        col = idx // rows_fit
        row = idx %  rows_fit
        px  = x0 + col * col_w + 4
        py  = y_text + row * row_h + (row_h - 6)
        cv2.putText(panel, f"car{tid}: {int(secs)}",
                    (px, py), FONT, SMALL_SCALE, FG_COLOR, THICK, cv2.LINE_AA)

    if total_pages > 1:
        badge = f"{page+1}/{total_pages}"
        (tw, th), _ = cv2.getTextSize(badge, FONT, SMALL_SCALE, THICK)
        bx = x0 + w - tw - 4
        by = y0 + h - 6
        cv2.putText(panel, badge, (bx, by),
                    FONT, SMALL_SCALE, (180, 180, 180), THICK, cv2.LINE_AA)

# -------- 3D minimap --------
def draw_minimap_3d(panel, rect_origin, rect_size, frame_size, dets, tracker, heat_buf):
    mm_x, mm_y = rect_origin
    mm_w, mm_h = rect_size
    fw, fh     = frame_size

    tex_w = mm_w - 2*PLANE_MARGIN
    tex_h = mm_h - 2*PLANE_MARGIN
    if tex_w <= 10 or tex_h <= 10:
        return

    heat_buf *= HEAT_DECAY
    centers = []
    for d in dets:
        x1, y1, x2, y2 = d["xyxy"]
        cx = (x1 + x2) * 0.5 / fw
        cy = (y1 + y2) * 0.5 / fh
        centers.append((cx, cy, d["name"]))
        px = int(np.clip(cx * (HEATMAP_W - 1), 0, HEATMAP_W - 1))
        py = int(np.clip(cy * (HEATMAP_H - 1), 0, HEATMAP_H - 1))
        heat_buf[py, px] = np.clip(heat_buf[py, px] + HEAT_DOT, 0.0, 1.0)

    heat_img = (heat_buf * 255).astype(np.uint8)
    heat_img = cv2.GaussianBlur(heat_img, HEAT_BLUR, 0)
    heat_color = cv2.applyColorMap(heat_img, cv2.COLORMAP_HOT)
    heat_color = cv2.resize(heat_color, (tex_w, tex_h), interpolation=cv2.INTER_LINEAR)

    ground = np.zeros((tex_h, tex_w, 3), dtype=np.uint8)
    for i in range(tex_h):
        t = i / max(1, tex_h - 1)
        shade = int(18 + 18 * (1 - t))
        ground[i, :] = (shade, shade, shade)
    cv2.addWeighted(heat_color, 0.35, ground, 0.65, 0, dst=ground)

    for r in range(1, GRID_ROWS):
        y = int(r * tex_h / GRID_ROWS)
        cv2.line(ground, (0, y), (tex_w, y), (75, 75, 75), 1, cv2.LINE_AA)
    for c in range(1, GRID_COLS):
        x = int(c * tex_w / GRID_COLS)
        cv2.line(ground, (x, 0), (x, tex_h), (75, 75, 75), 1, cv2.LINE_AA)

    for tid, tr in tracker.tracks.items():
        if tr["cls_name"] != "car": continue
        pts = tr["hist"]
        for k in range(1, len(pts)):
            x0 = int((pts[k-1][0] / fw) * tex_w)
            y0 = int((pts[k-1][1] / fh) * tex_h)
            x1 = int((pts[k][0]   / fw) * tex_w)
            y1 = int((pts[k][1]   / fh) * tex_h)
            alpha = k / len(pts)
            col = (0, int(120*alpha), 255)
            cv2.line(ground, (x0, y0), (x1, y1), col, 1, cv2.LINE_AA)

    heat_small = cv2.resize(heat_img, (tex_w, tex_h), interpolation=cv2.INTER_LINEAR)
    for cx, cy, name in centers:
        px = int(np.clip(cx * (tex_w - 1), 0, tex_w - 1))
        py = int(np.clip(cy * (tex_h - 1), 0, (tex_h - 1)))
        intensity = heat_small[py, px] / 255.0
        h = int(PILLAR_MAX_H * (0.35 + 0.65 * intensity))
        base_color = COLOR_CAR if name == "car" else COLOR_FREE
        top = max(0, py - h)
        cv2.line(ground, (px, py), (px, top), base_color, 2, cv2.LINE_AA)
        side = (int(base_color[0]*PILLAR_SIDE_DARKEN),
                int(base_color[1]*PILLAR_SIDE_DARKEN),
                int(base_color[2]*PILLAR_SIDE_DARKEN))
        cv2.line(ground, (px+1, py), (px+1, top), side, 1, cv2.LINE_AA)
        cv2.circle(ground, (px, py), PILLAR_BASE_R, base_color, -1, lineType=cv2.LINE_AA)

    inset = int(PERSPECTIVE_TOP_INSET * tex_w)
    dst = np.float32([
        [rect_origin[0] + PLANE_MARGIN + inset,           rect_origin[1] + PLANE_MARGIN],
        [rect_origin[0] + PLANE_MARGIN + tex_w - inset,   rect_origin[1] + PLANE_MARGIN],
        [rect_origin[0] + PLANE_MARGIN + tex_w,           rect_origin[1] + PLANE_MARGIN + tex_h],
        [rect_origin[0] + PLANE_MARGIN,                   rect_origin[1] + PLANE_MARGIN + tex_h],
    ])
    src = np.float32([[0,0],[tex_w-1,0],[tex_w-1,tex_h-1],[0,tex_h-1]])
    H = cv2.getPerspectiveTransform(src, dst)

    shadow = cv2.warpPerspective(ground, H, (panel.shape[1], panel.shape[0]))
    dx, dy = PLANE_SHADOW_OFFSET
    M = np.float32([[1,0,dx],[0,1,dy]])
    shadow = cv2.warpAffine(shadow, M, (panel.shape[1], panel.shape[0]))
    cv2.addWeighted(shadow, PLANE_SHADOW_ALPHA, panel, 1.0, 0, dst=panel)

    warped = cv2.warpPerspective(ground, H, (panel.shape[1], panel.shape[0]))
    mask = np.zeros_like(panel[:,:,0]); cv2.fillConvexPoly(mask, dst.astype(np.int32), 255)
    mask3 = cv2.merge([mask,mask,mask]); np.copyto(panel, warped, where=mask3.astype(bool))

# --------- panel drawing ----------
def draw_panel(panel, totals, times, frame_wh, dets, tracker, heat_buf):
    h, w = panel.shape[:2]
    x_left = PAD
    y = PAD + 12

    
    cv2.putText(panel, "Parking Stats", (x_left, y),
                FONT, TITLE_SCALE, FG_COLOR, THICK, cv2.LINE_AA)
    y += LINE_GAP
    cv2.line(panel, (PAD, y), (w - PAD, y), SEPARATOR, 1)
    y += SEC_GAP

    
    CARD_PAD   = 10
    CARD_H     = 54
    card_x1    = PAD
    card_y1    = y
    card_x2    = w - PAD
    card_y2    = y + CARD_H
    cv2.rectangle(panel, (card_x1, card_y1), (card_x2, card_y2),
                  (20, 20, 20), -1, cv2.LINE_AA)
    cv2.rectangle(panel, (card_x1, card_y1), (card_x2, card_y2),
                  (80, 80, 80), 1, cv2.LINE_AA)
    tx = card_x1 + CARD_PAD
    ty = card_y1 + CARD_PAD + 14
    cv2.putText(panel, f"Total free: {totals.get('free',0)}",
                (tx, ty), FONT, TEXT_SCALE, COLOR_FREE, THICK, cv2.LINE_AA)
    ty += 22
    cv2.putText(panel, f"Total occupied: {totals.get('car',0)}",
                (tx, ty), FONT, TEXT_SCALE, COLOR_CAR, THICK, cv2.LINE_AA)

    y = card_y2 + SEC_GAP
    cv2.line(panel, (PAD, y), (w - PAD, y), SEPARATOR, 1)
    y += SEC_GAP

    reserved_for_minimap = MINIMAP_H + (SEC_GAP + LINE_GAP)
    table_h = max(0, h - y - reserved_for_minimap - PAD)
    table_w = w - 2 * PAD
    draw_timer_table(panel, (x_left, y), (table_w, table_h), times, time.time())
    y += table_h

    y += LINE_GAP
    cv2.line(panel, (PAD, y), (w - PAD, y), SEPARATOR, 1)
    y += SEC_GAP

    mm_w = w - 2 * PAD
    mm_h = MINIMAP_H
    mm_x = (w - mm_w) // 2
    mm_y = min(y, h - PAD - mm_h)
    draw_minimap_3d(panel, (mm_x, mm_y), (mm_w, mm_h), frame_wh, dets, tracker, heat_buf)

# =======================
# SETUP
# =======================
if not os.path.exists(VIDEO_IN):
    raise FileNotFoundError(VIDEO_IN)

cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise RuntimeError("Could not open input video.")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# -------- ROI selection BEFORE processing (draw-only) --------
ok, first_frame = cap.read()
if not ok:
    raise RuntimeError("Could not read first frame for ROI drawing.")
rois = select_rois_interactive(first_frame)  
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)          

out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (fw + PANEL_W, fh))
if not out.isOpened():
    cap.release()
    raise RuntimeError("Could not open output video for writing.")


model = YOLO(MODEL_PATH)
names = model.names
swapped_names = {0: names[1], 1: names[0]}
color_by_name = {"car": COLOR_CAR, "free": COLOR_FREE, "Vehiculo": COLOR_DETR}


aerial_yolo = YOLO(AERIAL_MODEL_PATH)
aerial_names = aerial_yolo.names  

def aerial_detect_xyxy(frame_bgr):
    """
    Returns list of dicts: {"xyxy": np.array([x1,y1,x2,y2]), "conf": float, "name": "Vehiculo"}
    filtered to AERIAL_ACCEPT_IDS (class IDs).
    """
    res = aerial_yolo.predict(
        source=frame_bgr,
        conf=AERIAL_CONF_THR,
        iou=AERIAL_IOU_THR,
        imgsz=960,          
        agnostic_nms=False,
        max_det=1000,
        verbose=False
    )[0]

    out = []
    for b in res.boxes:
        conf = float(b.conf[0])
        cls  = int(b.cls[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())

        if (not AERIAL_ACCEPT_IDS) or (cls in AERIAL_ACCEPT_IDS):
            out.append({
                "xyxy": np.array([x1, y1, x2, y2], dtype=int),
                "conf": conf,
                "cls": cls,
                "name": "Vehiculo"   
            })
    return out

# Trackers
tracker = SimpleTracker(fps=fps, iou_match=IOU_MATCH)         
detr_tracker = SimpleTracker(fps=fps, iou_match=IOU_MATCH)    

frame_time = 1.0 / float(fps)
heat_buf = np.zeros((HEATMAP_H, HEATMAP_W), dtype=np.float32)

# =======================
# MAIN LOOP
# =======================
try:
    while True:
        ok, frame = cap.read()
        if not ok: break

       
        base = (cv2.bilateralFilter(frame, d=5, sigmaColor=50, sigmaSpace=50)
                if USE_BILATERAL else cv2.GaussianBlur(frame, (GAUSS_KSIZE, GAUSS_KSIZE), GAUSS_SIGMA))
        clean = base  

        
        results = model.predict(source=clean, conf=CONF_THRESH, iou=IOU_NMS1,
                                imgsz=640, agnostic_nms=True, max_det=1500, verbose=False)

        dets = []
        r = results[0]
        for b in r.boxes:
            conf = float(b.conf[0]); cls = int(b.cls[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            name = swapped_names.get(cls, str(cls)).lower()
            dets.append({"xyxy": np.array([x1,y1,x2,y2], dtype=int), "conf": conf, "cls": cls, "name": name})

        dets = nms_extra(dets, iou_thr=IOU_NMS2)
        dets = tracker.step(dets, frame_time)

        total_cars = sum(1 for d in dets if d["name"] == "car")
        total_free = sum(1 for d in dets if d["name"] == "free")

        
        tracker.render_tracks(frame, color_by_name)

        
        free_centers = []
        for d in dets:
            if d["name"] != "free": continue
            x1,y1,x2,y2 = d["xyxy"].tolist()
            draw_box_with_fill(frame, x1, y1, x2, y2, COLOR_FREE, alpha=0.17, border=2)
            draw_centered_label_in_box(frame, x1, y1, x2, y2, "free", COLOR_FREE)
            free_centers.append(box_center((x1,y1,x2,y2)))

        # -------- run AERIAL YOLO and TRACK only boxes inside ROI THIS FRAME --------
        aerial_dets_raw = aerial_detect_xyxy(clean)

        aerial_dets = []
        for d in aerial_dets_raw:
            x1, y1, x2, y2 = d["xyxy"].tolist()
            if not box_inside_roi((x1, y1, x2, y2), rois):
                continue
            aerial_dets.append(d)

        
        detr_tracked = detr_tracker.step(aerial_dets, frame_time)

        
        present_tids = set()
        for d in detr_tracked:
            if d.get("track_id") is not None and d.get("name") == "Vehiculo":
                present_tids.add(d["track_id"])

        veh_centers = []
        for tid in list(present_tids):
            tr = detr_tracker.tracks.get(tid)
            if tr is None:
                continue
            xyxy = tr["xyxy_smooth"].astype(int)
            box  = tuple(xyxy.tolist())
            if not box_inside_roi(box, rois):
                continue
            x1,y1,x2,y2 = box
            # draw_box_with_fill(frame, x1, y1, x2, y2, COLOR_DETR, alpha=0.18, border=2)
            # draw_centered_label_in_box(frame, x1, y1, x2, y2, "Vehiculo", COLOR_DETR)
            veh_centers.append(box_center(box))



        if veh_centers and free_centers:
            free_outside = [fc for fc in free_centers if not point_in_any_roi(fc, rois)]
            if free_outside:
                for vc in veh_centers:
                    nearest_free = min(free_outside, key=lambda c: (c[0]-vc[0])**2 + (c[1]-vc[1])**2)
                    draw_dotted_line(frame, vc, nearest_free, color=(255,255,255), gap=7, radius=2)



        # panel
        panel = np.full((fh, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
        draw_panel(panel, {"free": total_free, "car": total_cars}, tracker.times_summary(),
                   (fw, fh), dets, tracker, heat_buf)

        composed = np.zeros((fh, fw + PANEL_W, 3), dtype=np.uint8)
        composed[:, :fw] = frame
        composed[:, fw:] = panel
        out.write(composed)

        cv2.imshow("Parking (video + side panel)", composed)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    cap.release(); out.release(); cv2.destroyAllWindows()

print(f"Done. Wrote: {VIDEO_OUT}")
