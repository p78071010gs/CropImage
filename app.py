"""
梯形裁剪工具 - Streamlit 網頁版
點選四個角點 → 預覽 → 下載
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import io
import base64
from datetime import datetime

st.set_page_config(page_title="梯形裁剪工具", page_icon="✂️", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f172a; }
[data-testid="stSidebar"] { background: #1e293b; }
h1, h2, h3 { color: #f1f5f9; }
p, label { color: #94a3b8 !important; }

.coord-table {
    width: 100%; border-collapse: collapse;
    font-size: 13px; margin-top: 8px;
}
.coord-table th {
    background: #1e3a5f; color: #7dd3fc;
    padding: 6px 10px; text-align: left;
    border-bottom: 1px solid #334155;
}
.coord-table td {
    padding: 6px 10px; color: #cbd5e1;
    border-bottom: 1px solid #1e293b;
}
.coord-table tr:hover td { background: #1e293b; }

.dot-badge {
    display: inline-block;
    width: 12px; height: 12px; border-radius: 50%;
    margin-right: 6px; vertical-align: middle;
}
.tip-box {
    background: #1e3a5f; border-left: 4px solid #0a84ff;
    padding: 10px 16px; border-radius: 4px; margin: 8px 0;
    color: #cbd5e1 !important; font-size: 13px; line-height: 1.7;
}
.step-badge {
    display: inline-block; background: #0a84ff; color: white;
    border-radius: 50%; width: 22px; height: 22px; text-align: center;
    line-height: 22px; font-weight: bold; margin-right: 6px; font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# 工具函式
# ════════════════════════════════════════════════════════════════
CORNER_COLORS  = ["#00c8ff", "#64ff64", "#ffb400", "#ff50b4"]
CORNER_LABELS  = ["① 左上", "② 右上", "③ 右下", "④ 左下"]
CORNER_CV_COLS = [(0,200,255),(100,255,100),(255,180,0),(255,80,180)]

def order_points(pts):
    pts  = np.array(pts, dtype="float32")
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    out  = np.zeros((4, 2), dtype="float32")
    out[0] = pts[np.argmin(s)]
    out[2] = pts[np.argmax(s)]
    out[1] = pts[np.argmin(diff)]
    out[3] = pts[np.argmax(diff)]
    return out

def perspective_transform(src_img, pts):
    ordered = order_points(pts)
    tl, tr, br, bl = ordered
    dst_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    dst_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.array([[0,0],[dst_w-1,0],[dst_w-1,dst_h-1],[0,dst_h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(src_img, M, (dst_w, dst_h))

def img_to_bytes(img_bgr, fmt="jpg"):
    if fmt == "jpg":
        _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif fmt == "png":
        _, buf = cv2.imencode(".png", img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        _, buf = cv2.imencode(".bmp", img_bgr)
    return buf.tobytes()

def calc_canvas_size(img_w, img_h, max_w=800):
    if img_w > max_w:
        s = max_w / img_w
        return max_w, int(img_h * s), s
    return img_w, img_h, 1.0

def pil_to_data_url(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def draw_overlay(pil_img, pts):
    """Draw polygon + numbered dots onto a copy of the image."""
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    n = len(pts)
    # Lines between clicked points
    if n >= 2:
        for i in range(n - 1):
            cv2.line(bgr,
                     (int(pts[i][0]),   int(pts[i][1])),
                     (int(pts[i+1][0]), int(pts[i+1][1])),
                     (0, 200, 255), 2, cv2.LINE_AA)
    if n == 4:
        cv2.line(bgr,
                 (int(pts[3][0]), int(pts[3][1])),
                 (int(pts[0][0]), int(pts[0][1])),
                 (0, 200, 255), 2, cv2.LINE_AA)
    # Dot + label per point
    labels = ["①","②","③","④"]
    for i, pt in enumerate(pts):
        x, y = int(pt[0]), int(pt[1])
        col = CORNER_CV_COLS[i]
        cv2.circle(bgr, (x, y), 11, col, -1)
        cv2.circle(bgr, (x, y), 11, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(bgr, labels[i], (x + 14, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2, cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def build_fabric_json(bg_data_url, canvas_w, canvas_h, pts):
    """
    Embed background + circles entirely in initial_drawing.
    background_image=None avoids the broken image_to_url in newer Streamlit.
    """
    objects = [{
        "type": "image", "version": "4.4.0",
        "originX": "left", "originY": "top",
        "left": 0, "top": 0,
        "width": canvas_w, "height": canvas_h,
        "scaleX": 1, "scaleY": 1,
        "src": bg_data_url,
        "selectable": False, "evented": False,
        "lockMovementX": True, "lockMovementY": True,
        "hasControls": False, "hasBorders": False,
        "crossOrigin": "anonymous",
    }]
    for i, pt in enumerate(pts):
        objects.append({
            "type": "circle", "version": "4.4.0",
            "originX": "left", "originY": "top",
            "left": pt[0] - 11, "top": pt[1] - 11,
            "width": 22, "height": 22, "radius": 11,
            "fill": CORNER_COLORS[i],
            "stroke": "white", "strokeWidth": 2,
            "selectable": True, "evented": True,
        })
    return {"version": "4.4.0", "objects": objects}


# ════════════════════════════════════════════════════════════════
# 標題
# ════════════════════════════════════════════════════════════════
st.title("✂️ 梯形裁剪 / 透視校正工具")

# ════════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📂 上傳圖片")
    uploaded = st.file_uploader(
        "支援 JPG、PNG、BMP、WEBP",
        type=["jpg","jpeg","png","bmp","webp"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
<div class='tip-box'>
<b>操作步驟：</b><br>
<span class='step-badge'>1</span>上傳圖片<br>
<span class='step-badge'>2</span>依序點擊四個角點<br>
<span class='step-badge'>3</span>拖曳圓點微調位置<br>
<span class='step-badge'>4</span>點「預覽截圖」確認<br>
<span class='step-badge'>5</span>點「下載」儲存結果
</div>
""", unsafe_allow_html=True)
    st.markdown("---")

    if uploaded:
        st.markdown("**輸出格式**")
        fmt = st.radio("格式", ["JPG","PNG","BMP"], horizontal=True,
                       label_visibility="collapsed")

if uploaded is None:
    st.markdown("""
    <div style='text-align:center;padding:80px 0;color:#475569;'>
        <div style='font-size:64px'>📂</div>
        <div style='font-size:20px;margin-top:16px'>請從左側上傳圖片開始使用</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ════════════════════════════════════════════════════════════════
# 讀取圖片
# ════════════════════════════════════════════════════════════════
file_bytes = np.frombuffer(uploaded.read(), np.uint8)
original   = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if original is None:
    st.error("無法讀取圖片，請換一個檔案。"); st.stop()

H, W          = original.shape[:2]
fname_base    = uploaded.name.rsplit(".", 1)[0]
cW, cH, scale = calc_canvas_size(W, H, max_w=800)

# ── Session state ────────────────────────────────────────────────
for key, default in [("pts_canvas", []), ("last_file", None), ("result", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.last_file != uploaded.name:
    st.session_state.pts_canvas = []
    st.session_state.last_file  = uploaded.name
    st.session_state.result     = None

pts_canvas = st.session_state.pts_canvas
n = len(pts_canvas)

# ════════════════════════════════════════════════════════════════
# Layout: canvas | right panel
# ════════════════════════════════════════════════════════════════
col_canvas, col_right = st.columns([3, 2], gap="large")

# ── Canvas ───────────────────────────────────────────────────────
with col_canvas:
    if n < 4:
        st.markdown(f"**🖱️ 點擊第 {n+1} 個角點：{CORNER_LABELS[n]}**")
    else:
        st.markdown("**✅ 4個角點已選取 — 可拖曳圓點微調**")

    pil_resized = Image.fromarray(cv2.cvtColor(
        cv2.resize(original, (cW, cH)), cv2.COLOR_BGR2RGB))
    bg_pil = draw_overlay(pil_resized, pts_canvas) if pts_canvas else pil_resized

    fabric = build_fabric_json(pil_to_data_url(bg_pil), cW, cH, pts_canvas)

    canvas_result = st_canvas(
        fill_color           = "rgba(0,0,0,0)",
        stroke_color         = "#00c8ff",
        stroke_width         = 2,
        background_image     = None,
        update_streamlit     = True,
        width                = cW,
        height               = cH,
        drawing_mode         = "point" if n < 4 else "transform",
        point_display_radius = 11,
        initial_drawing      = fabric,
        key                  = "canvas_main",
    )

    # ── Parse canvas events ──────────────────────────────────────
    if canvas_result.json_data is not None:
        objs = canvas_result.json_data.get("objects", [])
        new_pts = [o for o in objs if o.get("type") == "point"]
        if new_pts and n < 4:
            for p in new_pts:
                if len(st.session_state.pts_canvas) < 4:
                    st.session_state.pts_canvas.append([p["left"], p["top"]])
            st.rerun()
        moved = [o for o in objs if o.get("type") == "circle"]
        if moved and n == 4:
            new_list = [[o["left"] + o.get("width",22)/2,
                         o["top"]  + o.get("height",22)/2] for o in moved]
            if new_list != st.session_state.pts_canvas:
                st.session_state.pts_canvas = new_list
                st.session_state.result = None   # invalidate old preview
                st.rerun()

# ── Right panel ──────────────────────────────────────────────────
with col_right:

    # ── 1. 點選座標表 ────────────────────────────────────────────
    st.markdown("#### 📍 已選角點座標")
    if n == 0:
        st.caption("尚未選取任何點")
    else:
        rows = ""
        for i, pt in enumerate(pts_canvas):
            # Convert canvas coords back to original image coords
            ox = int(pt[0] / scale)
            oy = int(pt[1] / scale)
            dot = f'<span class="dot-badge" style="background:{CORNER_COLORS[i]}"></span>'
            rows += f"<tr><td>{dot}{CORNER_LABELS[i]}</td><td>{ox}</td><td>{oy}</td></tr>"
        st.markdown(f"""
<table class='coord-table'>
  <thead><tr><th>角點</th><th>X（原圖px）</th><th>Y（原圖px）</th></tr></thead>
  <tbody>{rows}</tbody>
</table>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 2. 操作按鈕區 ────────────────────────────────────────────
    st.markdown("#### 🎛️ 操作")

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        reset_clicked = st.button("🔄 重設點位", use_container_width=True,
                                  disabled=(n == 0))
    with btn_col2:
        preview_clicked = st.button("👁️ 預覽截圖", use_container_width=True,
                                    type="primary", disabled=(n < 4))

    if reset_clicked:
        st.session_state.pts_canvas = []
        st.session_state.result     = None
        st.rerun()

    if preview_clicked and n == 4:
        pts_orig = [(pt[0] / scale, pt[1] / scale) for pt in pts_canvas]
        with st.spinner("透視校正中..."):
            result = perspective_transform(original, pts_orig)
        st.session_state.result = result

    st.markdown("---")

    # ── 3. 預覽 + 下載 ───────────────────────────────────────────
    st.markdown("#### ✂️ 裁剪結果")

    if st.session_state.result is None:
        placeholder_msg = "選取 4 個角點後點「預覽截圖」" if n < 4 else "點擊「👁️ 預覽截圖」查看結果"
        st.markdown(f"""
        <div style='background:#1e293b;border-radius:8px;padding:40px;
                    text-align:center;color:#475569;'>
            <div style='font-size:36px'>🖼️</div>
            <div style='margin-top:10px;font-size:13px'>{placeholder_msg}</div>
        </div>""", unsafe_allow_html=True)
    else:
        result = st.session_state.result
        rH, rW = result.shape[:2]
        st.success(f"完成！輸出尺寸：{rW} × {rH} px")
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)

        fmt_ext  = fmt.lower() if "fmt" in dir() else "jpg"
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        dl_name  = f"{fname_base}_crop_{ts}.{fmt_ext}"
        mime_map = {"jpg":"image/jpeg","png":"image/png","bmp":"image/bmp"}

        st.download_button(
            label               = f"⬇️ 下載 {fmt_ext.upper()}",
            data                = img_to_bytes(result, fmt_ext),
            file_name           = dl_name,
            mime                = mime_map.get(fmt_ext, "image/jpeg"),
            use_container_width = True,
            type                = "primary",
        )
