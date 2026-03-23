"""
梯形裁剪工具 - Streamlit 網頁版（Canvas 點擊版）
直接在圖片上點擊選取四個角點，支援拖曳微調
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
[data-testid="stAppViewContainer"] { background: #1a1a2e; }
[data-testid="stSidebar"] { background: #16213e; }
h1, h2, h3 { color: #e2e8f0; }
p, label, .stCaption { color: #94a3b8 !important; }
.step-badge {
    display:inline-block; background:#0a84ff; color:white;
    border-radius:50%; width:28px; height:28px; text-align:center;
    line-height:28px; font-weight:bold; margin-right:8px;
}
.tip-box {
    background:#1e3a5f; border-left:4px solid #0a84ff;
    padding:10px 16px; border-radius:4px; margin:8px 0;
    color:#cbd5e1 !important; font-size:14px;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# 幾何工具
# ════════════════════════════════════════════════════════════════
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

def calc_canvas_size(img_w, img_h, max_w=820):
    if img_w > max_w:
        s = max_w / img_w
        return max_w, int(img_h * s), s
    return img_w, img_h, 1.0

def pil_to_data_url(pil_img):
    """Convert PIL Image to base64 JPEG data URL."""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def build_fabric_objects(bg_data_url, canvas_w, canvas_h, pts_canvas):
    """
    Build fabric.js JSON with a locked background image + draggable corner circles.
    Avoids the broken background_image parameter in st_canvas entirely.
    """
    CORNER_COLORS = ["#00c8ff", "#64ff64", "#ffb400", "#ff50b4"]
    objects = [
        {
            "type": "image",
            "version": "4.4.0",
            "originX": "left",
            "originY": "top",
            "left": 0, "top": 0,
            "width": canvas_w, "height": canvas_h,
            "scaleX": 1, "scaleY": 1,
            "src": bg_data_url,
            "selectable": False,
            "evented": False,
            "lockMovementX": True,
            "lockMovementY": True,
            "hasControls": False,
            "hasBorders": False,
            "crossOrigin": "anonymous",
        }
    ]
    for i, pt in enumerate(pts_canvas):
        objects.append({
            "type": "circle",
            "version": "4.4.0",
            "originX": "left",
            "originY": "top",
            "left":   pt[0] - 10,
            "top":    pt[1] - 10,
            "width":  20, "height": 20,
            "radius": 10,
            "fill":   CORNER_COLORS[i],
            "stroke": "white",
            "strokeWidth": 2,
            "selectable": True,
            "evented": True,
        })
    return {"version": "4.4.0", "objects": objects}

def draw_polygon_on_pil(pil_img, pts_canvas, scale):
    """Draw selection polygon lines onto the image."""
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    if len(pts_canvas) >= 2:
        for i in range(len(pts_canvas) - 1):
            cv2.line(bgr,
                     (int(pts_canvas[i][0]),   int(pts_canvas[i][1])),
                     (int(pts_canvas[i+1][0]), int(pts_canvas[i+1][1])),
                     (0, 200, 255), 2, cv2.LINE_AA)
    if len(pts_canvas) == 4:
        cv2.line(bgr,
                 (int(pts_canvas[3][0]), int(pts_canvas[3][1])),
                 (int(pts_canvas[0][0]), int(pts_canvas[0][1])),
                 (0, 200, 255), 2, cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# ════════════════════════════════════════════════════════════════
# 標題
# ════════════════════════════════════════════════════════════════
st.title("✂️ 梯形裁剪 / 透視校正工具")

# ════════════════════════════════════════════════════════════════
# Sidebar：上傳 + 說明
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📂 上傳圖片")
    uploaded = st.file_uploader(
        "支援 JPG、PNG、BMP、WEBP",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
<div class='tip-box'>
<b>操作步驟：</b><br>
<span class='step-badge'>1</span>上傳圖片<br><br>
<span class='step-badge'>2</span>在圖上依序點擊<br>
&nbsp;&nbsp;&nbsp;&nbsp;① 左上 → ② 右上<br>
&nbsp;&nbsp;&nbsp;&nbsp;③ 右下 → ④ 左下<br><br>
<span class='step-badge'>3</span>拖曳圓點微調位置<br><br>
<span class='step-badge'>4</span>點「執行裁剪」下載
</div>
""", unsafe_allow_html=True)
    st.markdown("---")

    if uploaded:
        st.markdown("**輸出格式**")
        fmt = st.radio("格式", ["JPG", "PNG", "BMP"], horizontal=True,
                       label_visibility="collapsed")
        st.markdown("---")
        if st.button("🔄 重置角點", use_container_width=True):
            st.session_state.pts_canvas = []
            st.rerun()

if uploaded is None:
    st.markdown("""
    <div style='text-align:center; padding:80px 0; color:#475569;'>
        <div style='font-size:64px'>📂</div>
        <div style='font-size:20px; margin-top:16px'>請從左側上傳圖片開始使用</div>
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
cW, cH, scale = calc_canvas_size(W, H, max_w=820)

# session state 初始化
if "pts_canvas" not in st.session_state:
    st.session_state.pts_canvas = []
if "last_file" not in st.session_state:
    st.session_state.last_file = None

# 換新圖時自動重置
if st.session_state.last_file != uploaded.name:
    st.session_state.pts_canvas = []
    st.session_state.last_file  = uploaded.name

pts_canvas = st.session_state.pts_canvas

# ════════════════════════════════════════════════════════════════
# 主畫面
# ════════════════════════════════════════════════════════════════
col_canvas, col_result = st.columns([3, 2], gap="large")

with col_canvas:
    n = len(pts_canvas)
    label_next = ["① 左上", "② 右上", "③ 右下", "④ 左下"]
    if n < 4:
        st.markdown(f"**🖱️ 請在圖片上點擊第 {n+1} 個角點：{label_next[n]}**")
    else:
        st.markdown("**✅ 4個角點已選取，可拖曳圓點微調位置**")

    # Build background: draw polygon lines onto the resized photo
    pil_resized = Image.fromarray(cv2.cvtColor(
        cv2.resize(original, (cW, cH)), cv2.COLOR_BGR2RGB))
    bg_pil = draw_polygon_on_pil(pil_resized, pts_canvas, scale) if pts_canvas else pil_resized

    # Embed background + corner circles entirely inside initial_drawing.
    # background_image=None avoids the broken image_to_url code path.
    bg_url         = pil_to_data_url(bg_pil)
    fabric_drawing = build_fabric_objects(bg_url, cW, cH, pts_canvas)

    canvas_result = st_canvas(
        fill_color           = "rgba(0,0,0,0)",
        stroke_color         = "#00c8ff",
        stroke_width         = 2,
        background_image     = None,
        update_streamlit     = True,
        width                = cW,
        height               = cH,
        drawing_mode         = "point" if n < 4 else "transform",
        point_display_radius = 10,
        initial_drawing      = fabric_drawing,
        key                  = "canvas_main",
    )

    # Parse canvas output
    if canvas_result.json_data is not None:
        objs = canvas_result.json_data.get("objects", [])
        # New clicks (point mode) — ignore the background image object
        new_pts = [o for o in objs if o.get("type") == "point"]
        if new_pts and n < 4:
            for p in new_pts:
                if len(st.session_state.pts_canvas) < 4:
                    st.session_state.pts_canvas.append([p["left"], p["top"]])
            st.rerun()
        # Dragged circles (transform mode)
        moved = [o for o in objs if o.get("type") == "circle"]
        if moved and n == 4:
            new_list = []
            for o in moved:
                cx = o["left"] + o.get("width", 20) / 2
                cy = o["top"]  + o.get("height", 20) / 2
                new_list.append([cx, cy])
            if new_list != st.session_state.pts_canvas:
                st.session_state.pts_canvas = new_list
                st.rerun()

# ════════════════════════════════════════════════════════════════
# 右側：裁剪結果
# ════════════════════════════════════════════════════════════════
with col_result:
    st.markdown("**✂️ 裁剪結果**")

    if len(pts_canvas) < 4:
        st.markdown("""
        <div style='background:#1e293b; border-radius:8px; padding:40px;
                    text-align:center; color:#475569; margin-top:8px;'>
            <div style='font-size:40px'>🖼️</div>
            <div style='margin-top:12px'>選取4個角點後<br>結果會顯示在這裡</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        pts_orig = [(x / scale, y / scale) for x, y in pts_canvas]

        if st.button("🚀 執行裁剪", type="primary", use_container_width=True):
            with st.spinner("透視校正中..."):
                result = perspective_transform(original, pts_orig)
            rH, rW = result.shape[:2]
            st.session_state.result     = result
            st.session_state.result_dim = (rW, rH)

        if "result" in st.session_state:
            result = st.session_state.result
            rW, rH = st.session_state.result_dim
            st.success(f"完成！{rW} × {rH} px")
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                     use_container_width=True)

            fmt_ext  = fmt.lower() if "fmt" in dir() else "jpg"
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            dl_name  = f"{fname_base}_crop_{ts}.{fmt_ext}"
            mime_map = {"jpg": "image/jpeg", "png": "image/png", "bmp": "image/bmp"}

            st.download_button(
                label               = f"⬇️ 下載 {fmt_ext.upper()}",
                data                = img_to_bytes(result, fmt_ext),
                file_name           = dl_name,
                mime                = mime_map.get(fmt_ext, "image/jpeg"),
                use_container_width = True,
            )
