"""
梯形裁剪工具 - Streamlit 網頁版（Canvas 點擊版）
直接在圖片上點擊選取四個角點，支援拖曳微調
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime

# ── Compatibility patch ───────────────────────────────────────────────────────
# streamlit-drawable-canvas calls st.image_to_url() with the old 5-arg
# signature that was removed in Streamlit ≥ 1.37.  We replace it with a
# version that converts the PIL Image to a base64 data-URL itself so the
# rest of the canvas library (including _resize_img) still receives a PIL
# Image and only the final URL step is patched.
import streamlit_drawable_canvas as _sdc_module
import streamlit_drawable_canvas.__init__ as _sdc_init  # noqa: F401

def _patched_image_to_url(image, width, clamp, channels, output_format, image_id=""):
    buf = io.BytesIO()
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"

# Patch the reference used inside the canvas package
import streamlit_drawable_canvas.__init__ as _sdc
import streamlit.elements.image as _st_image_mod
_st_image_mod.image_to_url = _patched_image_to_url   # module-level ref
_sdc.st_image.image_to_url = _patched_image_to_url   # local alias inside canvas
# ─────────────────────────────────────────────────────────────────────────────

from streamlit_drawable_canvas import st_canvas

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

def draw_polygon_on_pil(pil_img, pts_canvas, scale):
    """在 PIL 圖上畫框線（回傳給 canvas 背景用）"""
    import cv2, numpy as np
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    if len(pts_canvas) >= 2:
        for i in range(len(pts_canvas)-1):
            cv2.line(bgr,
                     (int(pts_canvas[i][0]), int(pts_canvas[i][1])),
                     (int(pts_canvas[i+1][0]), int(pts_canvas[i+1][1])),
                     (0,200,255), 2, cv2.LINE_AA)
    if len(pts_canvas) == 4:
        cv2.line(bgr,
                 (int(pts_canvas[3][0]), int(pts_canvas[3][1])),
                 (int(pts_canvas[0][0]), int(pts_canvas[0][1])),
                 (0,200,255), 2, cv2.LINE_AA)
    labels = ["①","②","③","④"]
    colors = [(0,200,255),(100,255,100),(255,180,0),(255,80,180)]
    for i, pt in enumerate(pts_canvas):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(bgr, (x,y), 10, colors[i], -1)
        cv2.circle(bgr, (x,y), 10, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(bgr, labels[i], (x+13, y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 1, cv2.LINE_AA)
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
        type=["jpg","jpeg","png","bmp","webp"],
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
        fmt = st.radio("格式", ["JPG","PNG","BMP"], horizontal=True,
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

H, W       = original.shape[:2]
fname_base = uploaded.name.rsplit(".", 1)[0]
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

pts_canvas = st.session_state.pts_canvas  # list of [x,y] in canvas coords

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

    # 背景圖（帶框線）
    pil_orig = Image.fromarray(cv2.cvtColor(
        cv2.resize(original, (cW, cH)), cv2.COLOR_BGR2RGB))
    bg_img = draw_polygon_on_pil(pil_orig, pts_canvas, scale) if pts_canvas else pil_orig

    # Canvas 物件清單：每個角點畫一個可拖曳的圓
    CORNER_COLORS = ["#00c8ff","#64ff64","#ffb400","#ff50b4"]
    objects = []
    for i, pt in enumerate(pts_canvas):
        objects.append({
            "type": "circle",
            "version": "4.4.0",
            "left":   pt[0] - 10,
            "top":    pt[1] - 10,
            "width":  20, "height": 20,
            "fill":   CORNER_COLORS[i],
            "stroke": "white", "strokeWidth": 2,
            "selectable": True,
        })

    canvas_result = st_canvas(
        fill_color   = "rgba(0,0,0,0)",
        stroke_color = "#00c8ff",
        stroke_width = 2,
        background_image = bg_img,
        update_streamlit = True,
        width  = cW,
        height = cH,
        drawing_mode = "point" if n < 4 else "transform",
        point_display_radius = 10,
        initial_drawing = {"version": "4.4.0", "objects": objects} if objects else None,
        key = "canvas_main",
    )

    # 解析 canvas 回傳
    if canvas_result.json_data is not None:
        objs = canvas_result.json_data.get("objects", [])
        # 新點擊（point 模式）
        new_pts = [o for o in objs if o.get("type") == "point"]
        if new_pts and n < 4:
            for p in new_pts:
                if len(st.session_state.pts_canvas) < 4:
                    st.session_state.pts_canvas.append([p["left"], p["top"]])
            st.rerun()
        # 拖曳（transform 模式，circle 物件）
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
        # 轉回原始座標
        pts_orig = [(x / scale, y / scale) for x, y in pts_canvas]

        if st.button("🚀 執行裁剪", type="primary", use_container_width=True):
            with st.spinner("透視校正中..."):
                result = perspective_transform(original, pts_orig)
            rH, rW = result.shape[:2]
            st.session_state.result     = result
            st.session_state.result_dim = (rW, rH)

        if "result" in st.session_state:
            result   = st.session_state.result
            rW, rH   = st.session_state.result_dim
            st.success(f"完成！{rW} × {rH} px")
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                     use_container_width=True)

            fmt_ext  = fmt.lower() if "fmt" in dir() else "jpg"
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            dl_name  = f"{fname_base}_crop_{ts}.{fmt_ext}"
            mime_map = {"jpg":"image/jpeg","png":"image/png","bmp":"image/bmp"}

            st.download_button(
                label    = f"⬇️ 下載 {fmt_ext.upper()}",
                data     = img_to_bytes(result, fmt_ext),
                file_name= dl_name,
                mime     = mime_map.get(fmt_ext, "image/jpeg"),
                use_container_width = True,
            )
