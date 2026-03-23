"""
梯形裁剪工具 - Streamlit 網頁版
Trapezoid Crop Tool - Streamlit Web Edition

操作流程：
  1. 上傳圖片
  2. 用滑桿調整四個角點座標（或直接在圖上點擊預覽）
  3. 點「執行裁剪」
  4. 下載結果
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from datetime import datetime

# ════════════════════════════════════════════════════════════════
# 頁面設定
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="梯形裁剪工具",
    page_icon="✂️",
    layout="wide",
)

st.markdown("""
<style>
    .main { background-color: #1a1a2e; }
    h1 { color: #e2e2e2; }
    .stSlider label { color: #aaaaaa !important; font-size: 13px; }
    .corner-title { font-size: 14px; font-weight: bold; color: #4fc3f7; margin-bottom: 4px; }
    div[data-testid="stDownloadButton"] button {
        background-color: #0a84ff;
        color: white;
        font-size: 16px;
        padding: 10px 28px;
        border-radius: 8px;
        border: none;
        width: 100%;
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
    """pts: [(x,y)×4] 原始圖片座標"""
    ordered = order_points(pts)
    tl, tr, br, bl = ordered
    dst_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    dst_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.array([[0, 0], [dst_w-1, 0],
                    [dst_w-1, dst_h-1], [0, dst_h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(src_img, M, (dst_w, dst_h))


def draw_overlay(img, pts, labels=["①左上","②右上","③右下","④左下"]):
    """在圖上畫梯形框線與角點"""
    frame = img.copy()
    n = len(pts)
    if n >= 2:
        for i in range(n - 1):
            cv2.line(frame, tuple(map(int, pts[i])),
                     tuple(map(int, pts[i+1])), (0, 200, 255), 2, cv2.LINE_AA)
    if n == 4:
        cv2.line(frame, tuple(map(int, pts[3])),
                 tuple(map(int, pts[0])), (0, 200, 255), 2, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(frame, (x, y), 10, (0, 220, 30), -1)
        cv2.circle(frame, (x, y), 10, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, labels[i], (x+13, y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    return frame


def img_to_bytes(img_bgr, fmt="jpg"):
    if fmt == "jpg":
        _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif fmt == "png":
        _, buf = cv2.imencode(".png", img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        _, buf = cv2.imencode(".bmp", img_bgr)
    return buf.tobytes()


def resize_preview(img, max_w=900):
    h, w = img.shape[:2]
    if w > max_w:
        s = max_w / w
        return cv2.resize(img, (max_w, int(h * s))), s
    return img.copy(), 1.0


# ════════════════════════════════════════════════════════════════
# 標題
# ════════════════════════════════════════════════════════════════
st.title("✂️ 梯形裁剪 / 透視校正工具")
st.caption("上傳圖片 → 調整四個角點 → 執行裁剪 → 下載結果")

# ════════════════════════════════════════════════════════════════
# 步驟 1：上傳圖片
# ════════════════════════════════════════════════════════════════
st.markdown("### 📂 步驟 1：上傳圖片")
uploaded = st.file_uploader(
    "支援 JPG、PNG、BMP、WEBP",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.info("請上傳一張圖片開始使用。")
    st.stop()

# 讀取圖片
file_bytes = np.frombuffer(uploaded.read(), np.uint8)
original   = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if original is None:
    st.error("無法讀取圖片，請換一個檔案。")
    st.stop()

H, W = original.shape[:2]
fname_base = uploaded.name.rsplit(".", 1)[0]

st.success(f"已載入：**{uploaded.name}**　尺寸：{W} × {H} px")

# ════════════════════════════════════════════════════════════════
# 步驟 2：設定四個角點
# ════════════════════════════════════════════════════════════════
st.markdown("### 🎯 步驟 2：調整四個角點")
st.caption("拖動滑桿調整每個角點的 X / Y 座標（單位：像素）")

# 預設角點：圖片四個邊往內縮 5%
pad_x = int(W * 0.05)
pad_y = int(H * 0.05)
defaults = [
    (pad_x,   pad_y),    # 左上
    (W-pad_x, pad_y),    # 右上
    (W-pad_x, H-pad_y),  # 右下
    (pad_x,   H-pad_y),  # 左下
]
corner_names  = ["① 左上", "② 右上", "③ 右下", "④ 左下"]
corner_colors = ["#4fc3f7", "#81c784", "#ffb74d", "#f06292"]

pts = []
cols = st.columns(4)
for i, (col, name, color, (dx, dy)) in enumerate(
        zip(cols, corner_names, corner_colors, defaults)):
    with col:
        st.markdown(f"<div class='corner-title' style='color:{color}'>{name}</div>",
                    unsafe_allow_html=True)
        x = st.slider(f"X##{i}", 0, W, dx, key=f"x{i}")
        y = st.slider(f"Y##{i}", 0, H, dy, key=f"y{i}")
        pts.append((x, y))
        st.caption(f"({x}, {y})")

# ════════════════════════════════════════════════════════════════
# 即時預覽（含框線）
# ════════════════════════════════════════════════════════════════
st.markdown("### 🖼️ 預覽（梯形框線）")
preview_bgr, _ = resize_preview(original, max_w=900)
pH, pW = preview_bgr.shape[:2]
sx, sy  = pW / W, pH / H
pts_prev = [(x * sx, y * sy) for x, y in pts]
overlay  = draw_overlay(preview_bgr, pts_prev)
overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
st.image(overlay_rgb, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# 步驟 3：執行裁剪
# ════════════════════════════════════════════════════════════════
st.markdown("### ✂️ 步驟 3：執行裁剪並下載")

col_btn, col_fmt, col_empty = st.columns([2, 2, 4])
with col_fmt:
    fmt = st.selectbox("輸出格式", ["JPG", "PNG", "BMP"],
                       label_visibility="visible")

with col_btn:
    do_crop = st.button("🚀  執行裁剪", use_container_width=True,
                        type="primary")

if do_crop:
    with st.spinner("透視校正中..."):
        result = perspective_transform(original, pts)

    rH, rW = result.shape[:2]
    st.success(f"裁剪完成！輸出尺寸：**{rW} × {rH} px**")

    # 結果預覽
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    st.image(result_rgb, caption="裁剪結果", use_container_width=True)

    # 下載按鈕
    fmt_ext  = fmt.lower()
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    dl_name  = f"{fname_base}_crop_{ts}.{fmt_ext}"
    mime_map = {"jpg": "image/jpeg", "png": "image/png", "bmp": "image/bmp"}

    st.download_button(
        label=f"⬇️  下載 {dl_name}",
        data=img_to_bytes(result, fmt_ext),
        file_name=dl_name,
        mime=mime_map[fmt_ext],
        use_container_width=True,
    )
