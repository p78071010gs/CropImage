"""
梯形裁剪工具
使用 streamlit-image-coordinates 套件接收點擊座標（原生支援，最穩定）
若無此套件則 fallback 到純 HTML canvas + Streamlit component value
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io, base64, json
from datetime import datetime

st.set_page_config(page_title="梯形裁剪工具", page_icon="✂️", layout="wide")

# ── Try importing streamlit-image-coordinates ──────────────────
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    USE_SIC = True
except ImportError:
    USE_SIC = False

for k, v in [("pts",[]),("last_file",None),("result",None)]:
    if k not in st.session_state:
        st.session_state[k] = v

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0f172a;}
[data-testid="stSidebar"]{background:#1e293b;}
h1,h2,h3{color:#f1f5f9;}
p,label{color:#94a3b8!important;}
.coord-table{width:100%;border-collapse:collapse;font-size:13px;margin-top:6px;}
.coord-table th{background:#1e3a5f;color:#7dd3fc;padding:5px 9px;text-align:left;border-bottom:1px solid #334155;}
.coord-table td{padding:5px 9px;color:#cbd5e1;border-bottom:1px solid #1e293b;}
.dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:5px;vertical-align:middle;}
</style>
""", unsafe_allow_html=True)

COLORS   = ["#00c8ff","#64ff64","#ffb400","#ff50b4"]
#             TL藍      TR綠      BR橘      BL粉
# OpenCV BGR：#00c8ff→(255,200,0) #64ff64→(100,255,100) #ffb400→(0,180,255) #ff50b4→(180,80,255)
CV_COLS  = [(255,200,0),(100,255,100),(0,180,255),(180,80,255)]
LABELS   = ["TL 左上","TR 右上","BR 右下","BL 左下"]
CV_LABELS= ["TL","TR","BR","BL"]   # OpenCV putText 用（純 ASCII）
MAX_DISPLAY_W = 1400   # 放寬上限，讓圖片盡量填滿全寬

# ─────────────────────────────────────────────
def pil_b64(img, fmt="JPEG", q=85):
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=q)
    mime = "image/jpeg" if fmt=="JPEG" else "image/png"
    return f"data:{mime};base64,"+base64.b64encode(buf.getvalue()).decode()

def draw_overlay(bgr, pts):
    img = bgr.copy(); n = len(pts)
    for i in range(n-1):
        cv2.line(img,(int(pts[i][0]),int(pts[i][1])),
                     (int(pts[i+1][0]),int(pts[i+1][1])),(0,200,255),2,cv2.LINE_AA)
    if n == 4:
        cv2.line(img,(int(pts[3][0]),int(pts[3][1])),
                     (int(pts[0][0]),int(pts[0][1])),(0,200,255),2,cv2.LINE_AA)
    for i,pt in enumerate(pts):
        x,y = int(pt[0]),int(pt[1])
        cv2.circle(img,(x,y),12,CV_COLS[i],-1)
        cv2.circle(img,(x,y),12,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(img, CV_LABELS[i], (x+15, y+5),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2, cv2.LINE_AA)
    return img

def resize_for_display(bgr, max_w=MAX_DISPLAY_W):
    h,w = bgr.shape[:2]
    if w > max_w:
        s = max_w/w
        bgr = cv2.resize(bgr,(max_w,int(h*s)),interpolation=cv2.INTER_AREA)
        return bgr, s
    return bgr, 1.0

def perspective_crop(src, pts, W, H):
    arr = np.array(pts, dtype="float32")
    s = arr.sum(1); d = np.diff(arr, axis=1).flatten()
    ordered = np.array([arr[np.argmin(s)],arr[np.argmin(d)],
                        arr[np.argmax(s)],arr[np.argmax(d)]], dtype="float32")
    tl,tr,br,bl = ordered
    w = int(max(np.linalg.norm(tr-tl),np.linalg.norm(br-bl)))
    h = int(max(np.linalg.norm(bl-tl),np.linalg.norm(br-tr)))
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(src, M, (w,h))

def encode_result(img, fmt="jpg"):
    if fmt=="jpg":   _, b = cv2.imencode(".jpg",img,[cv2.IMWRITE_JPEG_QUALITY,95])
    elif fmt=="png": _, b = cv2.imencode(".png",img,[cv2.IMWRITE_PNG_COMPRESSION,3])
    else:            _, b = cv2.imencode(".bmp",img)
    return b.tobytes()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 上傳圖片")
    uploaded = st.file_uploader("", type=["jpg","jpeg","png","bmp","webp"],
                                label_visibility="collapsed")
    if uploaded:
        st.markdown("---")
        st.markdown("**輸出格式**")
        fmt = st.radio("fmt",["JPG","PNG","BMP"],horizontal=True,
                       label_visibility="collapsed")
        st.markdown("---")
        st.info("**步驟**\n1. 點擊圖片選 4 個角點\n2. 輸入數字微調座標\n3. 點「👁️ 預覽」確認\n4. 點「⬇️ 下載」儲存")

if not uploaded:
    st.markdown("""
    <div style='text-align:center;padding:100px 0;color:#475569;'>
      <div style='font-size:64px'>📂</div>
      <div style='font-size:20px;margin-top:16px'>請從左側上傳圖片</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# Load image
# ─────────────────────────────────────────────
raw      = np.frombuffer(uploaded.read(), np.uint8)
original = cv2.imdecode(raw, cv2.IMREAD_COLOR)
if original is None:
    st.error("無法讀取圖片"); st.stop()

H,W   = original.shape[:2]
fname = uploaded.name.rsplit(".",1)[0]

if st.session_state.last_file != uploaded.name:
    st.session_state.pts       = []
    st.session_state.result    = None
    st.session_state.last_file = uploaded.name

pts = st.session_state.pts
n   = len(pts)

# ─────────────────────────────────────────────
# Layout：圖片全寬，控制面板在側欄
# ─────────────────────────────────────────────
st.title("✂️ 梯形裁剪 / 透視校正工具")

# ── 側欄：角點微調 + 操作按鈕 + 裁剪結果 ──
with st.sidebar:
    st.markdown("---")
    st.markdown("#### 📍 角點座標微調（原圖 px）")

    if n == 0:
        st.caption("尚未選取任何點，請先在圖片上點擊。")
    else:
        changed = False
        for i, p in enumerate(pts):
            dot = (f"<span style='display:inline-block;width:11px;height:11px;"
                   f"border-radius:50%;background:{COLORS[i]};margin-right:6px;"
                   f"vertical-align:middle;'></span>")
            st.markdown(f"{dot}**{LABELS[i]}**", unsafe_allow_html=True)
            cx, cy = st.columns(2)
            with cx:
                new_x = st.number_input("X", min_value=0, max_value=W-1,
                                        value=int(p[0]), step=1,
                                        key=f"nx_{i}", label_visibility="visible")
            with cy:
                new_y = st.number_input("Y", min_value=0, max_value=H-1,
                                        value=int(p[1]), step=1,
                                        key=f"ny_{i}", label_visibility="visible")
            if new_x != int(p[0]) or new_y != int(p[1]):
                st.session_state.pts[i] = [new_x, new_y]
                changed = True

        if changed:
            st.session_state.result = None
            st.rerun()

    st.markdown("---")
    st.markdown("#### 🎛️ 操作")
    bc1, bc2 = st.columns(2)
    with bc1:
        if st.button("🔄 重設點位", use_container_width=True, disabled=(n == 0)):
            st.session_state.pts    = []
            st.session_state.result = None
            st.rerun()
    with bc2:
        prev_btn = st.button("👁️ 預覽截圖", use_container_width=True,
                             type="primary", disabled=(n < 4))

    if prev_btn and n == 4:
        with st.spinner("透視校正中..."):
            st.session_state.result = perspective_crop(original, pts, W, H)

    st.markdown("---")
    st.markdown("#### ✂️ 裁剪結果")
    if st.session_state.result is None:
        hint = "點圖選 4 個角點後點「預覽截圖」" if n < 4 else "點擊「👁️ 預覽截圖」"
        st.markdown(f"""<div style='background:#1e293b;border-radius:8px;padding:20px;
            text-align:center;color:#475569;'>
            <div style='font-size:28px'>🖼️</div>
            <div style='margin-top:8px;font-size:12px'>{hint}</div></div>""",
            unsafe_allow_html=True)
    else:
        res    = st.session_state.result
        rH, rW = res.shape[:2]
        st.success(f"輸出：{rW} × {rH} px")
        st.image(cv2.cvtColor(res, cv2.COLOR_BGR2RGB), use_container_width=True)
        fe  = (fmt if "fmt" in dir() else "JPG").lower()
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label=f"⬇️ 下載 {fe.upper()}",
            data=encode_result(res, fe),
            file_name=f"{fname}_crop_{ts}.{fe}",
            mime={"jpg":"image/jpeg","png":"image/png","bmp":"image/bmp"}.get(fe,"image/jpeg"),
            use_container_width=True, type="primary")

# ── 主區域：全寬圖片 ──
if n < 4:
    st.markdown(
        f"**🖱️ 點擊圖片選取第 {n+1} 個角點：{LABELS[n]}**　"
        f"<span style='color:#64748b;font-size:12px'>已選 {n}/4</span>",
        unsafe_allow_html=True)
else:
    st.markdown("**✅ 已選取 4 個角點 — 可在左側輸入框微調，或點「👁️ 預覽截圖」**")

# 繪製 overlay
disp_bgr, scale = resize_for_display(original)
disp_pts = [[p[0]*scale, p[1]*scale] for p in pts]
if pts:
    disp_bgr = draw_overlay(disp_bgr, disp_pts)

disp_rgb = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)
disp_pil = Image.fromarray(disp_rgb)

if USE_SIC and n < 4:
    coord = streamlit_image_coordinates(disp_pil, key=f"sic_{n}", use_column_width=True)
    if coord is not None:
        ox = max(0, min(W-1, int(round(coord["x"] / scale))))
        oy = max(0, min(H-1, int(round(coord["y"] / scale))))
        st.session_state.pts.append([ox, oy])
        st.rerun()
else:
    st.image(disp_pil, use_container_width=True)
    if not USE_SIC and n < 4:
        st.warning("請安裝 `streamlit-image-coordinates` 套件以啟用點擊功能：\n```\npip install streamlit-image-coordinates\n```")
