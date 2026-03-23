# ✂️ 梯形裁剪工具 — Streamlit 網頁版

上傳圖片 → 滑桿調整四個角點 → 透視校正 → 下載結果

---

## 🚀 免費部署到 Streamlit Cloud（5 分鐘完成）

### 步驟一：上傳到 GitHub

1. 在 GitHub 建立新的 **public** repository（例如 `trapezoid-crop`）
2. 將以下兩個檔案上傳至 repo 根目錄：
   - `app.py`
   - `requirements.txt`

### 步驟二：部署到 Streamlit Cloud

1. 前往 [https://share.streamlit.io](https://share.streamlit.io)
2. 用 GitHub 帳號登入（免費）
3. 點「**New app**」
4. 選擇你的 repository、branch（`main`）、主程式（`app.py`）
5. 點「**Deploy!**」→ 約 1～2 分鐘後自動完成

部署完成後會取得一個公開網址，例如：
```
https://your-name-trapezoid-crop-app-xxxx.streamlit.app
```
任何人用瀏覽器開啟即可使用，**完全免費、無需安裝任何軟體**。

---

## 💻 本機執行

```bash
pip install -r requirements.txt
streamlit run app.py
```

瀏覽器會自動開啟 `http://localhost:8501`

---

## 📦 檔案說明

| 檔案 | 說明 |
|------|------|
| `app.py` | Streamlit 主程式 |
| `requirements.txt` | Python 套件清單 |

---

## 🛠️ 功能

- 📂 上傳圖片（JPG / PNG / BMP / WebP）
- 🎯 四個角點各自的 X/Y 滑桿微調
- 🖼️ 即時預覽梯形框線
- ✂️ 透視校正輸出
- ⬇️ 下載結果（JPG / PNG / BMP）
