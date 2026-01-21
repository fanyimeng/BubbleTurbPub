import streamlit as st
import pandas as pd
import os
import shutil
import sys
import time
from bubturb import export_subcube_v3, bubble_pv_plot_v9
from astropy.coordinates import SkyCoord
from astropy import units as u
from spectral_cube import SpectralCube
from datetime import datetime

# ========== CONFIG UI ==========
st.set_page_config(layout="wide")
st.markdown("""
<style>
/* 全局默认字体稍大 */
html, body, [class*="css"] {
    font-size: 20px !important;
}

/* text_input 样式 */
input[type="text"] {
    font-size: 32px !important;
    padding: 10px 16px !important;
    height: 1.5em !important;
}

/* number_input 样式 */
input[type="number"] {
    font-size: 32px !important;
    padding: 10px 16px !important;
    height: 3.5em !important;
}

/* label 样式 */
label {
    font-size: 28px !important;
    font-weight: 700 !important;
}

/* 按钮整体样式 */
button[kind="primary"], div.stButton > button {
    font-size: 32px !important;
    line-height: 1.2 !important;
    padding: 16px 12px !important;
    min-height: 2em !important;
}

/* 按钮内部 span 字体 */
div.stButton button span {
    font-size: 32px !important;
    font-weight: bold !important;
    line-height: 1.2 !important;
}

/* 按钮内容居中 */
div.stButton button {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}
</style>
""", unsafe_allow_html=True)

# ========== PATH CONFIG ==========
DATA_TAB = "0407-sorted.tab"
FITS_PATH = "../data/jcomb_vcube_scaled2.fits"
SUBCUBE_PATH = "/Users/meng/alex/astro/m31/00bubble/subcube"
ORIGINAL_SUBCUBE_PATH = "/Users/meng/alex/astro/m31/00bubble/subcube_0407"
PVPLOT_DIR = "./pvplots"
TEMP_IMAGE = os.path.join(PVPLOT_DIR, "temp.png")

os.makedirs(SUBCUBE_PATH, exist_ok=True)
os.makedirs(PVPLOT_DIR, exist_ok=True)

mytime = '%y%m%d-%H%M'

def to_fwf(df, fname):
    from tabulate import tabulate
    df = df.copy()
    for col in ['maj_as', 'min_as', 'pa', 'vmin_kms', 'vmax_kms', 'v1', 'v2', 'v_ang']:
        if col in df.columns:
            df[col] = df[col].astype(int)
    content = tabulate(df.values.tolist(), list(df.columns), tablefmt="plain")
    with open(fname, "w") as f:
        f.write(content)
pd.DataFrame.to_fwf = to_fwf

def est_ellipse(row):
    c = SkyCoord(f"{row['ra_hms']} {row['dec_dms']}", unit=(u.hourangle, u.deg))
    return {
        'id': row['id'],
        'ra': c.ra.value,
        'dec': c.dec.value,
        'major': int(row['maj_as']),
        'minor': int(row['min_as']),
        'pa': int(row['pa']),
        'velocities': [int(row['vmin_kms']), int(row['vmax_kms'])],
        'collide': row['collide'],
        'v1': int(row['v1']),
        'v2': int(row['v2']),
        'v_ang': int(row['v_ang']),
        'good': str(row['good'])
    }

def plot_and_export(cube, ellipse):
    rating_str = ellipse['good']
    reg_id = f"{int(ellipse['id'])} ({rating_str})"
    subcube, _ = export_subcube_v3(cube, ellipse, scale_factor=2.2, v_plot_factor=2, output_file=None)
    bubble_pv_plot_v9(
        subcube, 38, plot_paths=True,
        plot_name=TEMP_IMAGE,
        ellipse=ellipse, reg_id=reg_id,
        collide=ellipse['collide'],
        reg_file='b86_J2000.reg', plot_extra_ellipses=True
    )
    for _ in range(10):
        if os.path.exists(TEMP_IMAGE) and os.path.getsize(TEMP_IMAGE) > 0:
            break
        time.sleep(0.1)
    save_path = os.path.join(PVPLOT_DIR, f"pvPlot-{int(ellipse['id']):03d}.png")
    shutil.copyfile(TEMP_IMAGE, save_path)
    st.success(f"图像保存至 {save_path} ✅")

def show_temp_image():
    for _ in range(10):
        try:
            if os.path.exists(TEMP_IMAGE) and os.path.getsize(TEMP_IMAGE) > 0:
                with open(TEMP_IMAGE, 'rb') as f:
                    st.image(f.read(), use_column_width=True)
                return
        except Exception:
            pass
        time.sleep(0.1)
    st.warning("⚠️ 图像未能加载成功，请重试 Replot。")

# ========== DATA LOAD ==========
try:
    df = pd.read_fwf(DATA_TAB, header=0, infer_nrows=int(1e6))
    if df.empty:
        st.error(f"读取失败：{DATA_TAB} 是空的")
        st.stop()
except pd.errors.EmptyDataError:
    st.error(f"读取失败：{DATA_TAB} 是空文件")
    st.stop()
    
for col in ['collide', 'in_b86']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower().map({'true': 1, 'false': 0, '1': 1, '0': 0}).fillna(0).astype(int)
if 'good' not in df.columns:
    df['good'] = '??'
if 'date' not in df.columns:
    df['date'] = datetime.today().strftime(mytime)
if not df.empty:
    df.to_fwf(DATA_TAB)
else:
    st.warning("⚠️ 当前 DataFrame 是空的，未写入文件")

id_list = df['id'].tolist()
if 'idx' not in st.session_state:
    st.session_state.idx = id_list[0]
curr_id = st.session_state.idx
row = df[df['id'] == curr_id].iloc[0]
cube_path = os.path.join(ORIGINAL_SUBCUBE_PATH, f"pv-original-{int(curr_id):03d}.fits")
cube = SpectralCube.read(cube_path, format='fits')

left, right = st.columns([0.2, 0.8])
with left:
    st.markdown(f"### ID: {curr_id}")
    ra_hms = st.text_input("ra_hms", value=row['ra_hms'])
    dec_dms = st.text_input("dec_dms", value=row['dec_dms'])
    col_maj, col_min = st.columns(2)
    maj_as = col_maj.number_input("maj_as", value=int(row['maj_as']), step=1, format="%d")
    min_as = col_min.number_input("min_as", value=int(row['min_as']), step=1, format="%d")
    pa = st.number_input("pa", value=int(row['pa']), step=1, format="%d")
    col_vmax, col_v2 = st.columns(2)
    vmax_kms = col_vmax.number_input("vmax_kms", value=int(row['vmax_kms']), step=1, format="%d")
    v2 = col_v2.number_input("v2", value=int(row['v2']), step=1, format="%d")
    col_vmin, col_v1 = st.columns(2)
    vmin_kms = col_vmin.number_input("vmin_kms", value=int(row['vmin_kms']), step=1, format="%d")
    v1 = col_v1.number_input("v1", value=int(row['v1']), step=1, format="%d")
    col_vang, col_col = st.columns(2)
    v_ang = col_vang.number_input("v_ang", value=int(row['v_ang']), step=1, format="%d")
    collide = col_col.selectbox("collide", options=[0, 1], index=int(row['collide']))
    in_b86 = None
    if 'in_b86' in df.columns:
        in_b86 = st.selectbox("in_b86", options=[0, 1], index=int(row['in_b86']))

    def save_and_plot(rating):
        old_row = df[df['id'] == curr_id].iloc[0]
        changes = []
        fields = ['ra_hms', 'dec_dms', 'maj_as', 'min_as', 'pa',
                  'vmin_kms', 'vmax_kms', 'v1', 'v2', 'v_ang', 'collide']
        new_values = [ra_hms, dec_dms, maj_as, min_as, pa, vmin_kms, vmax_kms, v1, v2, v_ang, int(collide)]

        for field, new_val in zip(fields, new_values):
            old_val = old_row[field]
            if str(old_val) != str(new_val):
                changes.append(f"{field}: {old_val} → {new_val}")

        df.loc[df['id'] == curr_id, fields] = new_values
        df.loc[df['id'] == curr_id, 'date'] = datetime.today().strftime(mytime)
        df.loc[df['id'] == curr_id, 'good'] = rating
        if in_b86 is not None:
            df.loc[df['id'] == curr_id, 'in_b86'] = int(in_b86)
        if not df.empty:
            df.to_fwf(DATA_TAB)
        else:
            st.warning("⚠️ 当前 DataFrame 是空的，未写入文件")

        if changes:
            st.info("已修改以下字段:\n" + "\n".join(changes))
        else:
            st.info("无参数变化，仅保存评级。")

    colg1, colg2, colg3, colg4 = st.columns(4)
    if colg1.button("🟢 pp"):
        save_and_plot("pp")
    if colg2.button("🔵 vv"):
        save_and_plot("vv")
    if colg3.button("🟡 pv"):
        save_and_plot("pv")
    if colg4.button("🔴 nn"):
        save_and_plot("nn")

with right:
    show_temp_image()

    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1, 1, 1])
    if col1.button("🔁 Replot"):
        ellipse = est_ellipse(row)
        plot_and_export(cube, ellipse)
        st.rerun()
    if col2.button("💾 保存subcube"):
        ellipse = est_ellipse(row)
        subcube_path = os.path.join(SUBCUBE_PATH, f'pv-original-{int(curr_id):03d}.fits')
        export_subcube_v3(cube, ellipse, scale_factor=3, v_plot_factor=2., output_file=subcube_path)
        st.success("subcube 已保存 ✅")
    if col3.button("⬅️ 上一个"):
        idx = id_list.index(curr_id)
        if idx > 0:
            prev_id = id_list[idx - 1]
            existing = os.path.join(PVPLOT_DIR, f"pvPlot-{int(prev_id):03d}.png")
            if os.path.exists(existing):
                shutil.copyfile(existing, TEMP_IMAGE)
            st.session_state.idx = prev_id
            st.rerun()
    if col4.button("➡️ 下一个"):
        idx = id_list.index(curr_id)
        if idx + 1 < len(id_list):
            next_id = id_list[idx + 1]
            existing = os.path.join(PVPLOT_DIR, f"pvPlot-{int(next_id):03d}.png")
            if os.path.exists(existing):
                shutil.copyfile(existing, TEMP_IMAGE)
            st.session_state.idx = next_id
            st.rerun()
    sel_id = col5.selectbox(
    "选择ID",  # 👈 用一个有意义的 label
    options=id_list,
    index=id_list.index(curr_id),
    label_visibility="collapsed"  # 👈 隐藏标签，但保持语义
)
    if sel_id != curr_id:
        st.session_state.idx = sel_id
        st.rerun()
    if col6.button("🕒 跳到最近修改"):
        latest_id = df.sort_values("date", ascending=False)['id'].iloc[0]
        st.session_state.idx = latest_id
        st.rerun()
    if col7.button("📂 载入已有图像"):
        existing = os.path.join(PVPLOT_DIR, f"pvPlot-{int(curr_id):03d}.png")
        if os.path.exists(existing):
            shutil.copyfile(existing, TEMP_IMAGE)
            st.rerun()

with st.expander("📁 路径信息（点击展开）"):
    st.text(f"数据表: {DATA_TAB}")
    st.text(f"主 FITS: {FITS_PATH}")
    st.text(f"原始 subcube 路径: {ORIGINAL_SUBCUBE_PATH}")
    st.text(f"subcube 保存路径: {SUBCUBE_PATH}")
    st.text(f"PV 图保存路径: {PVPLOT_DIR}")
