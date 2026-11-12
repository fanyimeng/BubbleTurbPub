# M31 气泡可视化：图件来源与核对记录

[返回中文开发规格](README.zh-CN.md) · [English figure provenance](figure-sources.md) · v0.2 · 2026 年 11 月 12 日

文档中的 WebP 图件随仓库提供。下列 DualVectra 路径是历史资产的相对来源标识，用于核对生成记录；阅读本文和查看配图无需访问 DualVectra。历史切块与最终目录按天空坐标和速度关联对齐，像素坐标以各自的 FITS 为准。

## 图 1｜M31 定位图与 Bubble 31

- 左图由 FAST 单独观测的积分强度 FITS `fast-rg_mom0_vminus700_to_minus30.fits` 绘制（SHA-256 `6233da69eb89bafd07afba8b9f87211b6c02c196422e4edb35ff6732509a3415`）。主文件头的望远镜和仪器均为 FAST，记录了对中心速度在 −700 至 −30 km/s 的通道做积分。波束 FWHM 为 174″。图件裁取星系周边区域，显示色标为 5–90 Jy beam⁻¹ km s⁻¹，采用线性拉伸。青色定位标记由 FITS 天球 WCS 投影 CAND238（Bubble 31）的目录中心得到。
- **概念模型：** `outputs/bubble31_blender/bubble31_shell_macro_v4_preview.png`，生成脚本 `script/build_bubble31_shell_geometry.py`。投影强度引导局部表面细节和透明度；壳深、前侧窗口及灯光由模型设定。该画面用来讨论展示目标，图 2–4 的网格才按观测 VRAD 构造。

## 图 2｜Bubble 104 双路线

- **输入：** 历史 `bubble104.fits`，SHA-256 `b9a746bef8b0d2c173b4bef3c33ced7154a09a6adcd0ead4b2337d540a9184c2`。主 HDU，维度 `[89,128,128]`（速度、y、x），`Jy/beam`，60″ 波束，VRAD/LSRK，原始通道间隔约 1.649 km/s。图中三通道按顺序为源数组索引 41、37、33（0 基），即 −92.03、−85.43、−78.84 km/s。
- **A：** `outputs/nine_bubbles_recommended_v1/bubble104_recommended` 的 `_preview.png`、`_run.json`，及同名 `_assets` 下的 `_cavity_F0p30.obj`、`_cavity_manifest.json` 和 `_cavity_qa.png`。表面取 0.30，圆润化 0.80，目标显示深度 1.00。
- **B：** `outputs/all_outer_ridge_v1/bubble104_outer_ridge_closed_v1` 的 `_preview.png`、`_run.json`，及同名 `_assets` 下的 `_envelope.obj`、`_envelope_manifest.json` 和 `_envelope_qa.png`。亮脊半径先验为 (21.6, 21.6) 像素，支持阈值比例 0.18，`hybrid`，目标显示深度 1.03。
- **绘法：** 两网格均以源图 (59, 69) 像素为坐标原点，分别用 manifest 的伸缩系数和速度零点把顶点还原到源 PPV。内壁零点取所用分析窗口内真实通道速度中位数，亮脊取峰值通道速度。把三角面与三个原始通道平面求交，叠加于未作附加平滑的切片，三幅共享线性强度色标。下排拼接两张 Blender 预览；各自取景，渲染高光由材质和灯光产生。CAND021 用于当前对象索引，小样应以对齐后的目录和切块重新定位。

## 图 3｜Bubble 54 速度端部

- **输入：** 历史 `bubble54.fits`，SHA-256 `7ca9822aeafd9d5c719d60c72d5ca81d4fff9577de6e8eb7217f6fae012583c7`。主 HDU，维度 `[56,128,128]`，`Jy/beam`，60″ 波束，VRAD/LSRK，通道间隔约 1.649 km/s。历史网格中心经切块 WCS 回算约为 00h40m20.70s、+41°02′38.4″，对应最终目录 **CAND067**；该目录还另有同号的 CAND054。
- **A：** `outputs/nine_bubbles_recommended_v1/bubble54_recommended_assets/` 下的 `bubble54_recommended_cavity_F0p30.obj`、对应 `_cavity_manifest.json` 和 `_cavity_qa.png`。表面取 0.30，圆润化 0.50，目标显示深度 1.00。分割保留 −390.45、−388.81 km/s 两个原始通道中心。
- **B：** `outputs/all_outer_ridge_v1/bubble54_outer_ridge_closed_v1_assets/` 下的 `bubble54_outer_ridge_closed_v1_envelope.obj`、对应 `_envelope_manifest.json` 和 `_envelope_qa.png`。阈值比例 0.18，`hybrid`，目标显示深度 1.03。关联外扩上限为 6 个原始通道，选定支持段为 −408.59 至 −380.56 km/s，manifest 标记较高速度端受关联外扩限制；几何收口可延到该标记之外。
- **绘法：** 与图 2 相同方式还原两个网格到源 PPV。取源图 y=60（0 基）平面，将网格截线叠加到原始 PV 切片。细竖线标出原始通道中心，蓝带宽度取原始通道间隔，浅橙虚线为关联外扩约束下的亮脊支持端点。上下图使用相同线性强度色标。

## 图 4｜Bubble 32 四视图

- **原图：** `outputs/outer_ridge_final118_s3v6_v2/packages/selected_9_four_views_v1/images/` 下的 `bubble32_view_000.png`、`bubble32_view_090.png`、`bubble32_view_180.png`、`bubble32_view_270.png`。均与同包 `four_view_render_manifest.json` 的 SHA-256 一致，原尺寸均为 1100 × 1100。
- **几何和显示：** `bubble32_outer_ridge_closed_final118_v1`，CAND074，阈值比例 0.18，`hybrid`。按四视图 manifest，模型绕世界 Z（Dec 显示轴）转动，相机与灯光固定。四幅使用同一个裁切框并按同一比例缩小。该对象几何状态为 `review`：先验偏离约束在超过 10% 的方位上起作用；渲染状态为 `success`。这组图用于讨论轮廓随视角的变化。
- **批次报告：** `outputs/outer_ridge_final118_s3v6_v2/reports/final118_outer_ridge_preview_report_12_per_page.pdf` 供历史批次内定位，具体视角和源图以四视图 manifest 为准。

## 图 5｜拟议点、边缘与加密实验

使用确定性的二维模拟信号生成。三幅共用同一发射场和固定随机种子 415；第三幅保留基础点集并在有信号的壳附近增加采样。实线和虚线示意视角相关边缘的两侧；未使用观测 FITS，也未执行三维透明合成或 Unity 导入。

## 历史实现索引

|范围|DualVectra 来源标识|可核对内容|
|---|---|---|
|内壁入口与分割| `script/fits_cube_to_blend.py`；`script/build_bubble_ppv_cavity_fill.py` |空洞得分、连通域、距离场、surface/roundness/depth 及显示变换|
|亮脊入口与分割| `script/fits_cube_to_outer_ridge_blend.py`；`script/build_bubble_ppv_outer_ridge_envelope.py`；`config/outer_ridge_bubbles_v1.json` |径向搜索、支持段、窗口和关联截断、三种收口、对象参数|
|两批原型| `outputs/nine_bubbles_recommended_v1/README.md`；`outputs/all_outer_ridge_v1/README.md` |16 个内壁网格、少通道端盖诊断、17 个亮脊 component|
|final118| `config/outer_ridge_final118_v1.README.md`；`outputs/outer_ridge_final118_s3v6_v2/batch_summary.json` 和 `status.tsv` |2026 年 8 月 9 日几何与渲染状态、复核原因及逐对象续跑策略|
