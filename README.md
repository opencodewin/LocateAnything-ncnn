# LocateAnything-ncnn

基于 [ncnn](https://github.com/Tencent/ncnn) 的 LocateAnything-3B（grounding VLM）推理实现。它是独立工程，仅从 `ncnn_llm` 拷入了该模型运行所需的最小代码集。

## 特性
- 支持动态输入推理（接近原始项目），拆分为 6 个子图：vision_embed / vision_encoder / vision_projector / text_embed / text_decoder(KV) / lm_head。
- 支持 MTP 并行窗口解码 + 结构化坐标 token（`<box><x1><y1><x2><y2></box>`）；也支持纯逐 token AR 解码（`--no-mtp`，两者输出结果一致）。
- 图像处理严格按照原始项目中的 `MAX_DIM=1024`：最大边超过 1024 时用 LANCZOS 预缩后，再做自适应 grid。
- `--vulkan-device <idx>` 支持多 GPU 选卡；`--fp16` 切换推理精度（默认 fp32）。

## 已知问题

### 平台 × 精度状态

| 后端 | 权重 | 推理 | 结果 |
| --- | --- | --- | --- |
| CPU · Apple Silicon (macOS) | fp16 | fp32 | ✅ 正确 |
| CPU · Intel x86 (Windows) | fp16 | fp32 | ✅ 正确 |
| CPU · AMD x86 (Linux) | fp16 | fp32 | ❌ 无输出：ncnn **SDPA 的 x86 实现**在 AMD 上算错（与 AVX512 / Packed Mat 无关） |
| Vulkan · NVIDIA (1080Ti/5090) | fp16 | fp16 | ❌ 发散（两者一致，故不单是「无 FP16 算术」所致） |
| Vulkan · NVIDIA | fp16 | fp32 | ⚠️ 超显存（原因未知） |
| Vulkan · NVIDIA | fp32 | — | ❓ 未测试 |
| Vulkan · Apple M1 Pro (MoltenVK) | fp16 | fp16/fp32 | ❌ 全错，一致性仅约 4.6% |

> CPU 无 FP16 硬件，其推理精度恒为 fp32。

### 已修复
ncnn ROPE/RotaryEmbed 的 Vulkan 实现 —— 上游 [PR #6834](https://github.com/Tencent/ncnn/pull/6834) 修复全宽 `cos/sin` 缓存（2D / vision RoPE），以 patch 形式构建时自动应用。注意其未覆盖 `src/layer/x86/*`；对本工程输入 CPU 侧是 no-op（真正修复的是 Vulkan 缓存步长）。

### AMD CPU (x86, Linux)
同一 x86 代码在 Windows Intel CPU 正常、AMD CPU 无任何输出。已排除：RoPE 补丁（CPU 侧 no-op）、Packed Mat 布局（宿主已自动解包）。**已明确根因：ncnn SDPA（scaled dot product attention）的 x86 实现存在 bug，在 AMD CPU 上算错**，与是否支持 AVX512 无关。进一步定位见 [跨平台一致性排查](#跨平台一致性排查)。

## 目录结构
```
3rdparty/ncnn   # submodule：ncnn 源码，构建时自动应用 patches/
patches/        # 按前缀数字排序，逐个 git apply（幂等，已应用会跳过）
src/            # 运行时 + tokenizer + image_utils + json
models/         # 外部模型目录，通过 --model 引用，不随工程提交
```

## 模型
模型提供 fp16 与 fp32 两种动态导出，下载后通过 `--model` 目录引用：
- fp32 动态：[`locate-anything-fp32`](通过网盘分享的文件：locate-anything-fp32
链接: https://pan.baidu.com/s/1qXqw-c4yRUOyMuUuqwqtUQ?pwd=ybfk 提取码: ybfk 
--来自百度网盘超级会员v7的分享)
- fp16 动态：[`locate-anything-fp16`](https://www.modelscope.cn/models/sizeofbeer/locate-anything-fp16)

## 依赖 / 构建
```bash
git submodule update --init --recursive     # 首次
cmake -G Ninja -B build -S . -DCMAKE_BUILD_TYPE=Release   # 需 Git 以自动应用 patches/
cmake --build build
```
默认启用 Vulkan，`-DLOCATE_NCNN_VULKAN=OFF` 关闭。

## 运行
```bash
./build/locate_main --model <fp32|fp16 模型目录> --image <img> --prompt <query> [选项]
```
- `--vulkan`、`--vulkan-device <idx>`：启用 GPU 与多卡选卡（越界回退 0）。
- `--fp16`：推理精度，仅作用于 Vulkan 文本链；视觉链与 CPU 恒为 fp32。
- `--threads N`、`--greedy`、`--max-new-tokens N`。
- `--save <out.png>`：画框保存（默认 `<image>_locate.png`）；`--no-draw` 关闭。

输出除结构化坐标 token 外，会逐框打印归一化/像素坐标；像素 = 归一化 × 原图宽高（`src/utils/draw_utils.h`）。

## 平台测试
`bench_platform` 对同一输入按 cpu-fp32 → gpu-fp32 → gpu-fp16 顺序运行，统计各平台端到端耗时，并以 cpu-fp32 输出为参考衡量文本一致性（归一化 Levenshtein，1=完全一致）。greedy 确定性解码；每平台 warmup 1 次后测 N 次。

```bash
cmake --build build --target bench_platform
./build/bench_platform --image ../datas/football.jpg --prompt human --max-new-tokens 20 --iter 1  # 快速验证
./build/bench_platform --image ../datas/football.jpg --prompt human --iter 3                     # 正式统计
```
可选：`--threads N`、`--max-new-tokens N`、`--vulkan-device <idx>`、`--no-mtp`（纯 AR）、`--cpu-only`（跳过已知不可用的 Vulkan）、`--save-dir <dir>`（每配置存一张带框图）。

**注意**：
- 一致性只衡量平台输出是否彼此相同，不代表检出全部目标（`datas/football.jpg` 有 **8 人**）；检测质量需另用标注数据评估。
- MTP 与 AR 结果一致；框数差异来自 token 预算截断（一个框恰 6 token，`max_new=20` 仅够 3 框）。跨平台/跨模式对比时 `--max-new-tokens` 须**一致且给足**。

## 跨平台一致性排查（某平台无输出 / 结果发散时）
运行时会打印可直接跨平台比对的「指纹」，用来定位发散发生在哪一级：

| 输出 | 含义 |
| --- | --- |
| `SHAPE <子图.输出> dims=.. w=.. h=.. c=.. pack=..` | 子图输出的 Mat 布局。**`pack` 必须 = 1**；非 1 时会额外打印 `NOTE ... unpack packN -> pack1`，宿主已自动解包（x86 AVX512 等架构上 ncnn 可能返回 pack4/pack8，未解包会导致坐标全错） |
| `DEBUG feat w=.. h=.. min=.. max=.. mean=..` | 视觉特征（vision_projector 输出）统计 |
| `FEATSUM` | 视觉特征求和，跨平台应逐位一致 |
| `PREFILL top5 logits(V=..): id:logit ...` | prompt 末位 top-5 logits，跨平台应逐位一致 |

更深入的逐元素对比（md5 或 numpy 比对）：
```bash
LA_DUMP_VISION=/tmp/feat.f32 LA_DUMP_TEXT=/tmp/pre \
  ./locate_main --model <模型目录> --image ../datas/football.jpg --prompt human --max-new-tokens 20 --threads 8
md5sum /tmp/feat.f32 /tmp/pre_hidden.f32 /tmp/pre_logits.f32
```
- `LA_DUMP_VISION=<path>`：投影后的视觉特征 `[n_tokens, hidden]` fp32。
- `LA_DUMP_TEXT=<prefix>`：prefill 末位 hidden（`<prefix>_hidden.f32`）与 logits（`<prefix>_logits.f32`）。

**基准指纹**（macOS M1 Pro，CPU fp32，`models/locate-anything-fp16` + `datas/football.jpg` + `--prompt human --max-new-tokens 20`）：
```
DEBUG feat w=2048 h=925 min=-29.245 max=14.516 mean=-0.0010
FEATSUM -1853.253568
PREFILL top5 logits(V=152681): 151672:23.3104 151645:8.5655 151668:8.5340 151741:7.0714 152191:6.9475
md5 feat.f32        = 98fa1daf78da0786cbd7d1e05e504f87
md5 pre_hidden.f32  = 7fd2130dd0dae61da7b94ecb6e0b65c6
md5 pre_logits.f32  = 7f7446504468f4712cfeb3b0384de00a
```
**判定**：
- `FEATSUM` 不一致 → 视觉链（vision_embed / vision_encoder / vision_projector 或宿主 pos_emb 插值 / moon RoPE）发散。
- `FEATSUM` 一致但 `PREFILL top5` 不一致 → 文本链（text_embed / decoder prefill / lm_head）发散。
- 两种都一致但结果仍错 → 问题在生成（解码循环 / MTP / 采样），非前向骨架。