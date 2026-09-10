# LocateAnything-ncnn

基于 [`ncnn`](https://github.com/Tencent/ncnn) 的 LocateAnything-3B（grounding VLM）推理实现。它是独立工程，仅从 [`ncnn_llm`](https://github.com/futz12/ncnn_llm) 拷入了该模型运行所需的最小代码集。

## 特性
- 支持动态输入推理（接近原始项目），拆分为 6 个子图：vision_embed / vision_encoder / vision_projector / text_embed / text_decoder(KV) / lm_head。
- 支持 MTP 并行窗口解码 + 结构化坐标 token（`<box><x1><y1><x2><y2></box>`）；也支持纯逐 token AR 解码（`--no-mtp`，两者输出结果一致）。
- 图像处理严格按照原始项目中的 `MAX_DIM=1024`：最大边超过 1024 时用 LANCZOS 预缩后，再做自适应 grid。
- `--vulkan-device <idx>` 支持多 GPU 选卡；`--fp16` 启用 2 字节存储（bf16 优先 / fp16 回退），计算恒为 fp32（默认全 fp32）；`--weights-in-host`（仅离散 GPU / 非 mac）把模型权重 offload 到系统内存，解决 fp32 超显存。

## 已知问题

### 平台 × 精度状态

| 后端 | 权重 | 存储 / 计算 | 结果 |
| --- | --- | --- | --- |
| CPU · Apple Silicon (macOS) | fp16 | fp32 计算 | ✅ 正确 |
| CPU · Intel x86 (Windows) | fp16 | fp32 计算 | ✅ 正确 |
| CPU · AMD x86 (Linux) | fp16 | fp32 计算 | ❌ 无输出：ncnn **SDPA 的 x86 实现**在 AMD 上算错（与 AVX512 / Packed Mat 无关） |
| Vulkan · NVIDIA | fp16/fp32 | fp32 存储 · fp32 计算 | ✅ 正确（MTP 与 `--no-mtp` 均已实测） |
| Vulkan · NVIDIA (1080Ti) | fp16/fp32 | fp16 存储回退（`--fp16`）· fp32 计算 | ❌ **仍发散** `<ref>!!!…`（经反复关闭 fp16 算术后确认：**Pascal 无 bf16 存储，fp16 存储回退在这张卡上就发散**，非算术问题） |
| Vulkan · NVIDIA (5090) | fp16/fp32 | bf16 存储（`--fp16`）· fp32 计算 | ⸺ **待回归**（方案已落地，bf16 保留 fp32 指数范围；此路径的预期正确） |
| Vulkan · Apple M1 Pro (MoltenVK) | fp16/fp32 | fp16/bf16 存储（`--fp16`）· fp32 计算 | ⸺ 待回归 |

> CPU 无 FP16 硬件，其推理精度恒为 fp32。
> **`--fp16` 的语义**：fp16 **算术**已在 `create_option()` 按 zimage-ncnn-vulkan 方案**关闭**（计算恒 fp32），`--fp16` 仅启用 2 字节**存储**——bf16 优先（无 bf16 存储时回退 fp16）。出发点：fp16 算术在 SDPA softmax / 长序列累加上塌缩或溢出（早期 1080Ti/M1 Pro 均复现 `<ref>!!!…`）。
> **实测结论（1080Ti，device1）**：**fp32 Vulkan（fp32 存储 + fp32 计算，含/不含 `--weights-in-host`、MTP 开/关）全部正确**，仅 CPU；之前一度误报「纯 fp32 也发散」，经二分（195846e→6ef60d8→2c09da2→HEAD）复核为异常/脏状态，**非工程变更所致**。真正的剩余问题只有 **fp16 存储**：在无 bf16 存储的 Pascal 1080Ti 上，fp16 存储回退即使算术全 fp32 也发散 → 该卡无可用 2 字节方案，只能回 fp32 存储；**bf16 存储路径（5090）为预期正确方案，回归待 5090 验证**。

### 已修复
ncnn ROPE/RotaryEmbed 的 Vulkan 实现 —— 上游 [PR #6834](https://github.com/Tencent/ncnn/pull/6834) 修复全宽 `cos/sin` 缓存（2D / vision RoPE），以 patch 形式构建时自动应用。注意其未覆盖 `src/layer/x86/*`；对本工程输入 CPU 侧是 no-op（真正修复的是 Vulkan 缓存步长）。

### AMD CPU (x86, Linux)
同一 x86 代码在 Windows Intel CPU 正常、AMD CPU 无任何输出。已排除：RoPE 补丁（CPU 侧 no-op）、Packed Mat 布局（宿主已自动解包）。**已明确根因：ncnn SDPA（scaled dot product attention）的 x86 实现存在 bug，在 AMD CPU 上算错**，与是否支持 AVX512 无关。注意此为**前向数值错误**（输出全错/无框），与下方 Vulkan fp16 的**精度发散**属两类不同问题。

## 目录结构
```
3rdparty/ncnn   # submodule：ncnn 源码，构建时自动应用 patches/
patches/        # 按前缀数字排序，逐个 git apply（幂等，已应用会跳过）
src/            # 运行时 + tokenizer + image_utils + json
models/         # 外部模型目录，通过 --model 引用，不随工程提交
```

## 模型
模型提供 fp16 与 fp32 两种动态导出，下载后通过 `--model` 目录引用：
- fp32 动态：[`locate-anything-fp32`](https://pan.baidu.com/s/1qXqw-c4yRUOyMuUuqwqtUQ?pwd=ybfk)（百度网盘，提取码 `ybfk`）
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
- `--vision-vulkan`：让视觉 3 子图（vision_embed / vision_encoder / vision_projector）也走 Vulkan；默认随 `--vulkan` 一致。⚠️ 须 `--vulkan` 已开启，否则无效（Vulkan 设备未初始化，视觉仍 CPU）。当前 MoltenVK 上 6 张子图同载 Vulkan 会 `VK_ERROR_DEVICE_LOST`，故 Apple 平台暂不可用，仅作开关预留。
- `--fp16`：启用 2 字节存储（bf16 优先，设备无 bf16 时回退 fp16），**同时作用于 Vulkan 文本链与视觉链**（二者一致）；计算恒为 fp32。默认全 fp32 存储 + fp32 计算。见「已知问题 / Vulkan fp16 发散」。
- `--threads N`、`--greedy`、`--max-new-tokens N`。
- `--weights-in-host`（仅离散 GPU）：权重 offload 到系统内存，缓解设备显存不足。
- `--save <out.png>`：画框保存（默认 `<image>_locate.png`）；`--no-draw` 关闭。

输出除结构化坐标 token 外，会逐框打印归一化/像素坐标；像素 = 归一化 × 原图宽高（`src/utils/draw_utils.h`）。

## 平台测试
`bench_platform` 对同一输入按 cpu-fp32 → gpu-fp32 → gpu-fp16 顺序运行，统计各平台端到端耗时，并以 cpu-fp32 输出为参考衡量文本一致性（归一化 Levenshtein，1=完全一致）。greedy 确定性解码；每平台 warmup 1 次后测 N 次。

```bash
cmake --build build --target bench_platform
./build/bench_platform --image ../datas/football.jpg --prompt human --max-new-tokens 20 --iter 1  # 快速验证
./build/bench_platform --image ../datas/football.jpg --prompt human --iter 3                     # 正式统计
```
可选：`--threads N`、`--max-new-tokens N`、`--vulkan-device <idx>`、`--no-mtp`（纯 AR）、`--cpu-only`（跳过已知不可用的 Vulkan）、`--vision-vulkan`（视觉 3 子图也走 Vulkan，须 `--vulkan` 已开启、否则对各 GPU 配置无效）、`--weights-in-host`（仅离散 GPU / 非 mac，权重 offload 到系统内存以跑通 Vulkan fp32）、`--save-dir <dir>`（每配置存一张带框图）。

**注意**：
- 一致性只衡量平台输出是否彼此相同，不代表检出全部目标（`datas/football.jpg` 有 **8 人**）；检测质量需另用标注数据评估。
- MTP 与 AR 结果一致；框数差异来自 token 预算截断（一个框恰 6 token，`max_new=20` 仅够 3 框）。跨平台/跨模式对比时 `--max-new-tokens` 须**一致且给足**。

## Vulkan fp16 发散（定位小结与修复）
早期复现：Vulkan 下 **fp16 算术**在 NVIDIA（1080Ti/5090）与 Apple M1 Pro 上均发散（坐标 token 全 0 → `<ref>!!!…`），而 fp32 计算被认为正确（selfcheck maxdiff=0，解码与 CPU 一致）。定位过程排除项：
- 非 Flash Attention（fp16 下强制禁用 flash，结果逐位不变）。
- 非 RoPE（PR #6834 已 patch，fp16 仍发散）。
- 非「仅文本 decoder」问题：视觉前向在 Vulkan fp16 下同样发散。
- 与权重/存储精度无关（fp16/fp32 权重行为相同）。

**结论与修正**：早期定位为 **fp16 数值管线**发散。修复（已落地，对齐 zimage-ncnn-vulkan）：在 `src/ncnn_llm_base.h` 的 `create_option()` 中关闭 `use_fp16_arithmetic`（恒 false），计算一律 fp32；`--fp16` 仅作 2 字节存储——bf16 优先，设备无 bf16 存储（如 Pascal 1080Ti）回退 fp16 存储。**最新实测**：fp32 Vulkan（含/不含 `--weights-in-host`、MTP 开/关）在 1080Ti 全部正确（先前误报「纯 fp32 也发散」为异常状态，二分复核推翻）；剩余问题只在 **fp16 存储回退**——1080Ti 上即使算术全 fp32，`--fp16`（fp16 存储）仍发散，说明该 Pascal 卡无可用 2 字节方案，只能回 fp32 存储；**bf16 存储路径（5090）为预期正确方案，回归待 5090 验证**（`--vulkan --fp16 --weights-in-host` + `bench_platform` 对照 cpu-fp32）。