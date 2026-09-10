# LocateAnything-ncnn

基于 [`ncnn`](https://github.com/Tencent/ncnn) 的 LocateAnything-3B（grounding VLM）推理实现。它是独立工程，仅从 [`ncnn_llm`](https://github.com/futz12/ncnn_llm) 拷入了该模型运行所需的最小代码集。

## 特性
- 支持动态输入推理（接近原始项目），拆分为 6 个子图：vision_embed / vision_encoder / vision_projector / text_embed / text_decoder(KV) / lm_head。
- 支持 MTP 并行窗口解码 + 结构化坐标 token（`<box><x1><y1><x2><y2></box>`）；也支持纯逐 token AR 解码（`--no-mtp`，两者输出结果一致）。
- 图像处理严格按照原始项目中的 `MAX_DIM=1024`：最大边超过 1024 时用 LANCZOS 预缩后，再做自适应 grid。
- `--vulkan-device <idx>` 支持多 GPU 选卡；`--fp16` 切换推理精度（默认 fp32）；`--weights-in-host`（仅离散 GPU / 非 mac）把模型权重 offload 到系统内存，解决 fp32 超显存。

## 已知问题

### 平台 × 精度状态

| 后端 | 权重 | 推理 | 结果 |
| --- | --- | --- | --- |
| CPU · Apple Silicon (macOS) | fp16 | fp32 | ✅ 正确 |
| CPU · Intel x86 (Windows) | fp16 | fp32 | ✅ 正确 |
| CPU · AMD x86 (Linux) | fp16 | fp32 | ❌ 无输出：ncnn **SDPA 的 x86 实现**在 AMD 上算错（与 AVX512 / Packed Mat 无关） |
| Vulkan · NVIDIA (1080Ti/5090) | fp16 | fp16 | ❌ 发散（两者一致，故不单是「无 FP16 算术」所致） |
| Vulkan · NVIDIA | fp16 | fp32 | ⚠️ 默认超显存；`--weights-in-host` 后 ✅ 正确 |
| Vulkan · NVIDIA | fp32 | fp16 | ❌ 发散（与 fp16 权重行为相同） |
| Vulkan · NVIDIA | fp32 | fp32 | ⚠️ 默认超显存；`--weights-in-host` 后 ✅ 正确 |
| Vulkan · Apple M1 Pro (MoltenVK) | fp16 | fp16 | ❌ 发散（selfcheck decoder maxdiff≈20；与 Flash on/off 无关） |
| Vulkan · Apple M1 Pro (MoltenVK) | fp16 | fp32 | ✅ 正确（selfcheck maxdiff=0，解码与 CPU 一致） |

> CPU 无 FP16 硬件，其推理精度恒为 fp32。
> **Vulkan（NVIDIA）行为由推理精度决定，与权重精度（fp16/fp32）无关**：fp16 计算均发散；fp32 计算默认超显存，但加 `--weights-in-host` 将权重 offload 到系统内存后，**fp32 计算正确可运行**（已在 NVIDIA 实测）。故上述 fp16 权重与 fp32 权重各行结果一致。

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
- `--fp16`：推理精度，仅作用于 Vulkan 文本链；视觉链与 CPU 恒为 fp32。
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
可选：`--threads N`、`--max-new-tokens N`、`--vulkan-device <idx>`、`--no-mtp`（纯 AR）、`--cpu-only`（跳过已知不可用的 Vulkan）、`--save-dir <dir>`（每配置存一张带框图）。

**注意**：
- 一致性只衡量平台输出是否彼此相同，不代表检出全部目标（`datas/football.jpg` 有 **8 人**）；检测质量需另用标注数据评估。
- MTP 与 AR 结果一致；框数差异来自 token 预算截断（一个框恰 6 token，`max_new=20` 仅够 3 框）。跨平台/跨模式对比时 `--max-new-tokens` 须**一致且给足**。

## Vulkan fp16 发散（定位小结）
Vulkan 下**fp16 计算**在 NVIDIA（1080Ti/5090）与 Apple M1 Pro 上均发散（坐标 token 全 0 → `<ref>!!!…`），而**fp32 计算正确**（selfcheck maxdiff=0，解码与 CPU 一致）。已排除的候选与结论：
- 非 Flash Attention（fp16 下强制禁用 flash，结果逐位不变）。
- 非 RoPE（PR #6834 已 patch，fp16 仍发散）。
- 非通用 fp16 精度（同一模型视觉链 fp16 正确，仅文本 decoder 发散）。
- 与权重精度无关（fp16 权重与 fp32 权重行为相同，发散来自 **fp16 计算管线**）。

剩余嫌疑集中在**文本 decoder 的 fp16 SDPA 数值管线**（fp16 累加精度/softmax 在长序列下塌缩或溢出、KV 增量解码误差累积），视觉链为短序列单次 prefill 故正常。根治方向：让 SDPA 的 fp16 路径使用更高精度（fp32）累加，排查上游是否已有修复。