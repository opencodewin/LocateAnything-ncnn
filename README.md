# LocateAnything-ncnn

基于 [`ncnn`](https://github.com/Tencent/ncnn) 的 LocateAnything-3B（grounding VLM）推理实现。它是独立工程，仅从 [`ncnn_llm`](https://github.com/futz12/ncnn_llm) 拷入了该模型运行所需的最小代码集。

## 特性
- 支持动态输入推理（接近原始项目），拆分为 6 个子图：vision_embed / vision_encoder / vision_projector / text_embed / text_decoder(KV) / lm_head。
- 支持 MTP 并行窗口解码 + 结构化坐标 token（`<box><x1><y1><x2><y2></box>`）；也支持纯逐 token AR 解码（`--no-mtp`，两者输出结果一致）。
- 图像处理严格按照原始项目中的 `MAX_DIM=1024`：最大边超过 1024 时用 LANCZOS 预缩后，再做自适应 grid。
- `--vulkan-device <idx>` 支持多 GPU 选卡；`--fp16` 启用 2 字节激活存储（bf16 优先 / fp16 回退），不是强制使用 fp16 算术。默认是 fp32 存储路径。`--weights-in-host`（仅离散 GPU / 非 mac）把模型权重 offload 到系统内存，缓解 fp32 显存压力。

### 精度和 CPU 路径的说明

本文中的“fp32”通常表示模型的默认存储/数据路径，不表示 CPU 内部每条指令都以 fp32 标量执行。ncnn 会根据编译选项、运行时 CPU 能力和具体算子选择不同 kernel，例如：

- x86 的 SSE/AVX/AVX2/AVX-512/FMA 等 SIMD kernel 可能以向量形式处理 fp32；
- BF16、FP16、INT8 等 kernel 会使用对应的数据类型或转换路径；
- GEMM、SDPA、量化和卷积等算子可能分别选择不同的实现，不能只根据“CPU”或“AVX”推断统一的计算数据类型；
- 本项目默认 CPU 推理关闭 2 字节存储，因此本次 CPU 对照测试使用的是 fp32 数据路径，但这不等于所有底层指令都是 fp32 标量运算。

因此，README 中将“存储类型”“算子输入/输出类型”“累加类型”和“SIMD 指令集”分开描述。`--fp16` 在本项目中主要控制 Vulkan 的 2 字节激活存储；它不应被理解为“整条推理链都改成 fp16 算术”。

## 已知问题

### 平台 × 精度状态

| 后端 / 硬件 | 测试配置 | 存储与计算说明 | 结果 |
| --- | --- | --- | --- |
| CPU · Apple Silicon (macOS) | 默认 CPU 路径 | 默认 fp32 数据路径；实际 kernel 由 ARM/NEON 等能力决定 | ✅ 正确 |
| CPU · Intel x86 (Windows) | 默认 CPU 路径 | 默认 fp32 数据路径；实际使用 SSE/AVX 等编译和运行时分派 | ✅ 正确 |
| CPU · AMD x86 (Linux) | AVX-512 开启/关闭 | 默认 fp32 数据路径；x86 SDPA 的 packed KV cache 经过修复后两种配置均正确 | ✅ 正确 |
| Vulkan · NVIDIA | fp32 存储 | fp32 存储路径；MTP 与 `--no-mtp` 均已实测 | ✅ 正确 |
| Vulkan · NVIDIA (5090) | `--fp16` | 使用 bf16 存储时正确 ✅；使用 fp16 存储时发散 ❌ | ⚠️ 取决于实际存储类型 |
| Vulkan · NVIDIA (1080Ti) | `--fp16` | 无 bf16 存储，回退到 fp16 存储 | ❌ 发散，出现 `<ref>!!!…` |
| Vulkan · Apple M1 Pro (MoltenVK) | `--fp16` | 回退到 fp16 存储，无 bf16 存储 | ❌ 发散 |

> `--fp16` 的语义：在本项目中主要启用 2 字节激活存储，Vulkan 优先选择 bf16；设备不支持 bf16 时回退到 fp16。`use_fp16_arithmetic` 已关闭，因此不能把该选项理解成“强制整条推理链使用 fp16 算术”。
> 当前实测表明，Vulkan 的 fp32 存储路径正确；2 字节存储是否正确取决于实际使用的类型。bf16 保留与 fp32 相同的指数宽度，在 5090 上正确；fp16 的指数范围较小，在 SDPA softmax 和长序列累加中发散。没有 bf16 存储能力的设备应使用默认 fp32 存储。

### 已修复
ncnn ROPE/RotaryEmbed 的 Vulkan 实现 —— 上游 [PR #6834](https://github.com/Tencent/ncnn/pull/6834) 修复全宽 `cos/sin` 缓存（2D / vision RoPE），以 patch 形式构建时自动应用。注意其未覆盖 `src/layer/x86/*`；对本工程输入 CPU 侧是 no-op（真正修复的是 Vulkan 缓存步长）。

### AMD CPU (x86, Linux) 的 KV cache 修复

早期测试中，AMD x86 CPU 出现无输出或坐标错误，而 Intel CPU 正常。问题最终定位到 MTP 的 `trim_kv()`：它把 ncnn SDPA 的 KV cache 当成普通二维 FP32 Mat 逐行 `memcpy`。但 x86 SDPA 为 SIMD kernel 使用了 panel/interleaved cache layout；AVX-512、AVX2 等路径的 panel 宽度还可能不同。

修复方式是只修改 KV cache 的逻辑序列长度 `Mat.h`，保留 ncnn 已经创建的底层 buffer、capacity、allocator 和物理布局，不再自行复制或重建 cache。该修改已在 AMD x86 上验证：

- AVX-512 开启：正确；
- AVX-512 关闭：正确；
- 开启 MTP 与 `--no-mtp`：均正确。

这说明问题不是 AMD 的浮点运算错误，也不是 AVX-512 或 BF16 精度问题，而是宿主代码错误地处理了 ncnn 的 packed KV cache。ncnn 作者关于 KV cache `Mat` layout 尚不稳定的说明正适用于此处：普通 `clone`、按逻辑行 `truncate` 或假定 `elempack == 1` 都不能证明物理布局是普通逐行布局。

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
- `--fp16`：启用 2 字节激活存储（bf16 优先，设备无 bf16 时回退 fp16），**同时作用于 Vulkan 文本链与视觉链**（二者一致）；本项目关闭 fp16 arithmetic。默认使用 fp32 存储路径。见「已知问题 / Vulkan 2 字节存储问题」。
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

## Vulkan 2 字节存储问题（定位小结）
早期测试把问题归因于 Vulkan fp16 arithmetic；后续关闭 fp16 arithmetic，并分别比较 fp32、bf16 和 fp16 存储后，确认主要原因是激活存储类型的指数范围，而不是单纯的算术开关。定位过程中排除了：

- Flash Attention：禁用后结果没有改变；
- RoPE：上游 PR #6834 已应用，但 2 字节存储问题仍存在；
- 单独的文本 decoder：视觉前向在相同 2 字节存储配置下也会受到影响；
- 权重存储：fp16/fp32 权重的差异不是主要原因，问题出在激活存储类型。

**结论**：`src/ncnn_llm_base.h` 中已关闭 `use_fp16_arithmetic`；`--fp16` 只负责选择 2 字节激活存储，bf16 优先、无 bf16 时回退 fp16。fp32 Vulkan 路径已验证正确；5090 上 bf16 存储正确而 fp16 存储发散，原因是 bf16 保留与 fp32 相同的指数宽度，而 fp16 的指数范围较小。1080Ti 和 M1 Pro 没有可用的 bf16 存储能力，应使用默认 fp32 存储。