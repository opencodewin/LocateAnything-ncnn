# LocateAnything-ncnn

基于 [`ncnn`](https://github.com/Tencent/ncnn) 的 LocateAnything-3B（grounding VLM）推理实现。它是独立工程，仅从 [`ncnn_llm`](https://github.com/futz12/ncnn_llm) 拷入了该模型运行所需的最小代码集。

## 特性
- 支持动态输入推理（接近原始项目），拆分为 6 个子图：vision_embed / vision_encoder / vision_projector / text_embed / text_decoder(KV) / lm_head。
- 支持 MTP 并行窗口解码 + 结构化坐标 token（`<box><x1><y1><x2><y2></box>`）；也支持纯逐 token AR 解码（`--no-mtp`，预算充足时输出语义一致）。
- 图像处理严格按照原始项目中的 `MAX_DIM=1024`：最大边超过 1024 时用 LANCZOS 预缩后，再做自适应 grid。
- `--vulkan` 启用 Vulkan，`--vulkan-device <idx>` 选择设备；`--fp16` 明确选择 IEEE FP16 激活存储，`--bf16` 明确选择 BF16 激活存储，不再按设备能力自动优先 BF16。默认是 FP32 存储；两个低精度参数互斥，且需要 `--vulkan`。这些选项控制运行时激活存储，与 `--model` 指向的模型权重文件格式相互独立。`--weights-in-host`（仅离散 GPU / 非 mac）把模型权重 offload 到系统内存，缓解 fp32 显存压力。

### 精度和 CPU 路径的说明

本文中的“fp32”通常表示模型的默认存储/数据路径，不表示 CPU 内部每条指令都以 fp32 标量执行。ncnn 会根据编译选项、运行时 CPU 能力和具体算子选择不同 kernel，例如：

- x86 的 SSE/AVX/AVX2/AVX-512/FMA 等 SIMD kernel 可能以向量形式处理 fp32；
- BF16、FP16、INT8 等 kernel 会使用对应的数据类型或转换路径；
- GEMM、SDPA、量化和卷积等算子可能分别选择不同的实现，不能只根据“CPU”或“AVX”推断统一的计算数据类型；
- 本项目默认 CPU 推理关闭 2 字节存储，因此本次 CPU 对照测试使用的是 fp32 数据路径，但这不等于所有底层指令都是 fp32 标量运算。

因此，README 中将“存储类型”“算子输入/输出类型”“累加类型”和“SIMD 指令集”分开描述。`--fp16` 和 `--bf16` 只选择 Vulkan 激活存储类型，不代表整条推理链都使用相同算术精度。

## 已知问题

### 平台 × 精度状态

| 后端 / 硬件 | 测试配置 | 存储与计算说明 | 结果 |
| --- | --- | --- | --- |
| CPU · Apple Silicon (macOS) | 默认 CPU 路径 | 默认 fp32 数据路径；实际 kernel 由 ARM/NEON 等能力决定 | ✅ 正确 |
| CPU · Intel x86 (Windows) | 默认 CPU 路径 | 默认 fp32 数据路径；实际使用 SSE/AVX 等编译和运行时分派 | ✅ 正确 |
| CPU · AMD x86 (Linux) | AVX-512 开启/关闭 | 默认 fp32 数据路径；x86 SDPA 的 packed KV cache 经过修复后两种配置均正确 | ✅ 正确 |
| Vulkan · NVIDIA | fp32 存储 | fp32 存储路径；MTP 与 `--no-mtp` 均已实测 | ✅ 正确 |
| Vulkan · NVIDIA (5090) | `--bf16` / `--fp16` | 分别强制 BF16 / IEEE FP16 存储；有针对性修复覆盖 | ✅ 正确 |
| Vulkan · NVIDIA (1080Ti) | `--fp16` | 强制 IEEE FP16 存储；不支持 BF16 时 `--bf16` 会明确报错 | ✅ 正确 |
| Vulkan · Apple M1 Pro (MoltenVK) | fp32 存储 | 完整 `football.jpg` prompt 触发 `vkWaitForFences failed -4`；在 `3cfacf1` 上同样复现 | ⚠️ 长 prompt device lost |
| Vulkan · Apple M1 Pro (MoltenVK) | `--fp16 --max-new-tokens 9` | IEEE FP16 MTP 修复 cache stride 后输出完整框 | ✅ 正确 |

> `--fp16` / `--bf16` 分别强制选择对应的 Vulkan 激活存储，不自动回退到另一种存储；设备不支持请求的类型时会报错。两者同时指定也会提示冲突并拒绝启动。设备报告支持 BF16 storage 时会启用 `use_fp16_arithmetic` 选项，最终可用能力仍由 ncnn 按设备特性校验。5090 和 1080Ti 上此前观察到的问题已定位并修复，但当前没有这两张卡可供复测。
> M1 Pro 上完整图片的 FP32 Vulkan 推理曾触发 MoltenVK `vkWaitForFences failed -4`，并在历史提交 `3cfacf1` 上复现；这是独立的 device-lost 故障，不能据此判断 FP32 数值正确性。FP16 MTP 的 KV 步长修复已在 M1 Pro 完整图片、greedy、9-token 测试中验证。

### Vulkan RMSNorm 的 FP16 溢出

Vulkan RMSNorm 的旧实现先在输入存储类型中计算平方，再把平方值交给归约。IEEE FP16 最大有限值为 65504，因此当输入绝对值超过约 256 时，平方可能溢出为无穷大；归约得到无效的均方值后，归一化系数可能变为零，表现为 decoder 某些输出行全零、后续 logits 分叉。这个问题是 RMSNorm 的 FP16 中间值溢出，不是单纯由 SDPA softmax 或长序列累加造成。

项目在 [0001-vulkan-rmsnorm-fp16-overflow.patch](patches/0001-vulkan-rmsnorm-fp16-overflow.patch) 中为 ncnn Vulkan RMSNorm 增加融合归约：直接在 FP32 累加器中计算并累加平方，避免将 `x*x` 写回 FP16 张量，也省去平方临时张量；pack1 和 pack4 路径都包含在补丁中。该补丁由项目 CMake 在配置时自动应用。它修复的是 RMSNorm 的中间值溢出，不意味着所有设备、所有 FP16 算子都已在每种配置上完成实机验证。

另一个独立问题是 MTP 窗口结束后 Vulkan KV cache 仅缩短逻辑长度、仍保留旧 `cstep`，后续 cache 增长会破坏历史 KV。LocateAnything 只在 Vulkan 提取出的标准 pack1 cache 上复制有效前缀并重建精确长度；CPU/x86 的 packed cache 仍只更新逻辑长度，以免破坏其物理布局。

### AMD CPU (x86, Linux) 的 KV cache 修复

早期测试中，AMD x86 CPU 出现无输出或坐标错误，而 Intel CPU 正常。问题最终定位到 MTP 的 `trim_kv()`：它把 ncnn SDPA 的 KV cache 当成普通二维 FP32 Mat 逐行 `memcpy`。但 x86 SDPA 为 SIMD kernel 使用了 panel/interleaved cache layout；AVX-512、AVX2 等路径的 panel 宽度还可能不同。

修复方式是只修改 KV cache 的逻辑序列长度 `Mat.h`，保留 ncnn 已经创建的底层 buffer、capacity、allocator 和物理布局，不再自行复制或重建 cache。该修改已在 AMD x86 上验证：

- AVX-512 开启：正确；
- AVX-512 关闭：正确；
- 开启 MTP 与 `--no-mtp`：均正确。

这说明问题不是 AMD 的浮点运算错误，也不是 AVX-512 或 BF16 精度问题，而是宿主代码错误地处理了 ncnn 的 packed KV cache。ncnn 作者关于 KV cache `Mat` layout 尚不稳定的说明正适用于此处：普通 `clone`、按逻辑行 `truncate` 或假定 `elempack == 1` 都不能证明物理布局是普通逐行布局。

## 目录结构
```
3rdparty/ncnn   # submodule：ncnn 源码，配置构建时自动应用 patches/
patches/        # 项目维护的 ncnn 补丁；按前缀数字排序并幂等应用
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
macOS 构建默认关闭 NCNN_SIMPLEVK，以便通过系统 Vulkan loader（通常由 MoltenVK 提供）加载 Vulkan；也可显式设置 `-DNCNN_SIMPLEVK=OFF`：
```bash
cmake -G Ninja -B build -S . -DCMAKE_BUILD_TYPE=Release -DNCNN_SIMPLEVK=OFF
```

## 运行
```bash
./build/locate_main --model <模型目录> --image <img> --prompt <query> [选项]
```
- `--vulkan`：启用 Vulkan；`--vulkan-device <idx>`：选择 GPU（索引越界时回退到 0）。
- `--vision-vulkan`：额外让视觉 3 子图（vision_embed / vision_encoder / vision_projector）走 Vulkan。默认只有文本链走 Vulkan、视觉链仍走 CPU；该选项需要 `--vulkan`。当前 M1 Pro/MoltenVK 上同时将六个子图放在 Vulkan 曾触发 `VK_ERROR_DEVICE_LOST`，视觉 Vulkan 路径应视作未验证。
- `--fp16`：明确启用 IEEE FP16 激活存储；`--bf16`：明确启用 BF16 激活存储。两个参数互斥，必须配合 `--vulkan`，且所选类型必须被 GPU 支持。所选存储类型作用于 Vulkan 子图；默认只有文本子图走 Vulkan，若同时启用 `--vision-vulkan`，视觉子图也使用所选存储类型。默认使用 FP32 存储路径。见「已知问题 / Vulkan 2 字节存储问题」。
- `--threads N`、`--greedy`、`--max-new-tokens N`。
- `--weights-in-host`（仅离散 GPU）：权重 offload 到系统内存，缓解设备显存不足。
- `--save <out.png>`：画框保存（默认 `<image>_locate.png`）；`--no-draw` 关闭。

输出除结构化坐标 token 外，会逐框打印归一化/像素坐标；像素 = 归一化 × 原图宽高（`src/utils/draw_utils.h`）。

## 平台测试
`bench_platform` 对同一输入按 cpu-fp32 → gpu-fp32 → gpu-fp16（默认）顺序运行，也可通过 `--bf16` 将最后一项切换为 gpu-bf16；统计各平台端到端耗时，并以 cpu-fp32 输出为参考衡量文本一致性（归一化 Levenshtein，1=完全一致）。采用确定性 greedy 解码；每平台 warmup 1 次后测 N 次。

```bash
cmake --build build --target bench_platform
./build/bench_platform --image datas/football.jpg --prompt human --max-new-tokens 20 --iter 1  # 快速验证
./build/bench_platform --image datas/football.jpg --prompt human --iter 3                     # 正式统计
```
可选：`--threads N`、`--max-new-tokens N`、`--vulkan-device <idx>`、`--fp16` / `--bf16`（选择 GPU 低精度存储；默认 fp16，不能同时指定）、`--no-mtp`（纯 AR）、`--cpu-only`（跳过 Vulkan）、`--vision-vulkan`（视觉 3 子图也走 Vulkan）、`--weights-in-host`（仅离散 GPU / 非 mac，权重 offload 到系统内存）、`--save-dir <dir>`（每配置存一张带框图）。

**注意**：
- 一致性只衡量平台输出是否彼此相同，不代表检出全部目标（`datas/football.jpg` 有 **8 人**）；检测质量需另用标注数据评估。
- `--max-new-tokens` 是生成 token 的预算上限，不保证一定用满。MTP 按完整结构组提交，不能跨预算提交半组：`<ref>human</ref>` 占 3 token，后续完整 `<box><x1><y1><x2><y2></box>` 占 6 token；在当前 `football.jpg` 的 greedy 测试中，`max-new-tokens=9` 可生成一个完整框，设为 4 时只提交完整 ref，剩余 1 个 token 不会用于半个 box。`--no-mtp` 的 AR 则逐 token 生成，预算不足时可能截断在结构组中间。比较 MTP 与 AR 或不同平台时，使用相同且足以容纳所需结构组的预算；对一个 ref 加一个框至少设为 9。
- CPU 上 MTP 与 AR 在 `football.jpg`、`max-new-tokens=9` 下均输出一个目标框，坐标相差最多 1 个量化单位；Vulkan IEEE FP16 的 MTP 经 KV stride 修复后也输出一个完整框。

## Vulkan 2 字节存储问题（定位小结）
早期测试曾怀疑 Vulkan FP16 指数范围、arithmetic 开关或 SDPA softmax；逐层比较 decoder 输出后，确认某些全零行来自 RMSNorm 在 FP16 存储类型中的平方溢出。定位过程中还检查了：

- Flash Attention：禁用后结果没有改变；
- RoPE：上游 PR #6834 已应用，但 2 字节存储问题仍存在；
- 单独的文本 decoder：视觉前向在相同 2 字节存储配置下也会受到影响；
- 权重存储：fp16/fp32 权重的差异不是主要原因，问题出在激活存储类型。

**结果摘要**：`--fp16` 与 `--bf16` 分别选择 IEEE FP16 / BF16 激活存储；设备报告支持 BF16 storage 时设置 `use_fp16_arithmetic=true`，ncnn 随后仍会按设备能力与存储选项校验。Vulkan RMSNorm 的 IEEE FP16 平方溢出由上述 ncnn 补丁修复；MTP KV stride 问题由 LocateAnything 的 Vulkan 专用精确长度复制修复。修复后 M1 Pro 完整图片、greedy、9-token FP16 MTP 输出完整框；第二窗口 cache 前缀 K/V 对照差异为零。5090 的 BF16/FP16 与 1080Ti 的 FP16 路径属于修复覆盖范围，但尚待对应硬件复测。FP32 Vulkan 的 M1 Pro 长 prompt device-lost 仍未解决；macOS 默认关闭 SimpleVK 以使用系统 Vulkan loader。