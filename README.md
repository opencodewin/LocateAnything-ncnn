# LocateAnything-ncnn

基于 [ncnn](https://github.com/Tencent/ncnn) 的 LocateAnything-3B (grounding VLM) 推理实现。它是独立工程，仅从 `ncnn_llm` 拷入了该模型运行所需的最小代码集。

## 特性
- 支持动态输入推理（接近原始项目），拆分为 6 个子图：vision_embed / vision_encoder / vision_projector / text_embed / text_decoder(KV) / lm_head。
- 支持 MTP 并行窗口解码 + 结构化坐标 token (`<box><x1><y1><x2><y2></box>`)。
- 图像处理严格按照原始项目中的 `MAX_DIM=1024`，如果最大边超过 1024，采用 LANCZOS 预缩后再自适应 grid。
- CPU fp32/fp16 正确路径（**已验证：macOS Apple Silicon、Windows Intel x86**；AMD CPU 上无输出，见「已知问题」）；Vulkan 实验路径（测试机器为 1080ti，它不支持 fp16 计算且显存只有 11G，目前已知的是 fp16 的结果不正确，fp32 显存不够）

## 已知问题
- ~~ncnn 的 ROPE/RotaryEmbed Vulkan 实现存在问题~~ · 已修复：上游 [PR #6834](https://github.com/Tencent/ncnn/pull/6834) 支持全宽 `cos/sin` 缓存（2D / vision RoPE），已以 patch 方式在构建时自动应用，见 [patches](#构建--在-windows-msys2mingw-下构建未测试其他平台后续补齐)。
  - 注意：该补丁只覆盖 `arm / loongarch / mips / generic(rotaryembed.cpp) / vulkan`，**未覆盖 `src/layer/x86/*`**。对本工程的输入而言 CPU 侧是 no-op（视觉 RoPE 走半宽缓存、文本 RoPE 两半同值），真正被修的是 Vulkan 的缓存步长；但 x86 CPU 上若出现"全宽且两半不同值"的 2D RoPE 仍会算错，后续需要补 x86 补丁。
- Vulkan (GPU) 端到端输出错误。
  - **macOS（Apple M1 Pro, MoltenVK）实测：fp16 与 fp32 均完全发散**。`bench_platform` 定量结果显示两者输出坐标 token 全 0（最终为 `<ref>!!!…`），相对 CPU fp32 的文本一致性仅约 4.6%（详见下方「平台测试」）。虽然耗时快约 4.7×，但结果不可用。
  - **RTX 5090（Vulkan）实测：fp16 与 fp32 均无有效输出**（不产出任何可用 `<box>` 坐标）。
  - 测试机 1080ti 另有：fp16 计算不支持、fp32 显存(11G)不足的限制（见「特性」）。
- **AMD CPU（Linux）实测：无任何输出**（一个框都解不出来；同样的 x86 代码路径在 Windows Intel CPU 上正常）。
  - 已排除的方向：`patches/` 的 RoPE 补丁对本工程输入在 CPU 上是 no-op（见上），不是根因；差异更可能来自编译器 / 运行时 ISA 分派（GCC vs MSVC、是否启用 AVX512）或 ncnn 在该架构返回 packed Mat（`elempack=4/8`）。
  - 排查手段：运行时会打印各子图输出的 `SHAPE ... pack=N`（非 1 时会额外打印 `NOTE ... unpack packN -> pack1`）、视觉特征统计 `DEBUG feat`、以及跨平台指纹 `FEATSUM` / `PREFILL top5 logits`；`LA_DUMP_VISION=<path>` / `LA_DUMP_TEXT=<prefix>` 可 dump fp32 特征逐元素对比。宿主侧已对所有子图输出做"解包 + 摊平成规范 2D"处理（`la_canonical_2d` / `la_canonical_kv`），可排除 packed Mat 解释错位这一类问题。
  - 若怀疑 AVX512 kernel：使用下面的独立构建目录重编做二分。不要复用已有 `build` 目录，否则 CMakeCache 可能保留原来的 ISA 开关。

### AMD CPU ISA 隔离

先关闭 AVX512，确认问题是否来自 AMD 的 AVX512 专用 kernel：

```bash
cmake -G Ninja -B build-amd-no-avx512 -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DLOCATE_NCNN_VULKAN=OFF \
  -DNCNN_AVX512=OFF \
  -DNCNN_AVX512FP16=OFF \
  -DNCNN_AVX512BF16=OFF \
  -DNCNN_AVX512VNNI=OFF
cmake --build build-amd-no-avx512
```

使用与原构建完全相同的模型、图片、prompt、线程数和 token 数运行：

```bash
./build-amd-no-avx512/locate_main \
  --model <fp32模型目录> \
  --image datas/football.jpg \
  --prompt human \
  --max-new-tokens 20 \
  --threads 6 \
  --no-draw
```

如果仍然错误，再关闭所有高级 x86 ISA，建立 generic CPU 对照：

```bash
cmake -G Ninja -B build-amd-generic -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DLOCATE_NCNN_VULKAN=OFF \
  -DNCNN_RUNTIME_CPU=OFF \
  -DNCNN_AVX=OFF \
  -DNCNN_AVX2=OFF \
  -DNCNN_AVX512=OFF \
  -DNCNN_FMA=OFF \
  -DNCNN_FMA4=OFF \
  -DNCNN_F16C=OFF
cmake --build build-amd-generic
```

对比两种构建的 `DEBUG feat`、`FEATSUM`、`PREFILL top5 logits` 和 `Raw output`：

- generic 版本恢复正常：问题在 ncnn 的 x86 ISA kernel 或运行时分派，优先继续在 AVX512、AVX2、FMA 之间二分。
- `FEATSUM` 已不同：问题发生在 vision_embed / vision_encoder / vision_projector 或视觉 RoPE。
- `FEATSUM` 一致但 `PREFILL top5 logits` 不同：问题发生在 text_embed / text_decoder / lm_head。
- 两种构建都错误：问题不太可能只是 ISA kernel，应继续比较编译器、模型文件和 `LA_DUMP_VISION` / `LA_DUMP_TEXT` 的逐元素 dump。

## 目录结构
```
3rdparty/ncnn   # git submodule：ncnn 源码（构建时会自动应用 patches/ 下的补丁）
patches/        # 编译时自动应用到 ncnn 的补丁（按前缀数字排序，编号小的先应用）
src/            # 最小代码集：运行时 + tokenizer + image_utils + nlohmann/json
models/         # （外部）fp32 / fp16 动态模型目录，通过 --model 引用，不随工程提交
```

## 外部依赖
- ncnn（submodule）。首次克隆后执行：`git submodule update --init --recursive`
- Git（配置阶段用于把 `patches/` 下的补丁应用到 ncnn 子模块）。

## 模型
模型提供 fp16 和 fp32 两种动态导出，模型已在上传中。
- fp32 动态：`locate-anything-fp32`
- fp16 动态：[`locate-anything-fp16`](https://www.modelscope.cn/models/sizeofbeer/locate-anything-fp16)

## 构建 (目前在 Windows MSYS2/MinGW 下构建，未测试其他平台，后续补齐)
```bash
cmake -G Ninja -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
默认启用 Vulkan（`-DLOCATE_NCNN_VULKAN=OFF` 可关）。NCNN_VULKAN 宏由 ncnn 目标透出，基类据此切换 GPU。

> **补丁自动应用**：`cmake` 配置阶段会按 `patches/` 下的文件（前缀数字从小到大）逐个 `git apply` 到 ncnn 子模块，已应用的会自动跳过（幂等），无需手动处理。若补丁与当前 ncnn 版本不匹配会给出明确报错。

## 运行
```bash
./build/locate_main.exe --model <fp32|fp16 模型目录> --image <img> --prompt <query> [选项]
```
选项：
- `--vulkan`：文本链路走 Vulkan GPU；`--vulkan-device <idx>`：多 GPU 时指定设备序号（启动时会列出所有可用设备及编号；越界自动回退 0）。
- `--fp16`：推理精度（默认 fp32）。`--fp16` 仅作用于 Vulkan 文本链路，视觉链与 CPU 路径恒为 fp32，避免污染视觉特征 / CPU 无 FP16 硬件的回退。
- `--threads N`、`--greedy`、`--max-new-tokens N` 同前。
- `--save <out.png>`：把检测框画到原图上并保存（默认写到 `<image>_locate.png`）；`--no-draw` 关闭保存。

输出中除模型原始文本（含 `<box><x1><y1><x2><y2></box>` 结构化坐标 token）外，还会逐框打印归一化坐标与像素坐标，例如：
```
Raw output: <box><120><80><600><900></box>...
Detected 2 box(es) on 1280x720 image:
  #0 norm=(0.1200,0.0800)-(0.6000,0.9000) px=(154,58)-(768,648) size=615x591
```
归一化坐标 × 原图宽/高即像素坐标（`src/utils/draw_utils.h`：`parse_locate_boxes_text` / `draw_locate_boxes`）。

## 跨平台一致性排查（某平台无输出 / 结果发散时）
运行时会打印可直接跨平台比对的"指纹"，用来定位发散发生在哪一级：

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

**基准（macOS M1 Pro，CPU fp32，`models/locate-anything-fp16` + `datas/football.jpg` + `--prompt human --max-new-tokens 20`）**：
```
DEBUG feat w=2048 h=925 min=-29.245 max=14.516 mean=-0.0010
FEATSUM -1853.253568
PREFILL top5 logits(V=152681): 151672:23.3104 151645:8.5655 151668:8.5340 151741:7.0714 152191:6.9475
md5 feat.f32        = 98fa1daf78da0786cbd7d1e05e504f87
md5 pre_hidden.f32  = 7fd2130dd0dae61da7b94ecb6e0b65c6
md5 pre_logits.f32  = 7f7446504468f4712cfeb3b0384de00a
```
`FEATSUM` 不一致 → 视觉链（vision_embed/encoder/projector 或宿主 pos_emb 插值/moon RoPE）发散；`FEATSUM` 一致但 `PREFILL top5` 不一致 → 文本链（text_embed / decoder prefill / lm_head）发散。

## 平台测试
`bench_platform` 在同一输入下按 **cpu-fp32 → gpu-fp32 → gpu-fp16** 顺序运行，统计各平台端到端耗时（`end-to-end avg`），并以 **cpu-fp32 输出为参考**衡量其它平台文本一致性（归一化 Levenshtein，1 = 与参考完全一致）。使用 greedy 确定性解码，保证跨平台可比；每平台先 warmup 1 次（不计时）再测 N 次。

### 在某个平台上做测试的步骤
1. **构建**（含 `bench_platform`）。目标机没有可用的 Vulkan 实现时可以关掉只做 CPU 对比：`-DLOCATE_NCNN_VULKAN=OFF`。
   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --target bench_platform
   ```
2. **快速验证**（小 `--max-new-tokens`，确认平台能跑通、GPU 可枚举）：
   ```bash
   cd build
   ./bench_platform --image ../datas/football.jpg --prompt human --max-new-tokens 20 --iter 1
   ```
   启用 Vulkan 时，`[ncnn] Vulkan devices (N)` 会列出所有可用设备及编号，越界会自动回退到 0。
3. **正式统计**（默认 512 token；CPU 平台耗时较高，可按需调小）：
   ```bash
   ./bench_platform --image ../datas/football.jpg --prompt human --iter 3
   ```
   可选：`--threads N`、`--max-new-tokens N`、`--vulkan-device <idx>`、`--no-mtp`（纯逐 token AR 解码）、`--cpu-only`（只跑 CPU，跳过已知不可用的 Vulkan）、`--save-dir <dir>`（每个配置各存一张带框标注图 `<label>.png`）。
4. **解读结果**：
   - **耗时**：各平台 `end-to-end avg`，对比不同计算后端/精度的吞吐。
   - **一致性**：参考平台之外的 `sim(vs cpu-fp32)` 与 `identical`；接近 1 表示与 CPU fp32 输出一致，接近 0 表示发散（例如本仓库 macOS 上两个 Vulkan 配置都只有约 4.6%）。

**注意**：
- **参考基准 cpu-fp32 并不等于绝对真值**。`datas/football.jpg` 实际包含 **8 人**，但默认 `max_new` 较小时可能只输出少量 box —— 本工具的一致性只衡量「各平台输出是否彼此相同」，不代表「检出全部目标」。要评估检测质量（漏检/误检），应另用带标注数据做 recall / precision 评测。
- **MTP 与 AR 都是对的，差异不在解码算法，而在 `--max-new-tokens` 要设置得足够大**。`bench_platform` 实测：对同一输入，MTP（`default`）与 AR（`--no-mtp`）生成的 token 序列几乎逐位一致（唯一分歧是某框 x2 坐标差 2 个 bin，来自 MTP 并行窗口的数值近似，可忽略）。同一 `max_new=20` 时 MTP 只够提交 3 个完整框（一个框恰为 6 token：`box+4 坐标+/box`），到 3 框即被截断，看起来「只检出 3 人」；换成 `--no-mtp --max-new-tokens 48 --cpu-only` 则 AR 能继续框出到第 7 框（第 8 框被 limit 截断）。也就是说「框数变少」是 token 预算截断的假象，不是 MTP 或 AR 导致；MTP 并行与 AR 逐 token 都应产出相同结果。
- **跨平台/跨模式下对比时 `--max-new-tokens` 必须一致且给足**（或统一用自然 `im_end` 终止），否则框数/文本差异只是截断点的假象。图中 8 人一次性并非都会被框出，漏检仍主要是模型在该图的召回率问题，与 CPU/GPU、中英文、MTP/AR 无关。
