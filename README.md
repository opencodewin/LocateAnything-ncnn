# LocateAnything-ncnn

基于 [ncnn](https://github.com/Tencent/ncnn) 的 LocateAnything-3B (grounding VLM) 推理实现。它是独立工程，仅从 `ncnn_llm` 拷入了该模型运行所需的最小代码集。

## 特性
- 支持动态输入推理（接近原始项目），拆分为 6 个子图：vision_embed / vision_encoder / vision_projector / text_embed / text_decoder(KV) / lm_head。
- 支持 MTP 并行窗口解码 + 结构化坐标 token (`<box><x1><y1><x2><y2></box>`)。
- 图像处理严格按照原始项目中的 `MAX_DIM=1024`，如果最大边超过 1024，采用 LANCZOS 预缩后再自适应 grid。
- CPU fp32/fp16 正确路径；Vulkan 实验路径（测试机器为 1080ti，它不支持 fp16 计算且显存只有 11G，目前已知的是 fp16 的结果不正确，fp32 显存不够）

## 已知问题
- ~~ncnn 的 ROPE/RotaryEmbed Vulkan 实现存在问题~~ · 已修复：上游 [PR #6834](https://github.com/Tencent/ncnn/pull/6834) 支持全宽 `cos/sin` 缓存（2D / vision RoPE），已以 patch 方式在构建时自动应用，见 [patches](#构建--在-windows-msys2mingw-下构建未测试其他平台后续补齐)。
- Vulkan fp16 结果不正确、fp32 显存不足（测试机 1080ti 限制，详见特性）。

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
- fp16 动态：`locate-anything-fp16`

## 构建 (目前在 Windows MSYS2/MinGW 下构建，未测试其他平台，后续补齐)
```bash
cmake -G Ninja -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
默认启用 Vulkan（`-DLOCATE_NCNN_VULKAN=OFF` 可关）。NCNN_VULKAN 宏由 ncnn 目标透出，基类据此切换 GPU。

> **补丁自动应用**：`cmake` 配置阶段会按 `patches/` 下的文件（前缀数字从小到大）逐个 `git apply` 到 ncnn 子模块，已应用的会自动跳过（幂等），无需手动处理。若补丁与当前 ncnn 版本不匹配会给出明确报错。

## 运行
```bash
./build/locate_main.exe --model <fp32|fp16 模型目录> --image <img> --prompt <query> [--vulkan] [--threads N]
```
