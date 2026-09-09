# LocateAnything-ncnn

基于 [ncnn](https://github.com/Tencent/ncnn) 的 LocateAnything-3B (grounding VLM) 推理实现。它是独立工程，仅从 `ncnn_llm` 拷入了该模型运行所需的最小代码集。

## 特性
- 支持动态输入推理（接近原始项目），拆分为 6 个子图：vision_embed / vision_encoder / vision_projector / text_embed / text_decoder(KV) / lm_head。
- 支持 MTP 并行窗口解码 + 结构化坐标 token (`<box><x1><y1><x2><y2></box>`)。
- 图像处理严格按照原始项目中的 `MAX_DIM=1024`，如果最大边超过 1024，采用 LANCZOS 预缩后再自适应 grid。
- CPU fp32/fp16 正确路径；Vulkan 实验路径（测试机器为 1080ti，它不支持 fp16 计算且显存只有 11G，目前已知的是 fp16 的结果不正确，fp32 显存不够）

## 已知问题
- ncnn 下的 ROPE 的 Vulkan 实现存在问题

## 目录结构
```
3rdparty/ncnn   # git submodule：ncnn 源码
src/            # 最小代码集：运行时 + tokenizer + image_utils + nlohmann/json
models/         # （外部）fp32 / fp16 动态模型目录，通过 --model 引用，不随工程提交
```

## 外部依赖
- ncnn（submodule）。首次克隆后执行：`git submodule update --init --recursive`

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

## 运行
```bash
./build/locate_main.exe --model <fp32|fp16 模型目录> --image <img> --prompt <query> [--vulkan] [--threads N]
```
