#include <iostream>
#include <string>
#include <vector>
#include "ncnn_llm_locateanything.h"
#include "utils/draw_utils.h"
#include "utils/image_utils.h"
#include "utf8_args.h"

// LocateAnything-3B grounding CLI（ncnn_llm 补充实现）。
//
// 端到端跑 universal 导出的 fp16 六子图：
//   图片 + 自然语言 query -> <box><x1><y1><x2><y2></box> 结构化坐标 token。
// 这是 VLM 目标定位（grounding），不是 OCR 文字识别。
//
// 用法:
//   locate_main --model <models-dir> --image <image> [--prompt <question>] [--vulkan] [--threads N]
//               [--save <out.png>] [--no-draw]
//   （--model 默认 models/locate-anything-fp16，--image 必填）
// 默认会把带框的图标注结果写到 <image>_locate.png（--save 指定路径，--no-draw 关闭）。

int main(int argc, char** argv) {
    enable_utf8_console();
    std::vector<std::string> args = get_utf8_args(argc, argv);

    std::string model_path = "models/locate-anything-fp16";
    std::string image_path;
    std::string prompt;
    bool prompt_set = false;
    bool use_vulkan = false;
    bool greedy = false;
    bool use_fp16 = false;
    bool weights_in_host = false;
    int vulkan_device = 0;
    int threads = 4;
    int max_new = 512;
    std::string save_path;
    bool draw = true;

    for (size_t i = 1; i < args.size(); i++) {
        const std::string& arg = args[i];
        if (arg == "--model" && i + 1 < args.size()) model_path = args[++i];
        else if (arg == "--image" && i + 1 < args.size()) image_path = args[++i];
        else if (arg == "--prompt" && i + 1 < args.size()) { prompt = args[++i]; prompt_set = true; }
        else if (arg == "--vulkan") use_vulkan = true;
        else if (arg == "--vulkan-device" && i + 1 < args.size()) {
            vulkan_device = std::stoi(args[++i]);
            if (vulkan_device < 0) vulkan_device = 0;
        }
        else if (arg == "--fp16") use_fp16 = true;
#if !defined(__APPLE__)
        // macOS 为统一内存架构，host 权重无显存收益，忽略该标志（离散 GPU 才生效）。
        else if (arg == "--weights-in-host") weights_in_host = true;
#endif
        else if (arg == "--greedy") greedy = true;
        else if (arg == "--max-new-tokens" && i + 1 < args.size()) {
            max_new = std::stoi(args[++i]);
            if (max_new <= 0) max_new = 512;
        }
        else if (arg == "--threads" && i + 1 < args.size()) {
            threads = std::stoi(args[++i]);
            if (threads <= 0) threads = 4;
        }
        else if (arg == "--save" && i + 1 < args.size()) save_path = args[++i];
        else if (arg == "--no-draw") draw = false;
    }

    if (image_path.empty()) {
        fprintf(stderr, "Usage: %s --image <image_path> [--model <model_path>] [--prompt <question>]\n"
                        "       [--vulkan] [--vulkan-device <idx>] [--fp16] [--weights-in-host] [--threads N]\n"
                        "       [--save <out.png>] [--no-draw]\n",
                argv[0]);
        return 1;
    }

    printf("Loading LocateAnything model from %s (threads=%d, vulkan=%s%s%s, precision=%s)\n",
           model_path.c_str(), threads, use_vulkan ? "on" : "off",
           use_vulkan ? (" device=" + std::to_string(vulkan_device)).c_str() : "",
           use_vulkan && weights_in_host ? ", weights=host(offload)" : "",
           use_fp16 ? "fp16" : "fp32");

    ncnn_llm_locateanything la(model_path, use_vulkan, threads, vulkan_device, use_fp16,
                               weights_in_host);
    if (!la.ok()) {
        fprintf(stderr, "Failed to load LocateAnything model\n");
        return 1;
    }

    // 默认 grounding 指令：让模型把图中所有目标都用 box 标出。
    // 用户可用 --prompt 覆盖成更具体的 query。
    if (!prompt_set) {
        prompt = "Where is the object?";
    }

    printf("Loading image: %s\n", image_path.c_str());
    ncnn::Mat bgr = load_image_to_ncnn_mat(image_path);
    if (ncnn_mat_empty(bgr)) {
        fprintf(stderr, "Failed to load image: %s\n", image_path.c_str());
        return 1;
    }

    LocateGenerateConfig cfg;
    cfg.max_new_tokens = max_new;
    cfg.do_sample = !greedy;   // --greedy 关闭采样，确定性 argmax（用于与 torch 逐 token 对比定位）
    cfg.temperature = 0.7f;
    cfg.top_p = 0.9f;
    cfg.top_k = 50;
    cfg.repetition_penalty = greedy ? 1.0f : 1.1f;   // greedy 时关掉 rep/采样，纯净 argmax
    // 逐 token 打印到 stderr，便于实时观察 decode 进度（stdout 块缓冲看不动）
    cfg.callback = [](const std::string& t) { fprintf(stderr, "%s", t.c_str()); fflush(stderr); };

    printf("Grounding with prompt: %s\n", prompt.c_str());
    printf("Generating:\n");

    std::string out = la.run(bgr, prompt, cfg);
    fprintf(stderr, "\n");

    // 结构化输出：模型原始文本（含 <box><x1><y1><x2><y2></box>）+ 解析后的坐标
    printf("\nRaw output: %s\n", out.c_str());

    const std::vector<LocateBox>& boxes = la.last_boxes();
    printf("Detected %zu box(es) on %dx%d image:\n", boxes.size(), bgr.w, bgr.h);
    for (size_t i = 0; i < boxes.size(); i++) {
        printf("  %s\n", format_locate_box(boxes[i], bgr.w, bgr.h, (int)i).c_str());
    }

    // 在原图标注并保存
    if (draw) {
        const std::string out_png = save_path.empty() ? default_annotated_path(image_path) : save_path;
        if (draw_locate_boxes(bgr, boxes, out_png))
            printf("Annotated image saved: %s\n", out_png.c_str());
        else
            fprintf(stderr, "Failed to save annotated image: %s\n", out_png.c_str());
    }

    printf("\nDone.\n");
    return 0;
}