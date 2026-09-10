#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "ncnn_llm_locateanything.h"
#include "utils/draw_utils.h"
#include "utils/image_utils.h"
#include "utf8_args.h"

// 跨平台对比基准：同一输入分别在 CPU-fp32 / GPU-fp32 / GPU-fp16 下跑端到端推理，
// 统计端到端耗时，并以 CPU-fp32 输出为参考衡量其它平台的文本一致性。
//
// 用法:
//   bench_platform --image <img> [--model <dir>] [--prompt <q>] [--threads N]
//                  [--vulkan-device <idx>] [--max-new-tokens N] [--iter N]
//                  [--no-mtp]   纯逐 token AR（对比 MTP 窗口解码）
//                  [--cpu-only] 只跑 cpu-fp32，跳过两个 GPU 配置
//                  [--save-dir <dir>]  每个配置把带框标注图写到 <dir>/<label>.png

namespace {

struct BenchConfig {
    const char* label;
    bool vulkan;
    bool fp16;
};

const BenchConfig kConfigs[] = {
    {"cpu-fp32", false, false},
    {"gpu-fp32", true,  false},
    {"gpu-fp16", true,  true },
};

// 归一化文本一致性（基于 Levenshtein 距离）：[0,1]，1 = 逐字符完全一致。
double text_similarity(const std::string& a, const std::string& b) {
    const size_t n = a.size(), m = b.size();
    if (n == 0 && m == 0) return 1.0;
    if (n == 0 || m == 0) return 0.0;
    std::vector<size_t> prev(m + 1), cur(m + 1);
    for (size_t j = 0; j <= m; j++) prev[j] = j;
    for (size_t i = 1; i <= n; i++) {
        cur[0] = i;
        for (size_t j = 1; j <= m; j++) {
            const size_t cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
            cur[j] = std::min(std::min(prev[j] + 1, cur[j - 1] + 1), prev[j - 1] + cost);
        }
        prev.swap(cur);
    }
    return 1.0 - (double)prev[m] / (double)std::max(n, m);
}

} // namespace

int main(int argc, char** argv) {
    enable_utf8_console();
    // 重定向到文件时也按行刷新，便于实时观察进度
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::vector<std::string> args = get_utf8_args(argc, argv);

    std::string model_path = "models/locate-anything-fp16";
    std::string image_path;
    std::string prompt = "human";
    int threads = 4;
    int vulkan_device = 0;
    int max_new = 512;
    int iters = 3;
    bool use_mtp = true;
    bool cpu_only = false;
    std::string save_dir;

    for (size_t i = 1; i < args.size(); i++) {
        const std::string& a = args[i];
        if (a == "--model" && i + 1 < args.size()) model_path = args[++i];
        else if (a == "--image" && i + 1 < args.size()) image_path = args[++i];
        else if (a == "--prompt" && i + 1 < args.size()) prompt = args[++i];
        else if (a == "--threads" && i + 1 < args.size()) {
            threads = std::stoi(args[++i]);
            if (threads <= 0) threads = 4;
        } else if (a == "--vulkan-device" && i + 1 < args.size()) {
            vulkan_device = std::stoi(args[++i]);
            if (vulkan_device < 0) vulkan_device = 0;
        } else if (a == "--max-new-tokens" && i + 1 < args.size()) {
            max_new = std::stoi(args[++i]);
            if (max_new <= 0) max_new = 512;
        } else if (a == "--no-mtp") {
            use_mtp = false;
        } else if (a == "--cpu-only") {
            cpu_only = true;
        } else if (a == "--iter" && i + 1 < args.size()) {
            iters = std::stoi(args[++i]);
            if (iters <= 0) iters = 1;
        } else if (a == "--save-dir" && i + 1 < args.size()) {
            save_dir = args[++i];
        }
    }

    if (image_path.empty()) {
        fprintf(stderr,
            "Usage: %s --image <img> [--model <dir>] [--prompt <q>]\n"
            "       [--threads N] [--vulkan-device <idx>] [--max-new-tokens N] [--no-mtp] [--iter N]\n"
            "       [--save-dir <dir>]\n",
            argv[0]);
        return 2;
    }

    ncnn::Mat bgr = load_image_to_ncnn_mat(image_path);
    if (ncnn_mat_empty(bgr)) {
        fprintf(stderr, "failed to load image: %s\n", image_path.c_str());
        return 1;
    }

    printf("=== bench: model=%s image=%s prompt=\"%s\" threads=%d device=%d max_new=%d iter=%d mode=%s ===\n",
           model_path.c_str(), image_path.c_str(), prompt.c_str(), threads, vulkan_device,
           max_new, iters, use_mtp ? "mtp" : "ar");

    LocateGenerateConfig cfg;
    cfg.max_new_tokens = max_new;
    cfg.do_sample = false;  // greedy：确定性解码，保证跨平台可比
    cfg.use_mtp = use_mtp;
    cfg.repetition_penalty = 1.0f;
    // callback 留空：bench 只关心最终文本与框，不逐 token 打印

    std::string ref_text;
    bool have_ref = false;

    for (const auto& c : kConfigs) {
        if (cpu_only && c.vulkan) continue;   // --cpu-only：只跑 cpu-fp32，跳过两个 GPU 配置
        printf("--- %s ---\n", c.label);
        std::string out;
        std::vector<LocateBox> boxes;
        std::chrono::duration<double> elapsed(0);
        {
            ncnn_llm_locateanything la(model_path, c.vulkan, threads, vulkan_device, c.fp16);
            if (!la.ok()) {
                fprintf(stderr, "  FAIL construct/load model\n");
                continue;
            }
            // warmup 1 次：模型首次 run 含懒初始化，不计入计时
            la.run(bgr, prompt, cfg);
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; i++) out = la.run(bgr, prompt, cfg);
            auto t1 = std::chrono::steady_clock::now();
            elapsed = t1 - t0;
            boxes = la.last_boxes();
        }  // la 析构，释放模型/显存后再跑下一个配置

        const double avg = elapsed.count() / iters;
        printf("  end-to-end avg=%.3f s  (total=%.3f s / %d iter)\n", avg, elapsed.count(), iters);

        // 检测坐标（归一化 + 像素）
        printf("  boxes=%zu\n", boxes.size());
        for (size_t i = 0; i < boxes.size(); i++) {
            printf("    %s\n", format_locate_box(boxes[i], bgr.w, bgr.h, (int)i).c_str());
        }
        if (!save_dir.empty()) {
            std::string p = save_dir;
            if (p.back() != '/' && p.back() != '\\') p += "/";
            p += std::string(c.label) + ".png";
            if (draw_locate_boxes(bgr, boxes, p)) printf("  annotated: %s\n", p.c_str());
            else fprintf(stderr, "  failed to write annotated image: %s\n", p.c_str());
        }

        if (!have_ref) {
            ref_text = out;
            have_ref = true;
            printf("  reference(cpu-fp32) len=%zu out=\"%s\"\n", ref_text.size(), ref_text.c_str());
        } else {
            const bool identical = (out == ref_text);
            printf("  len=%zu sim(vs cpu-fp32)=%.4f identical=%s out=\"%s\"\n",
                   out.size(), text_similarity(ref_text, out), identical ? "yes" : "no",
                   out.c_str());
        }
    }
    return 0;
}