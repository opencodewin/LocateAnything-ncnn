#include "image_utils.h"

#include <algorithm>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

ncnn::Mat load_image_to_ncnn_mat(const std::string& image_path) {
    int w, h, c;
    unsigned char* data = stbi_load(image_path.c_str(), &w, &h, &c, 3);
    if (!data) {
        return ncnn::Mat();
    }

    ncnn::Mat bgr(w, h, 3, (size_t)1u);
    unsigned char* dst = (unsigned char*)bgr.data;
    
    for (int i = 0; i < w * h; i++) {
        dst[i * 3 + 0] = data[i * 3 + 2];
        dst[i * 3 + 1] = data[i * 3 + 1];
        dst[i * 3 + 2] = data[i * 3 + 0];
    }

    stbi_image_free(data);
    return bgr;
}

ncnn::Mat ncnn_mat_resize(const ncnn::Mat& src, int target_w, int target_h) {
    if (ncnn_mat_empty(src)) {
        return ncnn::Mat();
    }

    ncnn::Mat dst(target_w, target_h, 3, (size_t)1u);
    
    ncnn::resize_bilinear_c3((const unsigned char*)src.data, src.w, src.h, src.w * 3,
                              (unsigned char*)dst.data, target_w, target_h, target_w * 3);

    return dst;
}

ncnn::Mat ncnn_mat_resize_bicubic(const ncnn::Mat& src, int target_w, int target_h) {
    if (ncnn_mat_empty(src)) {
        return ncnn::Mat();
    }

    // ncnn::resize_bicubic works on planar float Mats. Convert the interleaved
    // BGR u8 buffer to planar BGR float, resize, then pack back to interleaved u8.
    ncnn::Mat src_planar(src.w, src.h, 3);
    for (int y = 0; y < src.h; y++) {
        const unsigned char* row = (const unsigned char*)src.data + (size_t)y * src.w * 3;
        float* b = src_planar.channel(0).row(y);
        float* g = src_planar.channel(1).row(y);
        float* r = src_planar.channel(2).row(y);
        for (int x = 0; x < src.w; x++) {
            b[x] = (float)row[x * 3 + 0];
            g[x] = (float)row[x * 3 + 1];
            r[x] = (float)row[x * 3 + 2];
        }
    }

    ncnn::Mat dst_planar(target_w, target_h, 3);
    ncnn::resize_bicubic(src_planar, dst_planar, target_w, target_h);

    ncnn::Mat dst(target_w, target_h, 3, (size_t)1u);
    unsigned char* dst_data = (unsigned char*)dst.data;
    for (int y = 0; y < target_h; y++) {
        const float* b = dst_planar.channel(0).row(y);
        const float* g = dst_planar.channel(1).row(y);
        const float* r = dst_planar.channel(2).row(y);
        unsigned char* dst_row = dst_data + (size_t)y * target_w * 3;
        for (int x = 0; x < target_w; x++) {
            dst_row[x * 3 + 0] = (unsigned char)std::min(255.0f, std::max(0.0f, b[x]));
            dst_row[x * 3 + 1] = (unsigned char)std::min(255.0f, std::max(0.0f, g[x]));
            dst_row[x * 3 + 2] = (unsigned char)std::min(255.0f, std::max(0.0f, r[x]));
        }
    }

    return dst;
}

// ============================================================================
// Lanczos (a=3) resize —— 复刻 torch 参考 hybrid_runtime.load_pil 的 MAX_DIM
// 预缩 kernel（PIL Image.Resampling.LANCZOS）：sinc(x)*sinc(x/3)，support=3。
// 输入/输出都是 BGR u8 interleaved（elemsize=1），可分离：先水平后垂直。
// 用 scale 感知的抽取：下采样时核在输入域展开 radius≈3*scale，并除以 scale
// 归一化到核的单位支撑 —— 与 PIL 对降采样的处理一致。
// ============================================================================
static double lanczos3_sinc(double v) {
    if (std::fabs(v) < 1e-8) return 1.0;
    const double t = v * 3.14159265358979323846;
    return std::sin(t) / t;
}
static double lanczos3_kernel(double x) {
    if (std::fabs(x) >= 3.0) return 0.0;
    return lanczos3_sinc(x) * lanczos3_sinc(x / 3.0);
}

// 对 interleaved u8 的单一 __轴__ 做 lanczos 重采样（读方向长 n_in、写方向长
// n_out，通道数 c）。stride_w = 相邻像素字节数（水平=3，垂直=行长 target_w*3）。
static void lanczos_resample_axis(const unsigned char* src, unsigned char* dst,
                                  int n_in, int n_out, int c,
                                  int stride_w, int out_stride_w) {
    const double scale = (double)n_in / (double)n_out;   // 输入像素 / 输出像素
    const int radius = (int)std::ceil(3.0 * std::max(1.0, scale)) + 1;
    for (int o = 0; o < n_out; o++) {
        const double co = (o + 0.5) * scale - 0.5;       // 输出中心对应的输入坐标
        const int base = (int)std::floor(co);
        double acc[3] = {0.0, 0.0, 0.0}, wsum = 0.0;
        for (int i = base - radius; i <= base + radius; i++) {
            if (i < 0 || i >= n_in) continue;
            const double w = lanczos3_kernel((co - i) / scale);
            const unsigned char* p = src + (size_t)i * stride_w;
            for (int cc = 0; cc < c; cc++) acc[cc] += w * (double)p[cc];
            wsum += w;
        }
        unsigned char* q = dst + (size_t)o * out_stride_w;
        for (int cc = 0; cc < c; cc++) {
            double v = acc[cc] / wsum;
            q[cc] = (unsigned char)std::min(255.0, std::max(0.0, v + (v < 0.0 ? -0.5 : 0.5)));
        }
    }
}

ncnn::Mat ncnn_mat_resize_lanczos(const ncnn::Mat& src, int target_w, int target_h) {
    if (ncnn_mat_empty(src)) {
        return ncnn::Mat();
    }

    // 1) 水平：BGR u8 interleaved -> tmp(target_w, in_h)
    ncnn::Mat tmp(target_w, src.h, 3, (size_t)1u);
    for (int y = 0; y < src.h; y++) {
        lanczos_resample_axis((const unsigned char*)src.data + (size_t)y * src.w * 3,
                              (unsigned char*)tmp.data + (size_t)y * target_w * 3,
                              src.w, target_w, 3, 3, 3);
    }

    // 2) 垂直：tmp(target_w, in_h) -> dst(target_w, target_h)
    ncnn::Mat dst(target_w, target_h, 3, (size_t)1u);
    for (int x = 0; x < target_w; x++) {
        lanczos_resample_axis((const unsigned char*)tmp.data + (size_t)x * 3,
                              (unsigned char*)dst.data + (size_t)x * 3,
                              src.h, target_h, 3, target_w * 3, target_w * 3);
    }
    return dst;
}

bool ncnn_mat_empty(const ncnn::Mat& mat) {
    return mat.empty() || mat.w <= 0 || mat.h <= 0;
}