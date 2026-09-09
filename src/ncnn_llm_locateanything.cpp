#include "ncnn_llm_locateanything.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>

#include "utils/image_utils.h"

using json = nlohmann::json;

// ============================================================================
// 工具：把 BGR u8 interleaved（load_image_to_ncnn_mat 结果）适配到 vision_embed
// 需要的 [1,3,H,W]、RGB、mean=std=0.5 归一化张量。
// 与 torch LocateAnythingImageProcessor.rescale 保持一致：直接双三次拉伸到
// 变体画布（448/896/1792），无 letterbox/pad。torch 侧是
//     target = ceil(尺寸 / (merge*patch)) * (merge*patch)，然后 BICUBIC resize
// 画布（正方形变体）即该目标；stretch 后模型输出的 0~1000 坐标为相对画布的
// 均匀映射，调用方按 x/1000*原宽 还原即可，无填充偏移。
// ============================================================================
static ncnn::Mat bgr_to_rgb_chw_normalized(const ncnn::Mat& bgr, int target_w, int target_h) {
    ncnn::Mat rgb(target_w, target_h, 3);
    rgb.fill(0.f);

    if (ncnn_mat_empty(bgr)) {
        return rgb;
    }

    ncnn::Mat resized = ncnn_mat_resize_bicubic(bgr, target_w, target_h);
    if (ncnn_mat_empty(resized)) {
        return rgb;
    }

    const unsigned char* rp = (const unsigned char*)resized.data;
    for (int y = 0; y < target_h; y++) {
        for (int x = 0; x < target_w; x++) {
            int si = (y * target_w + x) * 3;
            unsigned char B = rp[si + 0];
            unsigned char G = rp[si + 1];
            unsigned char R = rp[si + 2];
            // 归一化 (v / 255 - mean) / std，mean=std=0.5 -> v/255*2 - 1
            float fr = R / 255.f * 2.f - 1.f;
            float fg = G / 255.f * 2.f - 1.f;
            float fb = B / 255.f * 2.f - 1.f;
            rgb.channel(0).row(y)[x] = fr;
            rgb.channel(1).row(y)[x] = fg;
            rgb.channel(2).row(y)[x] = fb;
        }
    }
    return rgb;
}

// ============================================================================
// 构造 / 加载
// ============================================================================
ncnn_llm_locateanything::ncnn_llm_locateanything(const std::string& model_path,
                                                 bool use_vulkan, int num_threads,
                                                 int vulkan_device, bool use_fp16)
    : ncnn_llm_base(use_vulkan, num_threads > 0 ? num_threads : 4,
                    vulkan_device, use_fp16) {
    try {
        json config;
        {
            std::ifstream ifs(model_path + "/model.json");
            if (!ifs.is_open()) {
                fprintf(stderr, "[locateanything] cannot open %s/model.json\n", model_path.c_str());
                return;
            }
            ifs >> config;
        }
        model_type_ = config.value("model_type", config.value("type", std::string("locate_anything")));
        model_path_ = model_path;
        vulkan_ = use_vulkan;

        auto load_net = [&](const std::string& key, bool vk) {
            auto net = std::make_shared<ncnn::Net>();
            net->opt = create_option();
            net->opt.use_vulkan_compute = vk;
            // 视觉链与 CPU 副本固定 fp32，与 torch 参考一致；fp16 只开给 Vulkan 文本链路。
            if (!vk) {
                net->opt.use_fp16_packed = false;
                net->opt.use_fp16_storage = false;
                net->opt.use_fp16_arithmetic = false;
            }
            std::string p = model_path + "/" + config["params"][key]["param"].get<std::string>();
            std::string b = model_path + "/" + config["params"][key]["bin"].get<std::string>();
            if (net->load_param(p.c_str()) != 0 || net->load_model(b.c_str()) != 0) {
                fprintf(stderr, "[locateanything] fail to load %s (vk=%d)\n", key.c_str(), vk);
                return std::shared_ptr<ncnn::Net>();
            }
            return net;
        };
        // 视觉链固定 CPU(f32)：视觉特征(fp16 污染会毁整句)，文本默认同走 CPU；仅 --vulkan 时文本实验性走 GPU
        net_vision_embed_ = load_net("vision_embed", /*vk=*/false);
        net_vision_encoder_ = load_net("vision_encoder", /*vk=*/false);
        net_vision_projector_ = load_net("vision_projector", /*vk=*/false);
        net_text_embed_ = load_net("text_embed", vulkan_);
        net_text_decoder_ = load_net("text_decoder", vulkan_);
        net_lm_head_ = load_net("lm_head", vulkan_);
        // CPU 副本仅供 LA_SELFTEST 对比；纯 CPU 模式主副本即 CPU，直接别名省内存
        if (vulkan_) {
            net_text_decoder_cpu_ = load_net("text_decoder", /*vk=*/false);
            net_lm_head_cpu_ = load_net("lm_head", /*vk=*/false);
        } else {
            net_text_decoder_cpu_ = net_text_decoder_;
            net_lm_head_cpu_ = net_lm_head_;
        }
        // LA_SELFTEST=1：逐子图 CPU vs Vulkan 对比（诊断用，加载 CPU 侧视觉图）
        selftest_ = std::getenv("LA_SELFTEST") != nullptr && use_vulkan &&
                    std::getenv("LA_SELFTEST")[0] != '0';
        if (selftest_) {
            st_vision_embed_cpu_ = load_net("vision_embed", /*vk=*/false);
            st_vision_encoder_cpu_ = load_net("vision_encoder", /*vk=*/false);
            st_vision_projector_cpu_ = load_net("vision_projector", /*vk=*/false);
            st_text_embed_cpu_ = load_net("text_embed", /*vk=*/false);
        }
        if (!net_vision_embed_ || !net_vision_encoder_ || !net_vision_projector_ ||
            !net_text_embed_ || !net_text_decoder_ || !net_lm_head_) {
            fprintf(stderr, "[locateanything] some subgraph failed to load\n");
            return;
        }
        if (use_vulkan) {
            printf("[locateanything] Vulkan enabled\n");
        }

        // ---- tokenizer ----
        const json& tj = config["tokenizer"];
        std::string vf = model_path + "/" + tj["vocab_file"].get<std::string>();
        std::string mf = model_path + "/" + tj["merges_file"].get<std::string>();
        std::string type = tj.value("type", std::string("bbpe"));
        bpe_ = std::make_shared<BpeTokenizer>(BpeTokenizer::LoadFromFiles(
            vf, mf, SpecialTokensConfig{}, /*add_special_if_missing=*/true,
            /*fallback_to_chars=*/true, /*use_byte_encoder=*/type == "bbpe"));
        if (tj.contains("additional_special_tokens")) {
            std::vector<std::string> ext = tj["additional_special_tokens"].get<std::vector<std::string>>();
            // vocab.txt 已按 id 摊平（含全部 add 词），AddAdditionalSpecialToken 会命中已有 id，
            // 绝不 append（避免 id 移位）。
            for (const auto& t : ext) {
                bpe_->AddAdditionalSpecialToken(t, /*add_if_missing=*/false);
            }
        }
        if (tj.contains("eos")) eos_id_ = tj["eos"].get<int>();
        if (tj["special_ids"].contains("im_end")) im_end_id_ = tj["special_ids"]["im_end"].get<int>();
        if (tj["special_ids"].contains("image_token")) image_token_id_ = tj["special_ids"]["image_token"].get<int>();

        // ---- text / setting ----
        const json& st = config["setting"];
        hidden_ = st.value("hidden", hidden_);
        head_dim_ = st.value("head_dim", head_dim_);
        layers_ = st.value("layers", layers_);
        kv_heads_ = st.value("kv_heads", kv_heads_);
        heads_ = st.value("heads", heads_);
        vocab_ = st.value("vocab", vocab_);
        rope_theta_ = st.value("rope_theta", rope_theta_);
        mask_fill_ = st.value("mask_fill", mask_fill_);

        // ---- vision（v3 动态：单套动态图 + 宿主插值，无静态变体）----
        const json& vi = st["vision"];
        merge_ = vi.value("merge", merge_);
        v_patch_ = vi.value("patch_size", v_patch_);
        v_in_token_limit_ = vi.value("in_token_limit", v_in_token_limit_);
        vision_head_dim_ = vi.value("vision_head_dim", vision_head_dim_);
        vision_rope_theta_ = vi.value("vision_rope_theta", vision_rope_theta_);
        std::string pe = vi.value("pos_emb_file", std::string("pos_emb.bin"));
        // 原生 64x64 pos_emb（宿主按实际网格插值的源）
        load_pos_emb(model_path + "/" + pe);

        ok_ = true;
        printf("[locateanything] model_type=%s hidden=%d layers=%d kv_heads=%d "
               "image_token_id=%d n_img_tok=%d\n",
               model_type_.c_str(), hidden_, layers_, kv_heads_, image_token_id_, n_image_tokens_);
    } catch (const std::exception& e) {
        fprintf(stderr, "[locateanything] init exception: %s\n", e.what());
        ok_ = false;
    }
}

bool ncnn_llm_locateanything::load_pos_emb(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        fprintf(stderr, "[locateanything] cannot open pos_emb file %s\n", path.c_str());
        return false;
    }
    // expected [1, pos_emb_grid_^2, 1152] fp32（原生 64x64，宿主插值源）
    size_t n = (size_t)pos_emb_grid_ * pos_emb_grid_ * vision_hidden_;
    pos_emb_.create(1152, (int)(pos_emb_grid_ * pos_emb_grid_));
    pos_emb_.fill(0.f);
    ifs.read((char*)pos_emb_.data, (std::streamsize)(n * sizeof(float)));
    return !ifs.fail();
}

// 从 model.json setting.vision.variants 载入各档视觉变体。
// 宿主端 2x2 patch_merger（mirror 官方 patch_merger 的 view+permute 重排，无权重无插值）：
//   body=[L, v_dim]（L=gh*gw）-> out=[(gh/2)*(gw/2), 4*v_dim]。
//   out[rm, cx] 的第 (i,j) 个 merge 块、维度 d 取自 body 行 (rm*2+i)*gw + cx*2 + j。纯 gather。
static void patch_merge_2x2(const ncnn::Mat& body, int gh, int gw, ncnn::Mat& out) {
    const int dim = (int)body.w;         // v_dim（encoder 未 merge 输出 1152）
    const int merge = 2;
    const int mh = gh / merge, mw = gw / merge;
    const int m = mh * mw;
    const int merge_out = merge * merge * dim;
    out.create(merge_out, m);
    out.fill(0.f);
    for (int rm = 0; rm < mh; rm++) {
        for (int cx = 0; cx < mw; cx++) {
            const int t = rm * mw + cx;
            float* orow = out.row(t);
            for (int i = 0; i < merge; i++) {
                for (int j = 0; j < merge; j++) {
                    const int src_row = (rm * merge + i) * gw + cx * merge + j;
                    const float* srow = body.row(src_row);
                    const int od = (i * merge + j) * dim;
                    memcpy(orow + od, srow, (size_t)dim * sizeof(float));
                }
            }
        }
    }
}

// 宿主 pos_emb 插值：把原生 64x64（Learnable2DInterpPosEmb 的 weight）bicubic 到当前
// 网格，输出 [gh*gw, dim]（vision_embed 的 in1）。与 torch F.interpolate(mode="bicubic",
// align_corners=False) 逐位对齐（A=-0.75 系数、边界 clamp）。grid==64 时零插值直返。
void ncnn_llm_locateanything::host_bicubic_pos_emb(int grid_h, int grid_w, ncnn::Mat& out) {
    const int ih = pos_emb_grid_, iw = pos_emb_grid_;   // 原生 64x64
    const int dim = vision_hidden_;
    const int L = grid_h * grid_w;
    if (grid_h == ih && grid_w == iw) { out = pos_emb_; return; }  // 原生网格
    out.create(dim, L);
    auto cubic1 = [](float x, float A) { return ((A + 2.f) * x - (A + 3.f)) * x * x + 1.f; };
    auto cubic2 = [](float x, float A) { return ((A * x - 5.f * A) * x + 8.f * A) * x - 4.f * A; };
    auto coefs = [&cubic1, &cubic2](float t, float c[4]) {
        const float A = -0.75f;
        c[0] = cubic2(t + 1.f, A);
        c[1] = cubic1(t, A);
        c[2] = cubic1(1.f - t, A);
        c[3] = cubic2(2.f - t, A);
    };
    struct Tap { int idx[4]; float w[4]; };
    std::vector<Tap> yt(grid_h), xt(grid_w);
    const float sy_scale = (float)ih / grid_h, sx_scale = (float)iw / grid_w;
    for (int oy = 0; oy < grid_h; oy++) {
        float sy = (oy + 0.5f) * sy_scale - 0.5f;
        int y0 = (int)std::floor(sy); float fy = sy - y0;
        auto cl = [&](int v) -> int { return v < 0 ? 0 : (v >= ih ? ih - 1 : v); };
        int b = y0 - 1;
        yt[oy].idx[0] = cl(b);       yt[oy].idx[1] = cl(b + 1);
        yt[oy].idx[2] = cl(b + 2);   yt[oy].idx[3] = cl(b + 3);
        coefs(fy, yt[oy].w);
    }
    for (int ox = 0; ox < grid_w; ox++) {
        float sx = (ox + 0.5f) * sx_scale - 0.5f;
        int x0 = (int)std::floor(sx); float fx = sx - x0;
        auto cl = [&](int v) -> int { return v < 0 ? 0 : (v >= iw ? iw - 1 : v); };
        int b = x0 - 1;
        xt[ox].idx[0] = cl(b);       xt[ox].idx[1] = cl(b + 1);
        xt[ox].idx[2] = cl(b + 2);   xt[ox].idx[3] = cl(b + 3);
        coefs(fx, xt[ox].w);
    }
    const float* src = (const float*)pos_emb_.data;   // 行优先 [ih*iw, dim]
    std::vector<float> col(4);
    for (int oy = 0; oy < grid_h; oy++) {
        for (int ox = 0; ox < grid_w; ox++) {
            float* orow = out.row(oy * grid_w + ox);
            const Tap& ty = yt[oy];
            const Tap& tx = xt[ox];
            for (int d = 0; d < dim; d++) {
                for (int jx = 0; jx < 4; jx++) {
                    const int cj = tx.idx[jx];
                    const float* a0 = src + ((size_t)ty.idx[0] * iw + cj) * dim + d;
                    const float* a1 = src + ((size_t)ty.idx[1] * iw + cj) * dim + d;
                    const float* a2 = src + ((size_t)ty.idx[2] * iw + cj) * dim + d;
                    const float* a3 = src + ((size_t)ty.idx[3] * iw + cj) * dim + d;
                    col[jx] = ty.w[0] * a0[0] + ty.w[1] * a1[0] + ty.w[2] * a2[0] + ty.w[3] * a3[0];
                }
                orow[d] = tx.w[0] * col[0] + tx.w[1] * col[1] + tx.w[2] * col[2] + tx.w[3] * col[3];
            }
        }
    }
}

void ncnn_llm_locateanything::text_rope_cos_sin(int seq, int pos_start,
                                                ncnn::Mat& cos, ncnn::Mat& sin) {
    std::vector<int> pos(seq);
    for (int s = 0; s < seq; s++) pos[s] = pos_start + s;
    text_rope_cos_sin_at(pos, cos, sin);
}

// RoPE cos/sin 基于每行独立绝对位置（MTP 窗口里 rep 位置 = 末位，后续 mask 位置递增）。
void ncnn_llm_locateanything::text_rope_cos_sin_at(const std::vector<int>& pos,
                                                   ncnn::Mat& cos, ncnn::Mat& sin) {
    const int L = (int)pos.size();
    cos.create(head_dim_, L);
    sin.create(head_dim_, L);
    const int half = head_dim_ / 2;
    std::vector<float> inv_freq(half);
    for (int k = 0; k < half; k++) {
        inv_freq[k] = (float)std::pow(rope_theta_, -2.0 * k / head_dim_);
    }
    for (int s = 0; s < L; s++) {
        double p = pos[s];
        float* cr = cos.row(s);
        float* sr = sin.row(s);
        for (int d = 0; d < head_dim_; d++) {
            int k = d % half;
            double angle = p * (double)inv_freq[k];
            cr[d] = (float)std::cos(angle);
            sr[d] = (float)std::sin(angle);
        }
    }
}

void ncnn_llm_locateanything::moon_rope_cos(int grid_h, int grid_w,
                                            ncnn::Mat& cos, ncnn::Mat& sin) {
    const int L = grid_h * grid_w;
    int quarter = vision_head_dim_ / 4;             // 72/4 = 18
    int half = vision_head_dim_ / 2;                // 36
    cos.create(half, L);
    sin.create(half, L);
    std::vector<float> inv(quarter);
    for (int j = 0; j < quarter; j++) {
        inv[j] = (float)std::pow(vision_rope_theta_, -4.0 * j / vision_head_dim_);
    }
    for (int p = 0; p < L; p++) {
        double x = (int)(p % grid_w);
        double y = (int)(p / grid_w);
        float* cr = cos.row(p);
        float* sr = sin.row(p);
        for (int j = 0; j < quarter; j++) {
            double ax = x * inv[j];
            double ay = y * inv[j];
            cr[2 * j] = (float)std::cos(ax);
            cr[2 * j + 1] = (float)std::cos(ay);
            sr[2 * j] = (float)std::sin(ax);
            sr[2 * j + 1] = (float)std::sin(ay);
        }
    }
}

ncnn::Mat ncnn_llm_locateanything::causal_mask(int seq) {
    ncnn::Mat mask(seq, seq);
    mask.fill(0.f);
    for (int i = 0; i < seq; i++) {
        float* row = mask.row(i);
        for (int j = i + 1; j < seq; j++) {
            row[j] = mask_fill_;
        }
    }
    return mask;
}

// ============================================================================
// 子图驱动
// ============================================================================

// 子图输出的"规范化"。
//
// 背景：宿主代码按"规范 fp32 2D Mat（w=dim，h=rows，逐行连续）"解释每个子图的输出
// （row_of / hidden.row(i) 直接 memcpy hidden_*4 字节）。但 ncnn 各后端的 packing
// 支持是**按架构实现**的（arm neon / x86 sse-avx / avx512 各不相同），同一个导出图
// 在不同 CPU 上可能返回 elempack=4 或 8 的 Mat —— 此时 w 变成 dim/pack，memcpy 会
// 静默少拷，解码出的坐标就是垃圾（表现为"一个框都画不出来"）。视觉链之前就踩过同一
// 个坑（见 run_vision_features 里对 encoder 输出的重新装箱）。
//
// 因此所有子图输出统一先解包到 elempack=1，再摊平成规范 2D（w=dim，h=rows）。
static void la_shape(const char* tag, const ncnn::Mat& m) {
    static std::unordered_set<std::string> seen;
    std::string key = tag;
    if (seen.insert(key).second)
        fprintf(stderr, "[locateanything] SHAPE %s dims=%d w=%d h=%d c=%d pack=%d es=%zu\n",
                tag, m.dims, m.w, m.h, m.c, m.elempack, m.elemsize);
}

// 一行统计：w/h/min/max/nan 个数。NaN 会让 greedy argmax 恒返回下标 0（比较全 false），
// 表现为 commit 全 0、一个框都解不出来，所以这里显式把 NaN 数出来。
static void la_stat_line(const char* tag, const ncnn::Mat& m) {
    if (m.empty()) {
        fprintf(stderr, "[locateanything] STAT %s EMPTY (w=%d h=%d c=%d pack=%d)\n", tag, m.w, m.h,
                m.c, m.elempack);
        return;
    }
    const size_t n = (size_t)m.w * m.h * ((m.dims == 3) ? (size_t)m.c : 1u) * m.elempack;
    const float* p = (const float*)m.data;
    float mn = 1e30f, mx = -1e30f;
    size_t nan_cnt = 0;
    for (size_t i = 0; i < n; i++) {
        float v = p[i];
        if (std::isnan(v)) { nan_cnt++; continue; }
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    fprintf(stderr, "[locateanything] STAT %s w=%d h=%d c=%d pack=%d n=%zu min=%.4f max=%.4f nan=%zu\n",
            tag, m.w, m.h, m.c, m.elempack, n, mn, mx, nan_cnt);
}

// 单行 logits 的指纹：V / 最大值 / argmax / NaN 数（V=0 时 argmax 必然为 0）
static void la_logits_line(const char* tag, const ncnn::Mat& row) {
    if (row.empty()) {
        fprintf(stderr, "[locateanything] LOGITS %s EMPTY (w=%d h=%d)\n", tag, row.w, row.h);
        return;
    }
    const int V = row.w;
    const float* p = (const float*)row.data;
    float mx = -1e30f;
    int arg = 0;
    size_t nan_cnt = 0;
    for (int i = 0; i < V; i++) {
        if (std::isnan(p[i])) { nan_cnt++; continue; }
        if (p[i] > mx) { mx = p[i]; arg = i; }
    }
    fprintf(stderr, "[locateanything] LOGITS %s V=%d max=%.4f argmax=%d nan=%zu\n", tag, V, mx, arg,
            nan_cnt);
}

// elempack -> 1（已是 1 时零成本返回原 Mat）
static ncnn::Mat la_unpack(const ncnn::Mat& src, const char* tag) {
    if (src.elempack == 1) return src;
    ncnn::Mat dst;
    ncnn::Option opt;
    opt.blob_allocator = nullptr;
    opt.workspace_allocator = nullptr;
    ncnn::convert_packing(src, dst, 1, opt);
    if (dst.empty()) {
        fprintf(stderr, "[locateanything] WARN %s: unpack pack%d failed\n", tag, src.elempack);
        return src;
    }
    fprintf(stderr, "[locateanything] NOTE %s: unpack pack%d -> pack1 (w=%d h=%d c=%d)\n",
            tag, src.elempack, dst.w, dst.h, dst.c);
    return dst;
}

// 摊平成规范 2D Mat（w=expect_w，h=行数）。dims=3 且 c>1 时按"c 主序、y 次序"摊平。
static ncnn::Mat la_canonical_2d(const ncnn::Mat& src, int expect_w, const char* tag) {
    la_shape(tag, src);
    ncnn::Mat m = la_unpack(src, tag);
    if (m.dims == 2 && m.w == expect_w) return m;
    if (m.dims == 1 && m.w == expect_w) {
        ncnn::Mat out(expect_w, 1);
        memcpy(out.data, m.data, (size_t)expect_w * sizeof(float));
        return out;
    }
    if (m.dims == 3 && m.w == expect_w) {
        ncnn::Mat out(m.w, m.h * m.c);
        for (int c = 0; c < m.c; c++)
            for (int y = 0; y < m.h; y++)
                memcpy(out.row(c * m.h + y), m.channel(c).row(y), (size_t)m.w * sizeof(float));
        return out;
    }
    if (m.w != expect_w)
        fprintf(stderr, "[locateanything] WARN %s: w=%d != expect %d (dims=%d h=%d c=%d)\n",
                tag, m.w, expect_w, m.dims, m.h, m.c);
    return m;
}

// KV cache：保持 dims=3（w=head_dim, h=rows, c=kv_heads），只解包
static ncnn::Mat la_canonical_kv(const ncnn::Mat& src, const char* tag) {
    la_shape(tag, src);
    ncnn::Mat m = la_unpack(src, tag);
    if (m.dims != 3)
        fprintf(stderr, "[locateanything] WARN %s: unexpected dims=%d (expect 3)\n", tag, m.dims);
    return m;
}

ncnn::Mat ncnn_llm_locateanything::run_text_embed(const std::vector<int>& ids) {
    ncnn::Mat in((int)ids.size(), 1, (void*)ids.data());
    in = in.clone();
    ncnn::Mat out;
    ncnn::Extractor ex = net_text_embed_->create_extractor();
    ex.input("in0", in);
    ex.extract("out0", out);
    return la_canonical_2d(out, hidden_, "text_embed.out0");
}

ncnn::Mat ncnn_llm_locateanything::run_decoder(const ncnn::Mat& emb, const ncnn::Mat& cos,
                                               const ncnn::Mat& sin, const ncnn::Mat& mask,
                                               KVCache& kv, bool is_prefill) {
    // 文本统一走主副本（--vulkan 时为 Vulkan，含 decode；否则为 CPU）
    ncnn::Net* net = net_text_decoder_.get();
    ncnn::Mat out;
    ncnn::Extractor ex = net->create_extractor();
    ex.input("in0", emb);
    ex.input("in1", cos);
    ex.input("in2", sin);
    ex.input("in3", mask);

    // KV 输入
    for (int i = 0; i < layers_; i++) {
        if (is_prefill) {
            // 通道数 = kv_heads（导出图在接口暴露的 cache 通道即 GQA 的 kv 头数；
            // k/v 内部 Tile 到 heads 再喂 SDPA 发生在图内）；长度 0 表示空 cache
            ncnn::Mat empty(head_dim_, 0, kv_heads_);
            ex.input(("cache_k" + std::to_string(i)).c_str(), empty);
            ex.input(("cache_v" + std::to_string(i)).c_str(), empty.clone());
        } else {
            ex.input(("cache_k" + std::to_string(i)).c_str(), kv[i].first);
            ex.input(("cache_v" + std::to_string(i)).c_str(), kv[i].second);
        }
    }
    // KV 输出
    for (int i = 0; i < layers_; i++) {
        ncnn::Mat ok_, ov_;
        ex.extract(("out_cache_k" + std::to_string(i)).c_str(), ok_);
        ex.extract(("out_cache_v" + std::to_string(i)).c_str(), ov_);
        ok_ = la_canonical_kv(ok_, "decoder.out_cache_k");
        ov_ = la_canonical_kv(ov_, "decoder.out_cache_v");
        if (is_prefill) {
            kv.emplace_back(ok_, ov_);
        } else {
            kv[i].first = ok_;
            kv[i].second = ov_;
        }
    }
    ex.extract("out0", out);
    return la_canonical_2d(out, hidden_, "text_decoder.out0");
}

ncnn::Mat ncnn_llm_locateanything::run_lm_head(const ncnn::Mat& hidden) {
    ncnn::Mat out;
    ncnn::Extractor ex = net_lm_head_->create_extractor();
    ex.input("in0", hidden);
    ex.extract("out0", out);
    la_shape("lm_head.out0", out);
    return la_unpack(out, "lm_head.out0");
}

// ============================================================================
// MTP 并行窗口（block_size=6）解码。移植 torch 的 multi-token-prediction 路径：
//   一次 forward 输入 [rep | mask_tok*(k-1)]（位置 past-1..），KV 只由这些输入
//   （含 mask 占位）的 embed 计算并累积；输出每位置 hidden，lm_head 得到 6 个
//   并行 logits -> decode_bbox_avg 整段提交 <box>..</box>。
// 这与 torch hybrid_runtime 的 N_FUTURE 一致，从而复现 torch 的框（x/y 全对齐）。
// ============================================================================

// ncnn Mat(hidden,K)：rows=位置，w=hidden。取第 i 位置单行成 [hidden,1]。
static ncnn::Mat row_of(const ncnn::Mat& hidden, int i, int hidden_dim) {
    ncnn::Mat row(hidden_dim, 1);
    memcpy(row.data, hidden.row(i), (size_t)hidden_dim * sizeof(float));
    return row;
}

// 构建 [past+K, K] 的 MTP 块掩码（w=keys, h=queries，0=可见, mask_fill_=屏蔽）。
//   前 (K-block) 行：重喂的真实 token，纯因果（query q 只见 key <= past+q）。
//   后 block 行（窗口）：窗口内双向可见（对窗口自己的 6 个 key 全开），且能看
//   到窗口前所有历史 key；仅屏蔽窗口前一键（索引 past+K-block-1），等价于 torch
//   build_magi_scheduler_ranges 的 mtp_window 规则（blocked_k = k0-1）。
ncnn::Mat ncnn_llm_locateanything::mtp_mask(int K, int past) {
    ncnn::Mat mask(past + K, K);   // w=keys, h=queries
    mask.fill(0.f);
    const int w0 = K - block_size_;              // 窗口首行
    if (w0 > 0) {
        // 前 w0 行纯因果：屏蔽 key > past+q
        for (int q = 0; q < w0; q++) {
            float* r = mask.row(q);
            for (int j = past + q + 1; j < past + K; j++) r[j] = mask_fill_;
        }
    }
    // 窗口行：屏蔽窗口前一键（索引 past+w0-1），其余全可见（含窗口内双向）
    if (past + w0 - 1 >= 0) {
        for (int q = w0; q < K; q++) mask.row(q)[past + w0 - 1] = mask_fill_;
    }
    return mask;
}

// handle_pattern（hybrid 模式）等价实现：给定 6 长度 nt（x0 或 decode_bbox 结果），
// 返回要提交的 token 序列，并置 type。
static std::vector<int> la_handle_pattern(const std::vector<int>& nt, int box_start, int box_end,
                                          int coord_start, int coord_end, int ref_end,
                                          int null_id, int im_end_id, int none_id,
                                          std::string& type) {
    if (nt.empty()) { type = "im_end"; return {im_end_id}; }
    if (nt[0] == null_id || nt[0] == im_end_id) {
        type = "im_end"; return {im_end_id};
    }
    if (nt[0] == box_start && nt.size() >= 2 && nt[1] == none_id) {
        type = "empty_box"; return {box_start, none_id, box_end};
    }
    if (nt[0] == box_start) {
        int coord_ix = 1;
        std::vector<int> coords(nt.begin() + 1, nt.begin() + std::min((size_t)5, nt.size()));
        for (int c : coords) {
            if (c >= coord_start && c <= coord_end) coord_ix++;
            else break;
        }
        if (coord_ix == 5 && nt.size() >= 6 && nt[5] == box_end) {
            type = "coord_box"; return nt;                 // <box><x1><x2><y1><y2></box>
        }
        if (coord_ix == 3 && nt.size() >= 4 && nt[3] == box_end) {
            type = "point_box"; return {nt[0], nt[1], nt[2], nt[3]};   // <box><x><y></box>
        }
        type = "error_box";                              // hybrid: 切 AR
        return std::vector<int>(nt.begin(), nt.begin() + coord_ix);
    }
    std::vector<int> t = nt;
    for (size_t i = 0; i < t.size(); i++) {
        if (t[i] == null_id) { t.resize(i); break; }
    }
    if (t.size() >= 2 && t[t.size() - 1] == ref_end && t[t.size() - 2] == ref_end) t.pop_back();
    type = "ref_object";
    return t;
}

// 单行采样：greedy(do_sample=false) -> argmax；否则 temperature/top_k/top_p + 类别采样。
int ncnn_llm_locateanything::sample_logits_row(const float* lp, int V,
                                               const LocateGenerateConfig& cfg) {
    if (!cfg.do_sample) {
        int best = 0;
        for (int i = 1; i < V; i++) if (lp[i] > lp[best]) best = i;
        return best;
    }
    std::vector<float> probs((size_t)V);
    float t = cfg.temperature > 0.f ? cfg.temperature : 1.f;
    float maxv = lp[0];
    for (int i = 1; i < V; i++) if (lp[i] > maxv) maxv = lp[i];
    double sum = 0.0;
    for (int i = 0; i < V; i++) {
        probs[i] = (float)std::exp((double)(lp[i] - maxv) / t);
        sum += probs[i];
    }
    for (int i = 0; i < V; i++) probs[i] = (float)(probs[i] / sum);
    if (cfg.top_k > 0 && cfg.top_k < V) {
        std::vector<float> tmp = probs;
        std::nth_element(tmp.begin(), tmp.end() - cfg.top_k, tmp.end());
        float thr = tmp[tmp.size() - cfg.top_k];
        for (float& v : probs) if (v < thr) v = 0.f;
    }
    if (cfg.top_p < 1.f) {
        std::vector<std::pair<float, int>> vs;
        for (int i = 0; i < V; i++) vs.emplace_back(probs[i], i);
        std::sort(vs.begin(), vs.end(), std::greater<>());
        float cum = 0.f;
        std::vector<int> keep((size_t)V, 0);
        for (size_t i = 0; i < vs.size() && cum < cfg.top_p; i++) { keep[vs[i].second] = 1; cum += vs[i].first; }
        for (int i = 0; i < V; i++) if (!keep[i]) probs[i] = 0.f;
    }
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng_);
}

// decode_ref（torch generate_utils）：坐标解码失败时，若首 token 是 <ref> 且概率达标，
// 把后续每个位置取 top-keep_k 中最高概率的**非坐标** token（文本指代），拼成
// [ref_start, ...] 返回；失败返回空。
static std::vector<int> la_decode_ref(const std::vector<std::vector<float>>& probs, int V,
                                      int ref_start, int coord_start, int coord_end,
                                      int keep_k = 5, float start_thresh = 0.6f) {
    if (probs[0][ref_start] < start_thresh) return {};
    const int L = (int)probs.size();
    std::vector<int> out;
    for (int row = 1; row < L; row++) {
        const auto& pr = probs[row];
        // 模拟 torch topk(keep_k) 后取最高概率的合法(非坐标)token
        std::vector<float> topv(keep_k, -1e30f);
        std::vector<int> topi(keep_k, -1);
        for (int t = 0; t < V; t++) {
            for (int q = 0; q < keep_k; q++) {
                if (pr[t] > topv[q]) {
                    for (int r = keep_k - 1; r > q; r--) { topv[r] = topv[r - 1]; topi[r] = topi[r - 1]; }
                    topv[q] = pr[t]; topi[q] = t;
                    break;
                }
            }
        }
        int fv = -1;                      // 最高概率的"非坐标"token（ref 文本）
        for (int q = 0; q < keep_k; q++) {
            int id = topi[q];
            if (!(id >= coord_start && id <= coord_end)) { fv = id; break; }  // 第一个合法即最高
        }
        if (fv < 0) return {};
        out.push_back(fv);
    }
    std::vector<int> r;
    r.push_back(ref_start);
    r.insert(r.end(), out.begin(), out.end());
    return r;
}

// 移植 torch generate_utils.sample_tokens + decode_bbox_avg + decode_ref + handle_pattern（hybrid）。
std::vector<int> ncnn_llm_locateanything::mtp_window_decode(const std::vector<ncnn::Mat>& win,
                                                            std::string& type,
                                                            const LocateGenerateConfig& cfg) {
    const int K = (int)win.size();              // block_size_ = 6
    const int V = win[0].w;
    const int keep_k = 4;
    // torch decode_bbox_avg / is_valid_box_frame / uncertainty 路径的默认阈值（注意不是 0.6）
    const float start_thresh = 0.7f, end_thresh = 0.2f;

    // 每行 logits -> softmax 概率
    std::vector<std::vector<float>> probs(K);
    for (int i = 0; i < K; i++) {
        const float* lp = win[i];
        probs[i].resize(V);
        float m = lp[0];
        for (int t = 1; t < V; t++) if (lp[t] > m) m = lp[t];
        double s = 0.0;
        for (int t = 0; t < V; t++) { probs[i][t] = (float)std::exp(lp[t] - m); s += probs[i][t]; }
        for (int t = 0; t < V; t++) probs[i][t] = (float)(probs[i][t] / s);
    }
    // x0：每行采样（torch sample_tokens；greedy=argmax）
    std::vector<int> x0(K);
    for (int i = 0; i < K; i++) x0[i] = sample_logits_row(win[i], V, cfg);

    std::vector<int> nt;                      // 最终交给 handle_pattern 的 token 序列

    // ---- is_valid_box_frame + assemble（decode_bbox_avg, hybrid 阈值 0.7/0.2）----
    bool empty_box =
        probs[0][box_start_id_] >= start_thresh &&
        probs[1][none_id_] > 0.2f && probs[2][box_end_id_] > 0.2f &&
        probs[3][null_id_] > 0.1f && probs[4][null_id_] > 0.1f;
    if (empty_box) {
        // torch: empty_box -> [start, none, box_end, null, null, null] -> handle_pattern -> 3 tokens
        nt = {box_start_id_, none_id_, box_end_id_, null_id_, null_id_, null_id_};
    } else {
        float end_score = probs[5][box_end_id_] + probs[5][null_id_] + probs[5][im_end_id_];
        if (end_score >= end_thresh) {
            // 4 个坐标位置（rows 1..4）各取 top-k 并经"合法坐标 token"过滤
            const int CQ = 4;
            std::vector<int> first_valid_ids(CQ, 0);
            bool ok = true;
            for (int p = 0; p < CQ && ok; p++) {
                const auto& pr = probs[p + 1];
                std::vector<float> topv(keep_k, -1e30f);
                std::vector<int> topi(keep_k, -1);
                for (int t = 0; t < V; t++) {
                    for (int q = 0; q < keep_k; q++) {
                        if (pr[t] > topv[q]) {
                            for (int r = keep_k - 1; r > q; r--) { topv[r] = topv[r - 1]; topi[r] = topi[r - 1]; }
                            topv[q] = pr[t]; topi[q] = t;
                            break;
                        }
                    }
                }
                int fv = -1; float fvp = 0.f; int cnt = 0, vmin = coord_end_id_ + 1, vmax = coord_start_id_ - 1;
                for (int q = 0; q < keep_k; q++) {
                    int id = topi[q];
                    if (id >= coord_start_id_ && id <= coord_end_id_) {
                        if (fv < 0) { fv = id; fvp = topv[q]; }
                        cnt++;
                        if (id < vmin) vmin = id;
                        if (id > vmax) vmax = id;
                    }
                }
                if (fv < 0) { ok = false; break; }
                bool abnormal = (fvp < 0.9f) && (cnt > 1) && ((vmax - vmin) > 60);
                first_valid_ids[p] = abnormal ? 0 : fv;
            }
            if (ok) {
                nt = {box_start_id_, first_valid_ids[0], first_valid_ids[1],
                      first_valid_ids[2], first_valid_ids[3], box_end_id_};
            }
        }
    }

    // 框解码失败 -> decode_ref（<ref> 文本指代）-> 再失败 fallback(zeros)=is_box_empty -> x0
    if (nt.empty()) {
        std::vector<int> ref = la_decode_ref(probs, V, ref_start_id_, coord_start_id_, coord_end_id_);
        if (!ref.empty()) {
            nt = ref;
        } else {
            nt = x0;
        }
    }
    return la_handle_pattern(nt, box_start_id_, box_end_id_, coord_start_id_, coord_end_id_,
                             ref_end_id_, null_id_, im_end_id_, none_id_, type);
}

// 把 KV 裁到前 rows 行（每层 k/v 各 channel 取前 rows*head_dim）。torch 每步 MTP/AR
// forward 后都把 KV 截断到"真实生成 token 数"，丢弃 mask 占位产生的行，这里等价还原。
static void trim_kv(KVCache& kv, int rows) {
    for (auto& pr : kv) {
        if (pr.first.h > rows) {
            ncnn::Mat k(pr.first.w, rows, pr.first.c);
            ncnn::Mat v(pr.second.w, rows, pr.second.c);
            for (int ic = 0; ic < pr.first.c; ic++) {
                memcpy(k.channel(ic), pr.first.channel(ic), (size_t)rows * pr.first.w * sizeof(float));
                memcpy(v.channel(ic), pr.second.channel(ic), (size_t)rows * pr.second.w * sizeof(float));
            }
            pr.first = k;
            pr.second = v;
        }
    }
}

// MTP 并行窗口 + AR 回退 生成循环。
// 对齐 torch stock generate（generation_mode='hybrid'）：
//   - 每步把"上一窗口已提交、但 KV 只有 mask 占位"的真实 token 重喂，再拼 rep+mask×5 窗口；
//   - 前馈后把 KV 截断到真实 token 长度（trim_kv），使后续窗口的注意力键与位置完全一致；
//   - 窗口 6 个并行 logits -> mtp_window_decode（sample_tokens + decode_bbox + handle_pattern）；
//   - error_box 回退 AR 单 token 解码，box_end 再切回 MTP，im_end 终止。
void ncnn_llm_locateanything::decode_loop_mtp(std::string& out_text, std::vector<int>& stream,
                                              KVCache& kv, const LocateGenerateConfig& cfg,
                                              std::unordered_set<int>& history) {
    const int S = (int)stream.size();
    int cache_len = S;                    // stream[0:cache_len) 在 KV 里有真实 K/V
    bool in_mtp = true;
    int limit = S + cfg.max_new_tokens;

    auto emit = [&](const std::vector<int>& toks) {
        for (int t : toks) {
            std::string d = bpe_->decode({t}, true);
            // special token（<box>/<0>~<1000>/...）decode 会跳过 -> 兜底渲染成结构化文本，
            // 否则生成输出里看不到任何坐标。
            if (d.empty()) d = structured_token_text(t);
            if (cfg.callback) cfg.callback(d);
            out_text += d;
        }
    };
    // 单 token 因果前向：位置 pos 的真实 token -> 返回该位置 hidden。
    auto decode_single_causal = [&](int tok, int pos) -> ncnn::Mat {
        ncnn::Mat emb = run_text_embed({tok});          // [hidden,1]
        ncnn::Mat c, s;
        text_rope_cos_sin(1, pos, c, s);
        int past = (int)kv[0].first.h;
        ncnn::Mat m(past + 1, 1);
        m.fill(0.f);
        return run_decoder(emb, c, s, m, kv, false);
    };

    while ((int)stream.size() < limit) {
        if (!in_mtp) {
            // ---- AR 回退（对齐 torch _sample_token_in_ar + 混合终止规则）----
            // 把错误框提交但未缓存的真实 token 逐个补进 KV，取最后一个的 hidden 采样下一个。
            // 终止/切换：box_end -> 切回 MTP；坐标或 none -> 继续 AR；其它任何 token 视为
            // im_end 终止（官方 else 分支）。坐标/NONE/盒尾 token 会在下一轮 grid 里被重喂缓存。
            ncnn::Mat last_hid;
            while (cache_len < (int)stream.size()) {
                int t = stream[cache_len];
                last_hid = decode_single_causal(t, cache_len);   // 因果单 token 前向，回填 KV
                cache_len++;
            }
            int cur = stream.back();
            if (cur == eos_id_ || cur == im_end_id_) break;
            ncnn::Mat lo = run_lm_head(last_hid);
            // rep penalty（greedy 时 rep=1 无影响）+ 采样
            ncnn::Mat l2 = lo.clone();
            if (cfg.repetition_penalty != 1.0f) {
                float* pp = l2;
                for (int id : history) {
                    if (id < l2.w) {
                        float v = pp[id];
                        pp[id] = v > 0.f ? v / cfg.repetition_penalty : v * cfg.repetition_penalty;
                    }
                }
            }
            int next = sample_logits_row(l2, l2.w, cfg);
            stream.push_back(next);
            emit({next});
            history.insert(next);
            // next 尚未写入 KV（cache_len 仍==回填后长度），留到下一轮 blk/回填处理。
            bool coord_or_none =
                (next >= coord_start_id_ && next <= coord_end_id_) || next == none_id_;
            if (next == eos_id_ || next == im_end_id_) break;
            if (next == box_end_id_) { in_mtp = true; continue; }   // box_end_ar -> 切 MTP
            if (!coord_or_none) break;                               // 其余 token = im_end -> 终止
            continue;                                                // coord_ar 继续 AR
        }

        // ---- MTP 窗口 ----
        int L = (int)stream.size();
        int past = (int)kv[0].first.h;
        fprintf(stderr, "[locateanything] WINSTATE L=%d past=%d cache_len=%d blk=%d kvh(w=%d h=%d c=%d pack=%d)\n",
                L, past, cache_len, block_size_, kv[0].first.w, kv[0].first.h, kv[0].first.c,
                kv[0].first.elempack);
        // 块 = [未缓存真实 token][rep=stream[-1]][mask_tok × (block-1)]
        std::vector<int> blk;
        for (int i = cache_len; i < L; i++) blk.push_back(stream[i]);
        blk.push_back(stream[L - 1]);                     // rep（= 末位真实 token，再喂一次）
        for (int j = 0; j < block_size_ - 1; j++) blk.push_back(mask_tok_id_);
        const int K = (int)blk.size();

        // 块内逐行绝对位置：重喂的真实 token = 其 stream 下标；窗口首行（rep 作为 query）
        // = L（第一个新 token）；其余窗口行 = L+1..L+block-1。窗口行必须落在"新位置",
        // 而非偏 -1 到 [L-1]，否则首个生成 token 会占用最后 prompt/box 的位置（RoPE 错位
        // => 模型复读同一框）。见 mtp_mask 的 blocked_k = k0-1（k0 = L）。
        std::vector<int> bpos(K);
        int m = 0;
        for (int i = cache_len; i < L; i++) bpos[m++] = i;
        bpos[m++] = L;
        for (int j = 1; j < block_size_; j++) bpos[m++] = L + j;

        ncnn::Mat blk_emb = run_text_embed(blk);          // [hidden,K]
        la_stat_line("blk_emb", blk_emb);
        ncnn::Mat cos, sin;
        text_rope_cos_sin_at(bpos, cos, sin);
        ncnn::Mat mask = mtp_mask(K, past);
        ncnn::Mat hidden = run_decoder(blk_emb, cos, sin, mask, kv, false);   // [hidden,K]

        // 窗口 logits = 最后 block_size_ 行
        int w0 = K - block_size_;
        std::vector<ncnn::Mat> win;
        la_stat_line("decoder.window_hidden", hidden);
        for (int i = w0; i < K; i++) {
            win.push_back(run_lm_head(row_of(hidden, i, hidden_)));   // w=vocab
        }
        la_logits_line("win0", win[0]);
        la_logits_line("winLast", win[K - w0 - 1]);
        std::string type;
        std::vector<int> commit = mtp_window_decode(win, type, cfg);

        fprintf(stderr, "[locateanything] WINDOW type=%s commit=[", type.c_str());
        for (size_t z = 0; z < commit.size(); z++) fprintf(stderr, "%s%d", z ? "," : "", commit[z]);
        fprintf(stderr, "] stream=%zu limit=%d\n", stream.size(), limit);

        for (int t : commit) stream.push_back(t);
        emit(commit);
        if (type == "im_end") break;
        trim_kv(kv, L);                                   // 丢弃窗口，KV 只留真实 [0:L)
        fprintf(stderr, "[locateanything] KVSTAT after-trim rows=%d (w=%d h=%d c=%d pack=%d) expect=%d\n",
                (int)kv[0].first.h, kv[0].first.w, kv[0].first.h, kv[0].first.c,
                kv[0].first.elempack, L);
        cache_len = L;                                    // 真实 token 缓存到 L（提交前）
        if (type == "error_box") in_mtp = false;          // 出错 -> AR
        // coord_box / point_box / empty_box / ref_object 继续 MTP
    }
    fprintf(stderr, "[locateanything] LOOP-END stream=%zu limit=%d last=%d\n", stream.size(), limit, stream.empty() ? -1 : stream.back());
}

// 纯逐 token AR 生成循环：每步在增量位置因果前向、采样 1 个新 token 追加进 KV，
// 直到遇到 eos/im_end 或达 max_new_tokens。与 decode_loop_mtp 的"AR 回退"供应商相同，
// 但全程不用 MTP 窗口，用于对比 MTP 与纯 AR 的输出差异。
void ncnn_llm_locateanything::decode_loop_ar(std::string& out_text, std::vector<int>& stream,
                                             KVCache& kv, const LocateGenerateConfig& cfg,
                                             std::unordered_set<int>& history) {
    int limit = (int)stream.size() + cfg.max_new_tokens;

    auto emit = [&](const std::vector<int>& toks) {
        for (int t : toks) {
            std::string d = bpe_->decode({t}, true);
            if (d.empty()) d = structured_token_text(t);   // 结构 token 兜底渲染（同 MTP）
            if (cfg.callback) cfg.callback(d);
            out_text += d;
        }
    };
    // 单 token 因果前向：位置 pos 的真实 token -> 返回该位置 hidden（KV 追加一行）。
    auto decode_single_causal = [&](int tok, int pos) -> ncnn::Mat {
        ncnn::Mat emb = run_text_embed({tok});          // [hidden,1]
        ncnn::Mat c, s;
        text_rope_cos_sin(1, pos, c, s);
        int past = (int)kv[0].first.h;
        ncnn::Mat m(past + 1, 1);
        m.fill(0.f);
        return run_decoder(emb, c, s, m, kv, false);
    };

    int pos = (int)stream.size();   // 下一个要解码的真实位置（全文皆已预填充，KV 长度==stream 大小）
    while ((int)stream.size() < limit) {
        int cur = stream.back();
        if (cur == eos_id_ || cur == im_end_id_) break;
        ncnn::Mat hid = decode_single_causal(cur, pos);
        pos++;
        la_stat_line("ar.hidden", hid);
        ncnn::Mat lo = run_lm_head(hid);
        la_logits_line("ar.logits", lo);
        ncnn::Mat l2 = lo.clone();
        if (cfg.repetition_penalty != 1.0f) {
            float* pp = l2;
            for (int id : history) {
                if (id < l2.w) {
                    float v = pp[id];
                    pp[id] = v > 0.f ? v / cfg.repetition_penalty : v * cfg.repetition_penalty;
                }
            }
        }
        int next = sample_logits_row(l2, l2.w, cfg);
        stream.push_back(next);
        emit({next});
        history.insert(next);
        fprintf(stderr, "[locateanything] AR tok=%d (pos=%d)\n", next, pos);
        if (next == eos_id_ || next == im_end_id_) break;
    }
    fprintf(stderr, "[locateanything] AR-LOOP-END stream=%zu limit=%d last=%d\n",
            stream.size(), limit, stream.empty() ? -1 : stream.back());
}

// 视觉链：返回 [1, n_image_tokens, hidden] 图像特征（image_features 承载该 Mat）
ncnn::Mat ncnn_llm_locateanything::run_vision_features(const ncnn::Mat& bgr,
                                                       ncnn::Mat& image_features) {
    // 1) torch 参考 load_pil 先对 max(w,h)>MAX_DIM(=1024) 的图做 LANCZOS 预缩到
    //    1024（hybrid_runtime.load_pil，Image.LANCZOS）。缺这一步会导致 ncnn 直喂
    //    原图、grid 与 torch 不一致（§7.6/§7.10.1 大图预处理缺口）。预缩后再算 grid。
    int ow = bgr.w, oh = bgr.h;
    ncnn::Mat work = bgr;
    const int MAX_DIM = 1024;
    if (std::max(ow, oh) > MAX_DIM) {
        const double s = (double)MAX_DIM / std::max(ow, oh);
        int tw = (int)std::lround(ow * s), th = (int)std::lround(oh * s);
        tw = std::max(1, tw); th = std::max(1, th);
        work = ncnn_mat_resize_lanczos(bgr, tw, th);
        ow = work.w; oh = work.h;
    }
    // 处理器 rescale 复刻：target = ceil(尺寸/28)*28（merge*patch=28），保纵横比、无填充。
    //    超大图再按 in_token_limit 等比缩。对应 LocateAnythingImageProcessor.rescale。
    if ((ow / v_patch_) * (oh / v_patch_) > v_in_token_limit_) {
        const float scale_ = std::sqrt((float)v_in_token_limit_ /
                                       ((ow / v_patch_) * (oh / v_patch_)));
        ow = (int)(ow * scale_); oh = (int)(oh * scale_);
    }
    const int pad = merge_ * v_patch_;            // 2*14 = 28
    int target_w = ((ow + pad - 1) / pad) * pad;
    int target_h = ((oh + pad - 1) / pad) * pad;
    // 越界保护（原始实现 grid>=512 抛错；这里钳制到原生 64 网格，保证 pos_emb 可插值）
    if (target_w / v_patch_ > pos_emb_grid_ || target_h / v_patch_ > pos_emb_grid_) {
        target_w = std::min(target_w, pad * pos_emb_grid_);
        target_h = std::min(target_h, pad * pos_emb_grid_);
    }
    grid_w_ = target_w / v_patch_;
    grid_h_ = target_h / v_patch_;

    ncnn::Mat img = bgr_to_rgb_chw_normalized(work, target_w, target_h);
    ncnn::Mat pos;
    host_bicubic_pos_emb(grid_h_, grid_w_, pos);   // [gh*gw, dim]
    if (selftest_) self_check_vision(img, pos);

    // vision_embed: in0=image in1=pos_emb
    ncnn::Mat v0;
    { ncnn::Extractor ex = net_vision_embed_->create_extractor();
      ex.input("in0", img); ex.input("in1", pos); ex.extract("out0", v0); }
    v0 = la_canonical_2d(v0, vision_hidden_, "vision_embed.out0");
    // vision_encoder: in0=[1,L,dim] in1/2=moon cos/sin（动态 gh/gw），输出未 merge body
    ncnn::Mat vcos, vsin;
    moon_rope_cos(grid_h_, grid_w_, vcos, vsin);
    ncnn::Mat v1;
    { ncnn::Extractor ex = net_vision_encoder_->create_extractor();
      ex.input("in0", v0); ex.input("in1", vcos); ex.input("in2", vsin); ex.extract("out0", v1); }
    // 关键：encoder 输出 Mat 若跨子图直喂，会被误读 shape（pack/拓扑元数据）。
    // 重新装箱成规范 fp32 2D Mat 再喂 patch_merge/投影（实测 raw 给 h=1，rebox 给 h=256）。
    // 现在统一走 la_canonical_2d：先把可能的 pack4/pack8 解包（不同 CPU 架构的 ncnn
    // packing 支持不同，AVX512 等平台会返回 pack8），再摊平成 [dim, L]。
    ncnn::Mat v1b = la_canonical_2d(v1, vision_hidden_, "vision_encoder.out0");
    // 宿主 2x2 patch_merge（不再留在 encoder 图里）
    ncnn::Mat v1m;
    patch_merge_2x2(v1b, grid_h_, grid_w_, v1m);   // [M, merge_out=4*dim]
    // vision_projector
    ncnn::Mat vfeat;
    { ncnn::Extractor ex = net_vision_projector_->create_extractor();
      ex.input("in0", v1m); ex.extract("out0", vfeat); }
    vfeat = la_canonical_2d(vfeat, hidden_, "vision_projector.out0");
    n_image_tokens_ = (grid_h_ / merge_) * (grid_w_ / merge_);
    // LA_DUMP_VISION=<path>：把投影后的 [n_tokens, hidden] fp32 特征 dump 成 .f32，
    // 供与 torch extract_feature+mlp1 逐元素对比（验证动态导出数值正确性）。
    if (const char* dump = std::getenv("LA_DUMP_VISION")) {
        FILE* f = fopen(dump, "wb");
        if (f) {
            fwrite(vfeat.data, sizeof(float), (size_t)vfeat.w * vfeat.h, f);
            fclose(f);
            fprintf(stderr, "[locateanything] dumped vision features %dx%d grid=%dx%d -> %s\n",
                    vfeat.w, vfeat.h, grid_h_, grid_w_, dump);
        }
    }
    image_features = vfeat;
    return vfeat;
}

// ============================================================================
// 结构化坐标输出：把 <box>/</box>/<0>~<1000> 等 special token 渲染成可见文本
// ============================================================================
std::string ncnn_llm_locateanything::structured_token_text(int id) const {
    if (id == box_start_id_) return "<box>";
    if (id == box_end_id_) return "</box>";
    if (id >= coord_start_id_ && id <= coord_end_id_)
        return "<" + std::to_string(id - coord_start_id_) + ">";
    if (id == ref_start_id_) return "<ref>";
    if (id == ref_end_id_) return "</ref>";
    if (id == null_id_) return "<null>";
    if (id == im_end_id_) return "<|im_end|>";
    return std::string();
}

// 从生成段 token 里解析 <box>[c1 c2 c3 c4]</box> / <box>[c1 c2]</box>（点框）。
// 坐标 token 区间 [coord_start, coord_end] 对应 0~1000 的量化值。
void ncnn_llm_locateanything::collect_boxes(const std::vector<int>& ids, size_t prompt_len) {
    last_boxes_.clear();
    auto is_coord = [&](int t) { return t >= coord_start_id_ && t <= coord_end_id_; };
    size_t i = prompt_len;
    while (i < ids.size()) {
        if (ids[i] != box_start_id_) { i++; continue; }
        size_t j = i + 1;
        std::vector<int> cs;
        while (j < ids.size() && is_coord(ids[j]) && cs.size() < 4) cs.push_back(ids[j++]);
        if (j < ids.size() && ids[j] == box_end_id_ && (cs.size() == 4 || cs.size() == 2)) {
            LocateBox b;
            if (cs.size() == 4) {
                b.x1 = (cs[0] - coord_start_id_) / 1000.f;
                b.y1 = (cs[1] - coord_start_id_) / 1000.f;
                b.x2 = (cs[2] - coord_start_id_) / 1000.f;
                b.y2 = (cs[3] - coord_start_id_) / 1000.f;
            } else {
                const float cx = (cs[0] - coord_start_id_) / 1000.f;
                const float cy = (cs[1] - coord_start_id_) / 1000.f;
                const float r = 0.01f;      // 点框画成 ±1% 的小方块
                b.x1 = cx - r; b.y1 = cy - r; b.x2 = cx + r; b.y2 = cy + r;
                b.is_point = true;
            }
            if (b.x1 > b.x2) std::swap(b.x1, b.x2);
            if (b.y1 > b.y2) std::swap(b.y1, b.y2);
            b.x1 = std::min(1.f, std::max(0.f, b.x1));
            b.y1 = std::min(1.f, std::max(0.f, b.y1));
            b.x2 = std::min(1.f, std::max(0.f, b.x2));
            b.y2 = std::min(1.f, std::max(0.f, b.y2));
            last_boxes_.push_back(b);
            i = j + 1;
            continue;
        }
        i++;
    }
}

// ============================================================================
// 端到端 grounding
// ============================================================================
std::string ncnn_llm_locateanything::run(const ncnn::Mat& bgr_image, const std::string& question,
                                         const LocateGenerateConfig& cfg) {
    if (!ok_) return std::string();
    std::string out_text;

    // 1) 视觉特征
    ncnn::Mat image_features;
    run_vision_features(bgr_image, image_features);
    {
        const float* fp = image_features;
        float mn = 1e30f, mx = -1e30f, sm = 0.f;
        int n = image_features.w * image_features.h;
        for (int i = 0; i < n; i++) {
            float v = fp[i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sm += v;
        }
        fprintf(stderr, "[locateanything] DEBUG feat w=%d h=%d min=%.3f max=%.3f mean=%.4f\n",
                image_features.w, image_features.h, mn, mx, sm / std::max(1, n));
    }

    // 2) 拼 prompt：system/user + 256 <IMG_CONTEXT> + question + assistant
    //    与 torch 的 _tokenize 完全一致：user 文本=_PROMPT + question + "."（见
    //    batch_utils/hybrid_runtime.py:1795）。早期版本漏了英文 grounding 指令前缀
    //    与句号，导致解码器所见 prompt 与 torch 不同，输出分叉（多余框/截断身体）。
    std::string img_ctx = "<IMG_CONTEXT>";
    std::string img_ph = "<image 1><img>";
    std::string img_ph_end = "</img>";
    const std::string user_text =
        "Locate all the instances that matches the following description: " + question + ".";
    // system 模板与 torch py_apply_chat_template 一致：assistant 后带换行
    // （"You are a helpful assistant.\n"，缺该 \n 会改变 token 边界，令 decoder 所见
    // prompt 与 torch 不同，影响终止/多余框）。
    std::string prompt =
        "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n"
        "<|im_start|>user\n" + img_ph;
    for (int i = 0; i < n_image_tokens_; i++) prompt += img_ctx;
    prompt += img_ph_end + user_text + "<|im_end|>\n<|im_start|>assistant\n";

    std::vector<int> ids = bpe_->encode(prompt);
    const int S = (int)ids.size();

    // 3) 找到 image token 的位置（应为连续的 n_image_tokens_ 个）
    std::vector<int> img_pos;
    for (int i = 0; i < S; i++) {
        if (ids[i] == image_token_id_) img_pos.push_back(i);
    }
    printf("[locateanything] prompt seq=%d imgs=%d (expect %d)\n", S, (int)img_pos.size(),
           n_image_tokens_);
    if ((int)img_pos.size() != n_image_tokens_) {
        fprintf(stderr, "[locateanything] image token count mismatch\n");
        return std::string();
    }

    // 4) embed + 把视觉特征 scatter 进 <IMG_CONTEXT> 所在行
    ncnn::Mat embed = run_text_embed(ids);               // Mat(hidden, S)
    const float* fp = image_features;                     // Mat(hidden, n_image_tokens_)
    if (image_features.w != hidden_) {
        fprintf(stderr, "[locateanything] image feature width unexpected %d\n", image_features.w);
        return std::string();
    }
    for (int j = 0; j < n_image_tokens_; j++) {
        float* dst = embed.row(img_pos[j]);
        const float* src = fp + (size_t)j * hidden_;
        std::memcpy(dst, src, (size_t)hidden_ * sizeof(float));
    }

    // 5) 生成 cos/sin/掩码，prefill 整个序列
    ncnn::Mat cos, sin;
    text_rope_cos_sin(S, 0, cos, sin);
    ncnn::Mat mask = causal_mask(S);

    KVCache kv;
    kv.reserve((size_t)layers_ * 2);
    ncnn::Mat hidden = run_decoder(embed, cos, sin, mask, kv, true);  // Mat(hidden, S)
    la_stat_line("prefill.embed", embed);
    la_stat_line("prefill.hidden", hidden);
    if (selftest_) self_check_decoder_prefill(hidden, embed, cos, sin, mask);

    // last-token hidden -> lm_head（供 selftest 校验；MTP 窗口第 0 行输出即首个生成 token，
    // 这里不单独提交，交由 decode_loop_mtp 并行窗口产出）
    ncnn::Mat last(2048, 1);
    std::memcpy((float*)last.data, hidden.row(S - 1), (size_t)hidden_ * sizeof(float));
    if (selftest_) self_check_lm_head(last);

    // 跨平台指纹：prompt 末位的 top-5 logits。同模型同输入下各平台必须逐位一致，
    // 出现分歧即说明子图/链路在某平台发散（用于排查"某 CPU 上不出框"）。
    {
        ncnn::Mat lg = run_lm_head(last);
        const int V = lg.w;
        std::vector<std::pair<float, int>> top;
        const float* lp = lg;
        for (int i = 0; i < V; i++) {
            if ((int)top.size() < 5) {
                top.emplace_back(lp[i], i);
                std::push_heap(top.begin(), top.end(), std::greater<>());
            } else if (lp[i] > top.front().first) {
                std::pop_heap(top.begin(), top.end(), std::greater<>());
                top.back() = std::make_pair(lp[i], i);
                std::push_heap(top.begin(), top.end(), std::greater<>());
            }
        }
        std::sort(top.begin(), top.end(), std::greater<>());
        fprintf(stderr, "[locateanything] PREFILL top5 logits(V=%d):", V);
        for (auto& kv : top) fprintf(stderr, " %d:%.4f", kv.second, kv.first);
        fprintf(stderr, "\n");
        // LA_DUMP_TEXT=<prefix>：dump prefill 末位 hidden 与 logits（fp32），
        // 供跨平台 md5 / 逐元素对比，定位"某平台无输出"的发散点。
        if (const char* pref = std::getenv("LA_DUMP_TEXT")) {
            auto dump = [](const std::string& p, const float* d, size_t n) {
                FILE* f = fopen(p.c_str(), "wb");
                if (!f) { fprintf(stderr, "[locateanything] dump failed: %s\n", p.c_str()); return; }
                fwrite(d, sizeof(float), n, f);
                fclose(f);
                fprintf(stderr, "[locateanything] dumped %s (%zu floats)\n", p.c_str(), n);
            };
            dump(std::string(pref) + "_hidden.f32", (const float*)last.data, (size_t)hidden_);
            dump(std::string(pref) + "_logits.f32", lp, (size_t)V);
        }
        // 视觉特征指纹（与 DEBUG feat 行的 min/max/mean 一起用于跨平台比对）
        double s = 0.0;
        for (int i = 0; i < image_features.w * image_features.h; i++) s += fp[i];
        fprintf(stderr, "[locateanything] FEATSUM %.6f\n", s);
    }

    // 6) MTP 并行窗口解码（+ AR 回退），对齐 torch generation_mode='hybrid'。
    //    ids 即 prompt token 流，函数内会追加生成 token；out_text 由回调逐 token 累积。
    std::unordered_set<int> history;
    if (cfg.use_mtp)
        decode_loop_mtp(out_text, ids, kv, cfg, history);
    else
        decode_loop_ar(out_text, ids, kv, cfg, history);

    // 7) 从生成 token 解析检测框（归一化 [0,1]），供调用方打印/画框。
    collect_boxes(ids, (size_t)S);
    return out_text;
}

// ============================================================================
// LA_SELFTEST 逐子图 CPU vs Vulkan 对比（诊断）
// ============================================================================
float ncnn_llm_locateanything::max_abs_diff(const ncnn::Mat& a, const ncnn::Mat& b) {
    if (a.w != b.w || a.h != b.h || a.c != b.c) {
        fprintf(stderr, "[selfcheck] SHAPE MISMATCH a(w=%d h=%d c=%d) b(w=%d h=%d c=%d)\n",
                a.w, a.h, a.c, b.w, b.h, b.c);
        return -1.f;
    }
    // 按 fp32 行序比较（两种实现都应产出 fp32 结果，dump 回 CPU）
    float m = 0.f;
    for (int ic = 0; ic < a.c; ic++) {
        const float* pa = a.channel(ic);
        const float* pb = b.channel(ic);
        for (int i = 0; i < a.h * a.w; i++) {
            float d = std::fabs(pa[i] - pb[i]);
            if (d > m) m = d;
        }
    }
    return m;
}

void ncnn_llm_locateanything::self_check_vision(const ncnn::Mat& img, const ncnn::Mat& pos) {
    fprintf(stderr, "[selfcheck]--- vision chain (identical inputs, VK vs CPU) ---\n");

    // vision_embed
    ncnn::Mat v0;
    { ncnn::Extractor ex = net_vision_embed_->create_extractor();
      ex.input("in0", img); ex.input("in1", pos); ex.extract("out0", v0); }
    ncnn::Mat v0cpu;
    { ncnn::Extractor ex = st_vision_embed_cpu_->create_extractor();
      ex.input("in0", img); ex.input("in1", pos); ex.extract("out0", v0cpu); }
    fprintf(stderr, "[selfcheck] vision_embed    maxdiff=%.4g (w=%d h=%d)\n",
            max_abs_diff(v0, v0cpu), v0.w, v0.h);

    // vision_encoder：统一喂 CPU 侧 v0（rebox）给两端
    ncnn::Mat vcos, vsin;
    moon_rope_cos(grid_h_, grid_w_, vcos, vsin);
    ncnn::Mat v0b(v0cpu.w, v0cpu.h);
    memcpy(v0b.data, v0cpu.data, (size_t)v0cpu.w * v0cpu.h * sizeof(float));
    ncnn::Mat v1, v1cpu;
    { ncnn::Extractor ex = net_vision_encoder_->create_extractor();
      ex.input("in0", v0b); ex.input("in1", vcos); ex.input("in2", vsin); ex.extract("out0", v1); }
    { ncnn::Extractor ex = st_vision_encoder_cpu_->create_extractor();
      ex.input("in0", v0b); ex.input("in1", vcos); ex.input("in2", vsin); ex.extract("out0", v1cpu); }
    fprintf(stderr, "[selfcheck] vision_encoder  maxdiff=%.4g (w=%d h=%d)\n",
            max_abs_diff(v1, v1cpu), v1.w, v1.h);

    // vision_projector：统一喂 CPU 侧 v1（rebox）给两端
    ncnn::Mat v1b(v1cpu.w, v1cpu.h);
    memcpy(v1b.data, v1cpu.data, (size_t)v1cpu.w * v1cpu.h * sizeof(float));
    ncnn::Mat vf, vfcpu;
    { ncnn::Extractor ex = net_vision_projector_->create_extractor();
      ex.input("in0", v1b); ex.extract("out0", vf); }
    { ncnn::Extractor ex = st_vision_projector_cpu_->create_extractor();
      ex.input("in0", v1b); ex.extract("out0", vfcpu); }
    fprintf(stderr, "[selfcheck] vision_projector maxdiff=%.4g (w=%d h=%d)\n",
            max_abs_diff(vf, vfcpu), vf.w, vf.h);
}

void ncnn_llm_locateanything::self_check_decoder_prefill(const ncnn::Mat& vk_hidden, const ncnn::Mat& emb,
                                                         const ncnn::Mat& cos, const ncnn::Mat& sin,
                                                         const ncnn::Mat& mask) {
    // 同一输入 + 空 KV，CPU decoder 重跑一遍，与主后端(Vulkan) prefill hidden 对比
    ncnn::Mat out;
    ncnn::Extractor ex = net_text_decoder_cpu_->create_extractor();
    ex.input("in0", emb); ex.input("in1", cos); ex.input("in2", sin); ex.input("in3", mask);
    for (int i = 0; i < layers_; i++) {
        ncnn::Mat e(head_dim_, 0, kv_heads_);
        ex.input(("cache_k" + std::to_string(i)).c_str(), e);
        ex.input(("cache_v" + std::to_string(i)).c_str(), e.clone());
    }
    ex.extract("out0", out);
    fprintf(stderr, "[selfcheck] decoder_prefill  maxdiff=%.4g (w=%d h=%d)\n",
            max_abs_diff(vk_hidden, out), vk_hidden.w, vk_hidden.h);
}

void ncnn_llm_locateanything::self_check_lm_head(const ncnn::Mat& hidden) {
    // 同一 hidden 喂 Vulkan 与 CPU lm_head，对比 logits
    ncnn::Mat vk, cpu;
    { ncnn::Extractor ex = net_lm_head_->create_extractor(); ex.input("in0", hidden); ex.extract("out0", vk); }
    { ncnn::Extractor ex = net_lm_head_cpu_->create_extractor(); ex.input("in0", hidden); ex.extract("out0", cpu); }
    fprintf(stderr, "[selfcheck] lm_head        maxdiff=%.4g (w=%d)\n",
            max_abs_diff(vk, cpu), vk.w);
}