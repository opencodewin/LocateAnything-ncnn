#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <mat.h>
#include <net.h>
#include <nlohmann/json.hpp>

#include "ncnn_llm_base.h"
#include "utils/draw_utils.h"
#include "utils/tokenizer/bpe_tokenizer.h"

// LocateAnything-3B 的 ncnn_llm 补充实现（grounding VLM）。
//
// 直接驱动 universal 项目导出的 6 个子图（vision_embed / vision_encoder /
// vision_projector / text_embed / text_decoder / lm_head），text_decoder 走
// 它自己的 KV 增量契约：in0=emb, in1=cos, in2=sin, in3=mask + cache_k<i>/
// cache_v<i> / out_cache_k<i> / out_cache_v<i>（与 ncnn_text_runtime 的
// in1=mask,in2=cos,in3=sin 顺序不同，故不复用而由本类自驱）。
//
// 输入序（都来自 universal export 的 pnnx 命名；以下为基础档 448 形状，实际按
// 选档后的变体（896/1792）替换）：
//   vision_embed         in0=image[1,3,448,448]  in1=pos_emb[1,1024,1152] -> out0[1,1024,1152]
//   vision_encoder       in0=[1,1024,1152] in1/2=cos/sin[1024,36]        -> out0[1,256,4608]
//   vision_projector     in0=[1,256,4608]                                 -> out0[1,256,2048]
//   text_embed           in0=ids[1,S] (int)                               -> out0[1,S,2048]
//   text_decoder         in0=emb in1=cos in2=sin in3=mask (+KV)           -> out0[1,S,2048]
//   lm_head              in0=hidden[1,1,2048]                             -> out0 logits[1,1,vocab]
//
// Grounding 流程：图像直接双三次拉伸到选档画布（448/896/1792，与 torch 处理器
// rescale 一致，无 letterbox）-> vision 链得到 256/1024/4096 个图像 token ->
// 把 prompt 里对应个数的 <IMG_CONTEXT>(id=image_token) 位置 scatter 进 text_embed
// 输出 -> decoder prefill+decode(KV) -> lm_head 采样（AR / "slow" 模式）。
// 输出为带结构化坐标 token 的文本，例如 "<box><x1><y1><x2><y2></box>"。

struct LocateGenerateConfig {
    int max_new_tokens = 2048;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 50;
    float repetition_penalty = 1.1f;
    bool do_sample = true;
    bool use_mtp = true;   // true: MTP 并行窗口解码（torch hybrid）；false: 纯逐 token AR
    std::function<void(const std::string&)> callback = nullptr;
};

class ncnn_llm_locateanything : public ncnn_llm_base {
public:
    ncnn_llm_locateanything(const std::string& model_path, bool use_vulkan, int num_threads,
                            int vulkan_device = 0, bool use_fp16 = false);

    bool ok() const { return ok_; }
    const std::string& model_type() const { return model_type_; }

    // 端到端 grounding：rgb_image 为 load_image_to_ncnn_mat 的结果（BGR u8 interleaved），
    // question 为自然语言 query/指令。返回模型生成的原始文本（含 <box><x1><y1><x2><y2></box>
    // 结构化坐标 token，坐标量化到 0~1000）。
    std::string run(const ncnn::Mat& bgr_image, const std::string& question,
                    const LocateGenerateConfig& cfg);

    // 上一次 run 解析出的检测框（归一化 [0,1]，相对原图宽高）。
    // 直接由生成的 token id 解析，不依赖文本渲染，最可靠。
    const std::vector<LocateBox>& last_boxes() const { return last_boxes_; }

private:
    void text_rope_cos_sin(int seq, int pos_start, ncnn::Mat& cos, ncnn::Mat& sin);
    // RoPE cos/sin 基于逐行绝对位置（MTP 窗口里 rep 与 mask 位置非连续、且 rep 会与
    // 前一块末位重复，故需任意位置数组）。
    void text_rope_cos_sin_at(const std::vector<int>& pos, ncnn::Mat& cos, ncnn::Mat& sin);
    void moon_rope_cos(int grid_h, int grid_w, ncnn::Mat& cos, ncnn::Mat& sin);
    ncnn::Mat causal_mask(int seq);
    ncnn::Mat run_vision_features(const ncnn::Mat& bgr, ncnn::Mat& image_features);
    ncnn::Mat run_text_embed(const std::vector<int>& ids);
    ncnn::Mat run_decoder(const ncnn::Mat& emb, const ncnn::Mat& cos, const ncnn::Mat& sin,
                          const ncnn::Mat& mask, KVCache& kv, bool is_prefill);
    ncnn::Mat run_lm_head(const ncnn::Mat& hidden);
    bool load_pos_emb(const std::string& path);
    // 结构 token（<box>/</box>/<0>~<1000>/<ref>/...）的文本渲染；非结构 token 返回空串。
    // 这些 token 在 tokenizer 里属于 special，decode 会跳过，导致生成文本里看不到坐标，
    // 故在此兜底渲染，使输出文本可被 parse_locate_boxes_text 直接解析。
    std::string structured_token_text(int id) const;
    // 从生成段（prompt 之后）的 token id 里解析 <box>..</box>，结果写入 last_boxes_。
    void collect_boxes(const std::vector<int>& ids, size_t prompt_len);
    // 宿主端动态视觉（v3）：网络按实际网格任意喂，pos_emb 用原生 64x64 权重做
    // torch 语义的 bicubic（align_corners=False）插值，patch_merger 移到宿主。
    void host_bicubic_pos_emb(int grid_h, int grid_w, ncnn::Mat& out);  // out=[gh*gw, dim]
    // 当前网格（run_vision_features 内更新）下的图像 token 数
    int image_token_count() const { return n_image_tokens_; }
    // 构建 [past+K, K] 的 MTP 块掩码：前 (K-block) 行为纯因果（重喂的真实 token），
    // 后 block 行（窗口）窗口内双向可见、屏蔽窗口前一键（索引 past+K-block-1）。
    ncnn::Mat mtp_mask(int K, int past);
    // 移植 torch generate_utils.sample_tokens + decode_bbox_avg + handle_pattern（hybrid 模式）。
    // win[i]=第 i 个并行位置的单行 logits(w=vocab)。返回提交的 token 序列，type 在
    //   {"im_end","empty_box","coord_box","point_box","error_box","ref_object"} 中。
    std::vector<int> mtp_window_decode(const std::vector<ncnn::Mat>& win, std::string& type,
                                       const LocateGenerateConfig& cfg);
    // 单行采样：greedy(do_sample=false) -> argmax；否则 temperature/top_k/top_p + 类别采样。
    int sample_logits_row(const float* lp, int V, const LocateGenerateConfig& cfg);
    // MTP 并行窗口 + AR 回退 的生成循环。stream=prompt ids，会被追加生成 token；
    // KV 以"真实 token 已缓存到 cache_len"的方式维护（每步重喂刚提交的真实 token，
    // 使 mask 占位不污染后续窗口的注意力键，等价于 torch 每步截断 KV 后重喂）。
    void decode_loop_mtp(std::string& out_text, std::vector<int>& stream, KVCache& kv,
                         const LocateGenerateConfig& cfg, std::unordered_set<int>& history);
    // 纯逐 token AR 生成循环（不构造 MTP 窗口）：每步因果前向 1 个新位置 -> lm_head 采样，
    // 遇见 eos/im_end 终止。供与 MTP 解码的输出对比（cfg.use_mtp=false 时启用）。
    void decode_loop_ar(std::string& out_text, std::vector<int>& stream, KVCache& kv,
                        const LocateGenerateConfig& cfg, std::unordered_set<int>& history);
    // 诊断：LA_SELFTEST=1 时，每个子图同时用 Vulkan + CPU 副本喂相同输入，打印 maxdiff。
    // 用于逐步定位 Vulkan 分歧点（视觉链/embed/prefill/lm_head）。
    void self_check_vision(const ncnn::Mat& img, const ncnn::Mat& pos);
    void self_check_decoder_prefill(const ncnn::Mat& vk_hidden, const ncnn::Mat& emb,
                                    const ncnn::Mat& cos, const ncnn::Mat& sin,
                                    const ncnn::Mat& mask);
    void self_check_lm_head(const ncnn::Mat& hidden);
    static float max_abs_diff(const ncnn::Mat& a, const ncnn::Mat& b);

private:
    // 6 个子图网络
    std::shared_ptr<ncnn::Net> net_vision_embed_;
    std::shared_ptr<ncnn::Net> net_vision_encoder_;
    std::shared_ptr<ncnn::Net> net_vision_projector_;
    std::shared_ptr<ncnn::Net> net_text_embed_;
    std::shared_ptr<ncnn::Net> net_text_decoder_;
    std::shared_ptr<ncnn::Net> net_lm_head_;
    // decode 阶段的 CPU 副本：ncnn Vulkan 对动态 shape 的 decode(seq=1 + KV 递增)
    // 会落到 CPU 且每次搬迁，实测比纯 CPU 更慢；故 decode 固定走 CPU 副本。
    std::shared_ptr<ncnn::Net> net_text_decoder_cpu_;
    std::shared_ptr<ncnn::Net> net_lm_head_cpu_;

    std::shared_ptr<BpeTokenizer> bpe_;
    ncnn::Mat pos_emb_;  // 原生 64x64 pos_emb = [1,4096,1152] fp32（宿主插值源）

    std::string model_type_;
    std::string model_path_;  // 模型目录（变体装载用）
    bool ok_ = false;
    bool vulkan_ = false;  // 主 6 子图是否走 Vulkan；decode 恒走 CPU
    bool selftest_ = false;  // LA_SELFTEST=1 开启逐子图 CPU vs Vulkan 对比
    // 诊断用：CPU 副本（比对主副本是否同样可用）
    std::shared_ptr<ncnn::Net> st_vision_embed_cpu_, st_vision_encoder_cpu_;
    std::shared_ptr<ncnn::Net> st_vision_projector_cpu_, st_text_embed_cpu_;

    // ---- text / LLM 配置 ----
    int hidden_ = 2048;
    int head_dim_ = 128;
    int layers_ = 36;
    int kv_heads_ = 2;      // GQA 的 kv 头数 = 导出图暴露的 cache 通道数（动态，随 config）
    int heads_ = 16;        // 注意力头数（k/v 在图内 Tile 到 heads 再喂 SDPA，不在 cache 接口暴露）
    int vocab_ = 152681;
    double rope_theta_ = 1e6;
    float mask_fill_ = -10000.0f;

    // ---- tokenizer 特殊 id ----
    int eos_id_ = -1;
    int im_end_id_ = -1;
    int image_token_id_ = 151665;
    // MTP box 解码所需（torch generate_utils 默认；mask_tok=text_mask_token_id）
    int mask_tok_id_ = 151676;   // <extra_token_999> 掩码占位
    int box_start_id_ = 151668, box_end_id_ = 151669;   // <box> </box>
    int coord_start_id_ = 151677, coord_end_id_ = 152677;  // <0> ~ <1000> 坐标 token 区间
    int ref_start_id_ = 151672, ref_end_id_ = 151673;   // <ref> </ref>
    int null_id_ = 152678, none_id_ = 4064;             // null / none
    int block_size_ = 6;   // N_FUTURE=config.block_size（MTP 并行窗口宽）

    // ---- vision 配置（v3 动态：单套动态图 + 宿主插值，无静态变体）----
    int grid_h_ = 64, grid_w_ = 64, merge_ = 2;  // 当前图网格（随输入图像变化）
    int v_patch_ = 14;              // patch_size
    int v_in_token_limit_ = 4096;   // 处理器 in_token_limit（超大图先等比缩）
    int pos_emb_grid_ = 64;         // 原生 pos_emb 网格（Learnable2DInterpPosEmb init）
    int vision_hidden_ = 1152;      // 单 token 通道（encoder 未 merge 输出 1152；merge 后 4608）
    int vision_head_dim_ = 72;
    double vision_rope_theta_ = 10000.0;
    int n_image_tokens_ = 256;      // run_vision_features 内按 (gh/merge)*(gw/merge) 更新

    std::vector<LocateBox> last_boxes_;   // 最近一次 run 解析出的检测框（归一化）
};