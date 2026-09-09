#include "draw_utils.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "image_utils.h"

// ============================================================================
// 标注绘制 + PNG 写出。
//
// 只依赖 libc：PNG 用 zlib 的 stored（未压缩）deflate 块直接封装，省掉 zlib 依赖；
// 文字用内置 5x7 点阵（数字），够标序号/坐标用。
// ============================================================================

namespace {

// 5x7 数字点阵：每列 1 字节，bit i = 第 i 行（bit0 = 顶部）
const unsigned char kFont5x7[10][5] = {
    {0x3E, 0x51, 0x49, 0x45, 0x3E},  // 0
    {0x00, 0x42, 0x7F, 0x40, 0x00},  // 1
    {0x42, 0x61, 0x51, 0x49, 0x46},  // 2
    {0x21, 0x41, 0x45, 0x4B, 0x31},  // 3
    {0x18, 0x14, 0x12, 0x7F, 0x10},  // 4
    {0x27, 0x45, 0x45, 0x45, 0x39},  // 5
    {0x3C, 0x4A, 0x49, 0x49, 0x30},  // 6
    {0x01, 0x71, 0x09, 0x05, 0x03},  // 7
    {0x36, 0x49, 0x49, 0x49, 0x36},  // 8
    {0x06, 0x49, 0x49, 0x29, 0x1E},  // 9
};

// 框颜色（BGR 顺序），依次轮换，便于区分多目标
const unsigned char kPalette[][3] = {
    {0, 0, 255},      // 红
    {0, 255, 0},      // 绿
    {0, 255, 255},    // 黄
    {255, 255, 0},    // 青
    {255, 0, 255},    // 品红
    {0, 165, 255},    // 橙
    {255, 0, 128},    // 紫
    {128, 255, 128},  // 浅绿
};
const int kPaletteSize = (int)(sizeof(kPalette) / sizeof(kPalette[0]));

// ---- 基础像素操作（img 为 BGR u8 interleaved，行距 = w*3）----

void blend_px(unsigned char* img, int w, int h, int x, int y, const unsigned char c[3],
              unsigned char a = 255) {
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    unsigned char* p = img + ((size_t)y * w + x) * 3;
    if (a >= 255) {
        p[0] = c[0]; p[1] = c[1]; p[2] = c[2];
        return;
    }
    p[0] = (unsigned char)((p[0] * (255 - a) + c[0] * a) / 255);
    p[1] = (unsigned char)((p[1] * (255 - a) + c[1] * a) / 255);
    p[2] = (unsigned char)((p[2] * (255 - a) + c[2] * a) / 255);
}

void fill_rect(unsigned char* img, int w, int h, int x0, int y0, int x1, int y1,
               const unsigned char c[3], unsigned char a = 255) {
    if (x1 < x0) std::swap(x0, x1);
    if (y1 < y0) std::swap(y0, y1);
    x0 = std::max(x0, 0); y0 = std::max(y0, 0);
    x1 = std::min(x1, w - 1); y1 = std::min(y1, h - 1);
    for (int y = y0; y <= y1; y++)
        for (int x = x0; x <= x1; x++) blend_px(img, w, h, x, y, c, a);
}

// 描边矩形：th 为线宽（像素）
void stroke_rect(unsigned char* img, int w, int h, int x0, int y0, int x1, int y1,
                 const unsigned char c[3], int th) {
    if (x1 < x0) std::swap(x0, x1);
    if (y1 < y0) std::swap(y0, y1);
    fill_rect(img, w, h, x0, y0, x1, y0 + th - 1, c);
    fill_rect(img, w, h, x0, y1 - th + 1, x1, y1, c);
    fill_rect(img, w, h, x0, y0, x0 + th - 1, y1, c);
    fill_rect(img, w, h, x1 - th + 1, y0, x1, y1, c);
}

// 点框：实心圆点 + 十字准星
void draw_point(unsigned char* img, int w, int h, int cx, int cy, int r,
                const unsigned char c[3], int th) {
    for (int y = cy - r; y <= cy + r; y++)
        for (int x = cx - r; x <= cx + r; x++) {
            int dx = x - cx, dy = y - cy;
            if (dx * dx + dy * dy <= r * r) blend_px(img, w, h, x, y, c);
        }
    fill_rect(img, w, h, cx - 3 * r, cy - th / 2, cx + 3 * r, cy - th / 2 + th - 1, c);
    fill_rect(img, w, h, cx - th / 2, cy - 3 * r, cx - th / 2 + th - 1, cy + 3 * r, c);
}

// 5x7 点阵文本（只支持数字，其它字符按空格处理），scale 为放大倍数
void draw_text(unsigned char* img, int w, int h, const std::string& s, int x, int y,
               const unsigned char c[3], int scale) {
    int cx = x;
    for (size_t i = 0; i < s.size(); i++) {
        unsigned char ch = (unsigned char)s[i];
        if (ch >= '0' && ch <= '9') {
            const unsigned char* g = kFont5x7[ch - '0'];
            for (int gx = 0; gx < 5; gx++) {
                for (int gy = 0; gy < 7; gy++) {
                    if (g[gx] & (1u << gy))
                        fill_rect(img, w, h, cx + gx * scale, y + gy * scale,
                                  cx + gx * scale + scale - 1, y + gy * scale + scale - 1, c);
                }
            }
        }
        cx += 6 * scale;
    }
}

// ---- 极简 PNG 写出（zlib stored 块）----

const unsigned long* crc_table() {
    static unsigned long tbl[256];
    static bool inited = false;
    if (!inited) {
        for (unsigned long n = 0; n < 256; n++) {
            unsigned long c = n;
            for (int k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            tbl[n] = c;
        }
        inited = true;
    }
    return tbl;
}

unsigned long adler32_buf(const unsigned char* data, size_t len) {
    unsigned long a = 1, b = 0;
    for (size_t i = 0; i < len; i++) {
        a = (a + data[i]) % 65521u;
        b = (b + a) % 65521u;
    }
    return (b << 16) | a;
}

void put_be32(FILE* f, unsigned long v) {
    fputc((int)((v >> 24) & 0xFF), f);
    fputc((int)((v >> 16) & 0xFF), f);
    fputc((int)((v >> 8) & 0xFF), f);
    fputc((int)(v & 0xFF), f);
}

void png_chunk(FILE* f, const char* type, const unsigned char* data, size_t len) {
    put_be32(f, len);
    fwrite(type, 1, 4, f);
    if (len) fwrite(data, 1, len, f);
    // CRC 覆盖 type + data
    unsigned long c = 0xFFFFFFFFu;
    const unsigned long* tbl = crc_table();
    for (int i = 0; i < 4; i++) c = tbl[(c ^ (unsigned char)type[i]) & 0xFF] ^ (c >> 8);
    for (size_t i = 0; i < len; i++) c = tbl[(c ^ data[i]) & 0xFF] ^ (c >> 8);
    put_be32(f, c ^ 0xFFFFFFFFu);
}

// rgb: 交错的 RGB u8（无 alpha），w/h 为尺寸
bool write_png(const std::string& path, const unsigned char* rgb, int w, int h) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;

    const unsigned char sig[8] = {137, 'P', 'N', 'G', '\r', '\n', 26, '\n'};
    fwrite(sig, 1, 8, f);

    unsigned char ihdr[13];
    ihdr[0] = (unsigned char)((w >> 24) & 0xFF);
    ihdr[1] = (unsigned char)((w >> 16) & 0xFF);
    ihdr[2] = (unsigned char)((w >> 8) & 0xFF);
    ihdr[3] = (unsigned char)(w & 0xFF);
    ihdr[4] = (unsigned char)((h >> 24) & 0xFF);
    ihdr[5] = (unsigned char)((h >> 16) & 0xFF);
    ihdr[6] = (unsigned char)((h >> 8) & 0xFF);
    ihdr[7] = (unsigned char)(h & 0xFF);
    ihdr[8] = 8;    // bit depth
    ihdr[9] = 2;    // color type: truecolor RGB
    ihdr[10] = 0;   // compression
    ihdr[11] = 0;   // filter
    ihdr[12] = 0;   // interlace
    png_chunk(f, "IHDR", ihdr, 13);

    // 原始扫描线：每行 1 字节 filter(0) + RGB
    const size_t stride = (size_t)w * 3 + 1;
    std::vector<unsigned char> raw((size_t)h * stride);
    for (int y = 0; y < h; y++) {
        unsigned char* dst = raw.data() + (size_t)y * stride;
        dst[0] = 0;
        memcpy(dst + 1, rgb + (size_t)y * w * 3, (size_t)w * 3);
    }

    // zlib 头（CMF=0x78, FLG=0x01）+ stored deflate 块 + adler32
    std::vector<unsigned char> z;
    z.push_back(0x78);
    z.push_back(0x01);
    size_t off = 0;
    do {
        size_t n = std::min<size_t>(65535, raw.size() - off);
        unsigned char last = (off + n >= raw.size()) ? 1 : 0;   // BFINAL=last, BTYPE=00(stored)
        z.push_back(last);
        z.push_back((unsigned char)(n & 0xFF));
        z.push_back((unsigned char)((n >> 8) & 0xFF));
        z.push_back((unsigned char)((~n) & 0xFF));
        z.push_back((unsigned char)(((~n) >> 8) & 0xFF));
        z.insert(z.end(), raw.begin() + off, raw.begin() + off + n);
        off += n;
    } while (off < raw.size());
    unsigned long ad = adler32_buf(raw.data(), raw.size());
    z.push_back((unsigned char)((ad >> 24) & 0xFF));
    z.push_back((unsigned char)((ad >> 16) & 0xFF));
    z.push_back((unsigned char)((ad >> 8) & 0xFF));
    z.push_back((unsigned char)(ad & 0xFF));

    png_chunk(f, "IDAT", z.data(), z.size());
    png_chunk(f, "IEND", nullptr, 0);

    fclose(f);
    return true;
}

}  // namespace

// ============================================================================
std::vector<LocateBox> parse_locate_boxes_text(const std::string& text) {
    std::vector<LocateBox> out;
    const std::string open = "<box>";
    size_t pos = 0;
    while (pos < text.size()) {
        size_t b = std::string::npos;
        // 大小写不敏感查找 "<box>"
        for (size_t i = pos; i + open.size() <= text.size(); i++) {
            bool hit = true;
            for (size_t k = 0; k < open.size(); k++) {
                if (std::tolower((unsigned char)text[i + k]) != open[k]) { hit = false; break; }
            }
            if (hit) { b = i; break; }
        }
        if (b == std::string::npos) break;

        // 收集 <box> 之后到 </box>（或下一个 <box>/结尾）之间的数字
        size_t i = b + open.size();
        std::vector<float> nums;
        bool closed = false;
        while (i < text.size()) {
            if (text[i] == '<') {
                bool is_end = true;
                for (size_t k = 0; k < 6; k++) {
                    if (i + k >= text.size() ||
                        std::tolower((unsigned char)text[i + k]) != "</box>"[k]) { is_end = false; break; }
                }
                if (is_end) { closed = true; i += 6; break; }
                if (i + open.size() <= text.size()) {
                    bool is_next = true;
                    for (size_t k = 0; k < open.size(); k++) {
                        if (std::tolower((unsigned char)text[i + k]) != open[k]) { is_next = false; break; }
                    }
                    if (is_next) break;
                }
            }
            if (std::isdigit((unsigned char)text[i]) ||
                (text[i] == '.' && i + 1 < text.size() && std::isdigit((unsigned char)text[i + 1]))) {
                char* endp = nullptr;
                double v = strtod(text.c_str() + i, &endp);
                if (endp && endp > text.c_str() + i) {
                    nums.push_back((float)v);
                    i = (size_t)(endp - text.c_str());
                    continue;
                }
            }
            i++;
        }
        if (nums.size() == 4 || (nums.size() == 2 && closed)) {
            bool quantized = false;
            for (float v : nums) if (v > 1.0f) quantized = true;
            LocateBox box;
            if (nums.size() == 4) {
                box.x1 = nums[0]; box.y1 = nums[1]; box.x2 = nums[2]; box.y2 = nums[3];
            } else {
                const float r = 0.01f;
                box.x1 = nums[0] - r; box.y1 = nums[1] - r;
                box.x2 = nums[0] + r; box.y2 = nums[1] + r;
                box.is_point = true;
            }
            if (quantized) {
                box.x1 /= 1000.f; box.y1 /= 1000.f; box.x2 /= 1000.f; box.y2 /= 1000.f;
                if (box.is_point) {
                    // 量化点框：还原成 1% 半径的小方块（先除后扩）
                    float cx = (nums[0]) / 1000.f, cy = (nums[1]) / 1000.f;
                    box.x1 = cx - 0.01f; box.y1 = cy - 0.01f;
                    box.x2 = cx + 0.01f; box.y2 = cy + 0.01f;
                }
            }
            if (box.x1 > box.x2) std::swap(box.x1, box.x2);
            if (box.y1 > box.y2) std::swap(box.y1, box.y2);
            box.x1 = std::min(1.f, std::max(0.f, box.x1));
            box.y1 = std::min(1.f, std::max(0.f, box.y1));
            box.x2 = std::min(1.f, std::max(0.f, box.x2));
            box.y2 = std::min(1.f, std::max(0.f, box.y2));
            out.push_back(box);
        }
        pos = i;
    }
    return out;
}

std::string format_locate_box(const LocateBox& box, int img_w, int img_h, int index) {
    char buf[256];
    int n = snprintf(buf, sizeof(buf), "#%d norm=(%.4f,%.4f)-(%.4f,%.4f)", index,
                     box.x1, box.y1, box.x2, box.y2);
    std::string s(buf, n > 0 ? n : 0);
    if (img_w > 0 && img_h > 0) {
        int px1 = (int)lroundf(box.x1 * img_w), px2 = (int)lroundf(box.x2 * img_w);
        int py1 = (int)lroundf(box.y1 * img_h), py2 = (int)lroundf(box.y2 * img_h);
        px1 = std::min(img_w - 1, std::max(0, px1));
        px2 = std::min(img_w - 1, std::max(0, px2));
        py1 = std::min(img_h - 1, std::max(0, py1));
        py2 = std::min(img_h - 1, std::max(0, py2));
        n = snprintf(buf, sizeof(buf), " px=(%d,%d)-(%d,%d) size=%dx%d%s", px1, py1, px2, py2,
                     px2 - px1 + 1, py2 - py1 + 1, box.is_point ? " [point]" : "");
        s.append(buf, n > 0 ? n : 0);
    }
    if (!box.label.empty()) s += " label=" + box.label;
    return s;
}

bool draw_locate_boxes(const ncnn::Mat& bgr, const std::vector<LocateBox>& boxes,
                       const std::string& out_png, int thickness) {
    if (ncnn_mat_empty(bgr) || out_png.empty()) {
        fprintf(stderr, "[draw] invalid image or output path\n");
        return false;
    }
    const int w = bgr.w, h = bgr.h;

    // 原图可能是 c=3/elemsize=1 的"BGR u8 interleaved"约定（见 image_utils）。
    ncnn::Mat canvas = bgr.clone();
    unsigned char* img = (unsigned char*)canvas.data;

    if (thickness <= 0) thickness = std::max(2, (int)lroundf(std::min(w, h) / 300.f));
    const int scale = std::max(1, (int)lroundf(thickness * 0.8f));
    const unsigned char white[3] = {255, 255, 255};

    for (size_t i = 0; i < boxes.size(); i++) {
        const LocateBox& b = boxes[i];
        const unsigned char* col = kPalette[i % kPaletteSize];
        int x0 = (int)lroundf(b.x1 * w), x1 = (int)lroundf(b.x2 * w);
        int y0 = (int)lroundf(b.y1 * h), y1 = (int)lroundf(b.y2 * h);

        if (b.is_point) {
            int cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
            draw_point(img, w, h, cx, cy, std::max(4, thickness * 2), col, thickness);
        } else {
            stroke_rect(img, w, h, x0, y0, x1, y1, col, thickness);
        }

        // 左上角序号徽标（黑底 + 白字）
        const int bw = (int)(((int)std::to_string(i).size()) * 6 * scale) + 2 * scale;
        const int bh = 7 * scale + 2 * scale;
        int bx = std::min(std::max(x0, 0), std::max(0, w - bw));
        int by = std::min(std::max(y0 - bh, 0), std::max(0, h - bh));
        const unsigned char black[3] = {0, 0, 0};
        fill_rect(img, w, h, bx, by, bx + bw, by + bh, black, 220);
        draw_text(img, w, h, std::to_string(i), bx + scale, by + scale, white, scale);
    }

    // BGR -> RGB
    std::vector<unsigned char> rgb((size_t)w * h * 3);
    for (int i = 0; i < w * h; i++) {
        rgb[(size_t)i * 3 + 0] = img[(size_t)i * 3 + 2];
        rgb[(size_t)i * 3 + 1] = img[(size_t)i * 3 + 1];
        rgb[(size_t)i * 3 + 2] = img[(size_t)i * 3 + 0];
    }
    if (!write_png(out_png, rgb.data(), w, h)) {
        fprintf(stderr, "[draw] failed to write %s\n", out_png.c_str());
        return false;
    }
    return true;
}

std::string default_annotated_path(const std::string& image_path) {
    size_t slash = image_path.find_last_of("/\\");
    std::string dir = (slash == std::string::npos) ? std::string() : image_path.substr(0, slash + 1);
    std::string name = (slash == std::string::npos) ? image_path : image_path.substr(slash + 1);
    size_t dot = name.find_last_of('.');
    std::string stem = (dot == std::string::npos || dot == 0) ? name : name.substr(0, dot);
    if (stem.empty()) stem = "output";
    return dir + stem + "_locate.png";
}
