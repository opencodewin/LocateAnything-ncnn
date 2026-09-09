#pragma once

#include <string>
#include <vector>

#include <mat.h>

// 一个定位结果框。坐标统一为**归一化** [0,1]（相对原图宽/高），
// 与模型输出的 <0>~<1000> 量化坐标一一对应（coord / 1000）。
struct LocateBox {
    float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;
    bool is_point = false;   // <box><x><y></box>：点框（画成实心圆点 + 十字）
    std::string label;       // 可选文本（<ref>...</ref> 指代结果等）
};

// 从模型输出文本里解析 <box>...</box>（大小写不敏感）。支持三种写法：
//   <box><123><45><678><900></box>   量化坐标 token（0~1000，自动 /1000）
//   <box>123 45 678 900</box>        已转成整数（同上，自动 /1000）
//   <box>0.12 0.34 0.56 0.78</box>   归一化浮点（直接使用）
// 只有 2 个数字时按"点框"处理（<box><x><y></box>）。
std::vector<LocateBox> parse_locate_boxes_text(const std::string& text);

// 单行格式化："#0 px=(12,34)-(56,78) norm=(0.012,0.034)-(0.056,0.078)"。
// img_w/img_h <= 0 时只输出归一化坐标。
std::string format_locate_box(const LocateBox& box, int img_w, int img_h, int index);

// 把归一化框画到 bgr（BGR u8 interleaved，load_image_to_ncnn_mat 的结果）上，
// 并保存为 PNG。无第三方依赖：内置 stored-deflate 的极简 PNG 编码器。
// thickness=0 时按图像尺寸自动取线宽。
bool draw_locate_boxes(const ncnn::Mat& bgr, const std::vector<LocateBox>& boxes,
                       const std::string& out_png, int thickness = 0);

// 默认标注输出路径：<dir>/<stem>_locate.png
std::string default_annotated_path(const std::string& image_path);
