#pragma once

// LocateAnything 运行期诊断工具：子图 shape dump + decode 过程 trace。
//
// 这些代码不参与推理计算，但排查"一个框都解不出来 / 坐标错位 / KV 行数不对"时是
// 唯一的现场证据来源，所以单独成文件保留，而不是把 fprintf 散在业务 .cpp 里：
//   - 默认输出到 stderr；
//   - 编译时加 -DLOCATE_NCNN_NO_TRACE 可整体静默（调用点保留，被折叠为空，
//     不会残留运行期开销）。

#include <cstdio>
#include <string>
#include <unordered_set>
#include <vector>

#include <mat.h>

#ifdef LOCATE_NCNN_NO_TRACE
#define LA_TRACE(...) ((void)0)
#define LA_SHAPE(tag, m) ((void)0)
#else
// 注意：fmt 必须是字符串字面量（宏内做拼接）。
#define LA_TRACE(...)                                                    \
    do {                                                                 \
        fprintf(stderr, "[locateanything] " __VA_ARGS__);                \
    } while (0)
#define LA_SHAPE(tag, m) locate_dbg::dump_shape_once((tag), (m))
#endif

namespace locate_dbg {

// 每个 tag 只打一次：稳态下同一子图的 shape 不变，逐次打印会淹没真正的 trace。
inline void dump_shape_once(const char* tag, const ncnn::Mat& m) {
    static std::unordered_set<std::string> seen;
    if (!seen.insert(tag).second) return;
    fprintf(stderr,
            "[locateanything] SHAPE %s dims=%d w=%d h=%d c=%d pack=%d es=%zu\n",
            tag, m.dims, m.w, m.h, m.c, m.elempack, m.elemsize);
}

// "[12,34,56]"
inline std::string join_ids(const std::vector<int>& ids) {
    std::string s = "[";
    for (size_t i = 0; i < ids.size(); i++) {
        if (i) s += ",";
        s += std::to_string(ids[i]);
    }
    s += "]";
    return s;
}

}  // namespace locate_dbg
