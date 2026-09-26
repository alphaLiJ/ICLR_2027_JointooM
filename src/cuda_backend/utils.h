#include <torch/extension.h>

constexpr std::array<uint16_t, 256> generate_pow2_table() {
    std::array<uint16_t, 256> table{};
    for (int i = 0; i < 256; ++i) {
        if (i <= 1) {
            table[i] = 1;
        } else {
            int n = i - 1;
            n |= n >> 1;
            n |= n >> 2;
            n |= n >> 4;
            table[i] = static_cast<uint16_t>(n + 1);
        }
    }
    return table;
}

alignas(64) static constexpr auto POW2_TABLE = generate_pow2_table();

inline int next_pow2_efficient(int x) {
    return POW2_TABLE[x];
}