// CPU-only Engram GGUF row regression: original Q8_0, Maya Q6_K, and mixed tables.
// Expected rows use ggml's independent dequantization traits, not the Engram decoder.
#include "ds41_engram.hpp"
#include "../../../tests/core/gguf_fixture.hpp"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unistd.h>

using namespace strata;
using namespace strata::v41;

namespace {

void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}

struct TempModel {
    std::string dir, path;
    TempModel() {
        char name[] = "/tmp/ds41-engram-XXXXXX";
        const char * p = ::mkdtemp(name);
        require(p != nullptr, "mkdtemp failed");
        dir = p;
        path = dir + "/tables.gguf";
    }
    ~TempModel() { ::unlink(path.c_str()); ::rmdir(dir.c_str()); }
};

struct CacheSetting {
    std::string previous;
    bool had_previous = false;
    CacheSetting() {
        if (const char * p = std::getenv("STRATA_ENGRAM_CACHE_ROWS")) {
            previous = p;
            had_previous = true;
        }
    }
    ~CacheSetting() {
        if (had_previous) ::setenv("STRATA_ENGRAM_CACHE_ROWS", previous.c_str(), 1);
        else ::unsetenv("STRATA_ENGRAM_CACHE_ROWS");
    }
};

Ds41Geometry geometry() {
    Ds41Geometry g;
    g.engram_dim = 256;
    g.engram_ngram = 3;
    g.engram_heads = 2;
    g.engram_pad = 0;
    g.engram_layers = {1, 14};
    g.engram_rows = {88, 102};
    g.engram_primes = {17, 19, 23, 29, 19, 23, 29, 31};
    g.engram_mult = {1, 3, 5, 7, 11, 13};
    g.engram_token_map = {0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89};
    return g;
}

std::vector<uint8_t> payload(uint32_t type, int rows, int table) {
    const size_t stride = ggml_row_size((ggml_type) type, 256);
    require(stride == (type == ty::Q8_0 ? 272u : 210u), "unexpected ggml row size");
    std::vector<uint8_t> bytes((size_t) rows * stride);
    uint32_t state = 0x12345678u + (uint32_t) table;
    for (uint8_t & x : bytes) {
        state = state * 1664525u + 1013904223u;
        x = (uint8_t) (state >> 24);
    }
    // Include signed and subnormal scales. Every row/block has different codes and scale;
    // a decoder or read accidentally fixed at row zero cannot pass.
    const uint16_t scales[] = {0x3400, 0xb800, 0x0001, 0x8003, 0x3c00};
    for (int row = 0; row < rows; ++row) {
        const int blocks = type == ty::Q8_0 ? 8 : 1;
        for (int block = 0; block < blocks; ++block) {
            const size_t at = (size_t) row * stride + (type == ty::Q8_0 ? (size_t) block * 34 : 208);
            const uint16_t scale = scales[(row + block + table) % 5];
            bytes[at] = (uint8_t) scale;
            bytes[at + 1] = (uint8_t) (scale >> 8);
        }
    }
    return bytes;
}

// Independently compute the n-gram ids, including each column's disjoint bucket offset.
std::vector<uint32_t> row_ids(const Ds41Geometry & g, const std::vector<int> & tokens,
                              int table, int pos0, int n) {
    std::vector<uint32_t> ids;
    for (int pos = pos0; pos < pos0 + n; ++pos) {
        int64_t rolling = (int64_t) g.engram_token_map[(size_t) tokens[(size_t) pos]] *
                          g.engram_mult[(size_t) table * 3];
        uint32_t offset = 0;
        for (int back = 1; back < 3; ++back) {
            rolling ^= (int64_t) g.engram_token_map[(size_t) tokens[(size_t) (pos - back)]] *
                       g.engram_mult[(size_t) table * 3 + (size_t) back];
            for (int head = 0; head < 2; ++head) {
                const uint32_t prime = (uint32_t) g.engram_primes[(size_t) table * 4 + (size_t) (back - 1) * 2 + (size_t) head];
                ids.push_back(offset + (uint32_t) (rolling % prime));
                offset += prime;
            }
        }
    }
    return ids;
}

void compare(const std::vector<std::vector<float>> & actual,
             const std::vector<std::vector<float>> & expected, const char * label) {
    require(actual.size() == expected.size(), std::string(label) + ": table count");
    for (size_t e = 0; e < expected.size(); ++e) {
        require(actual[e].size() == expected[e].size(), std::string(label) + ": row count");
        for (size_t i = 0; i < expected[e].size(); ++i) {
            if (std::memcmp(&actual[e][i], &expected[e][i], sizeof(float)) != 0) {
                throw std::runtime_error(std::string(label) + ": table " + std::to_string(e) +
                                         " value " + std::to_string(i) + " differs from ggml");
            }
        }
    }
}

void run(std::array<uint32_t, 2> types, int cache_rows, int rows_limit) {
    TempModel file;
    const Ds41Geometry g = geometry();
    std::vector<fixture::Tensor> tensors;
    std::array<std::vector<uint8_t>, 2> raw;
    for (size_t e = 0; e < types.size(); ++e) {
        tensors.push_back({"blk." + std::to_string(g.engram_layers[e]) + ".engram_embd.weight",
                           {256, (uint64_t) g.engram_rows[e]}, types[e], 1});
        raw[e] = payload(types[e], (int) g.engram_rows[e], (int) e);
    }
    const auto written = fixture::write(file.path, {}, tensors);
    {
        std::fstream f(file.path, std::ios::binary | std::ios::in | std::ios::out);
        for (size_t e = 0; e < raw.size(); ++e) {
            f.seekp((std::streamoff) (written.data_start + written.offsets[e]));
            f.write((const char *) raw[e].data(), (std::streamsize) raw[e].size());
        }
        require((bool) f, "writing table payloads failed");
    }
    const GgufModel model({file.path});
    ::setenv("STRATA_ENGRAM_CACHE_ROWS", std::to_string(cache_rows).c_str(), 1);
    Ds41Engram eng;
    std::string err;
    require(eng.open_gguf(g, model, rows_limit, 3, err), "open_gguf: " + err);
    constexpr int pos0 = 2, n = 48;
    std::vector<int> tokens((size_t) pos0 + n);
    for (size_t p = 0; p < tokens.size(); ++p) {
        tokens[p] = (int) ((p * 7 + p / 5) % g.engram_token_map.size());
        eng.set_token((int64_t) p, tokens[p]);
    }

    std::vector<std::vector<float>> expected(2);
    uint64_t reference_bytes = 0, grouped_bytes = 0, grouped_reads = 0, unique_rows = 0;
    for (size_t e = 0; e < types.size(); ++e) {
        const auto ids = row_ids(g, tokens, (int) e, pos0, n);
        const uint32_t readable = rows_limit > 0 ? (uint32_t) rows_limit : (uint32_t) g.engram_rows[e];
        require(rows_limit <= 0 || std::any_of(ids.begin(), ids.end(), [readable](uint32_t id) { return id >= readable; }),
                "capped fixture must exercise unreadable rows as well as nonzero readable rows");
        const size_t stride = ggml_row_size((ggml_type) types[e], 256);
        expected[e].assign(ids.size() * 256, 0.0f);
        std::vector<uint32_t> unique;
        bool nonzero = false;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] >= readable) continue;
            nonzero |= ids[i] > 0;
            unique.push_back(ids[i]);
            ggml_get_type_traits((ggml_type) types[e])->to_float(
                raw[e].data() + (size_t) ids[i] * stride, expected[e].data() + i * 256, 256);
            reference_bytes += stride;
        }
        require(nonzero, "fixture must exercise nonzero readable row ids in each table");
        std::sort(unique.begin(), unique.end());
        unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
        require(unique.size() < ids.size(), "fixture must exercise duplicate lookups");
        unique_rows += unique.size();
        const uint64_t base = written.data_start + written.offsets[e];
        for (size_t i = 0; i < unique.size();) {
            const uint64_t page = (base + (uint64_t) unique[i] * stride) / 4096;
            size_t j = i + 1;
            while (j < unique.size() && (base + (uint64_t) unique[j] * stride) / 4096 == page) ++j;
            grouped_bytes += ((uint64_t) unique[j - 1] - unique[i] + 1) * stride;
            ++grouped_reads;
            i = j;
        }
    }

    std::vector<std::vector<float>> actual;
    require(eng.rows_reference(pos0, n, actual, err), "reference reads: " + err);
    compare(actual, expected, "reference");
    require(eng.bytes_read() == reference_bytes, "reference byte accounting must use each table's row size");
    require(eng.rows(pos0, n, actual, err), "cold rows: " + err);
    compare(actual, expected, "cold/coalesced");
    require(eng.bytes_read() == reference_bytes + grouped_bytes, "coalesced byte accounting");
    require(eng.row_reads() == grouped_reads, "coalesced read count");
    require(eng.cache_hits() == 0, "cold cache must have no hits");
    require(eng.cache_misses() == (cache_rows ? unique_rows : 0), "cold cache unique misses");

    const uint64_t bytes_before = eng.bytes_read(), reads_before = eng.row_reads();
    auto batch = eng.start(pos0, n, actual, err);
    require(batch != nullptr && eng.wait(batch, err), "repeated async lookup: " + err);
    compare(actual, expected, "repeated/cache");
    const uint64_t cached = std::min<uint64_t>((uint64_t) cache_rows, unique_rows);
    require(eng.cache_hits() == cached, "repeat must hit every retained raw row, including mixed types");
    if (cached == unique_rows) {
        require(eng.bytes_read() == bytes_before && eng.row_reads() == reads_before,
                "fully cached repeat must perform no reads");
    } else {
        require(eng.bytes_read() > bytes_before && eng.row_reads() > reads_before,
                "uncached repeat must still read missing rows");
    }
    // A third lookup exercises the same cache offsets after a partially full cache rejects inserts.
    require(eng.rows(pos0, n, actual, err), "third lookup: " + err);
    compare(actual, expected, "third/cache");
    require(eng.cache_hits() == 2 * cached, "cache capacity must be counted in rows, not bytes");
    std::printf("ok Engram %s/%s cache=%d rows_limit=%d\n", strata::ggml_type_name(types[0]),
                strata::ggml_type_name(types[1]), cache_rows, rows_limit);
}

}  // namespace

int main() {
    CacheSetting restore_cache;
    try {
        for (const auto types : {std::array<uint32_t, 2>{ty::Q8_0, ty::Q8_0},
                                 std::array<uint32_t, 2>{ty::Q6_K, ty::Q6_K},
                                 std::array<uint32_t, 2>{ty::Q8_0, ty::Q6_K},
                                 std::array<uint32_t, 2>{ty::Q6_K, ty::Q8_0}}) {
            for (int cache_rows : {0, 7, 512}) run(types, cache_rows, 0);
            run(types, 512, 13);
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL Engram: %s\n", e.what());
        return 1;
    }
    return 0;
}
