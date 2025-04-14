/* Copyright (c) 2021-2024, InterDigital Communications, Inc
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted (subject to the limitations in the disclaimer
* below) provided that the following conditions are met:
*
* * Redistributions of source code must retain the above copyright notice,
*   this list of conditions and the following disclaimer.
* * Redistributions in binary form must reproduce the above copyright notice,
*   this list of conditions and the following disclaimer in the documentation
*   and/or other materials provided with the distribution.
* * Neither the name of InterDigital Communications, Inc nor the names of its
*   contributors may be used to endorse or promote products derived from this
*   software without specific prior written permission.
*
* NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
* THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
* CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
* NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
* PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
* OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
* WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
* OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
* ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_MSC_VER)
  #include <intrin.h>
  static inline uint64_t Rans64MulHi(uint64_t a, uint64_t b){
      return __umulh(a, b);
  }
#elif defined(__GNUC__)
  static inline uint64_t Rans64MulHi(uint64_t a, uint64_t b) {
      return static_cast<uint64_t>(((unsigned __int128)a * b) >> 64);
  }
#else
  #error Unknown/unsupported compiler!
#endif

// Lower bound for normalization (as in the original paper)
#define RANS64_L (1ull << 31)
#ifdef assert
  #define Rans64Assert assert
#else
  #define Rans64Assert(x)
#endif
  
namespace cpp_exts {

// --------------------------------------------------------------------------
// Precision and bypass parameters

constexpr int precision = 16;
constexpr uint16_t bypass_precision = 4; // number of bits in bypass mode
constexpr uint16_t max_bypass_val = (1 << bypass_precision) - 1;

using Rans64State = uint64_t;

// rANS encoder functions
static inline void Rans64EncInit(Rans64State* r) {
    *r = RANS64_L;
}

static inline void Rans64EncPut(Rans64State* r, uint32_t** pptr, uint32_t start, uint32_t freq, uint32_t scale_bits) {
    Rans64Assert(freq != 0);
    uint64_t x = *r;
    uint64_t x_max = ((RANS64_L >> scale_bits) << 32) * freq;
    if (x >= x_max) {
        *pptr -= 1;
        **pptr = static_cast<uint32_t>(x);
        x >>= 32;
        Rans64Assert(x < x_max);
    }
    *r = ((x / freq) << scale_bits) + (x % freq) + start;
}

static inline void Rans64EncFlush(Rans64State* r, uint32_t** pptr) {
    uint64_t x = *r;
    *pptr -= 2;
    (*pptr)[0] = static_cast<uint32_t>(x >> 0);
    (*pptr)[1] = static_cast<uint32_t>(x >> 32);
}

// rANS decoder functions
static inline void Rans64DecInit(Rans64State* r, uint32_t** pptr) {
    uint64_t x = (static_cast<uint64_t>((*pptr)[0]) << 0) |
                 (static_cast<uint64_t>((*pptr)[1]) << 32);
    *pptr += 2;
    *r = x;
}

static inline uint32_t Rans64DecGet(Rans64State* r, uint32_t scale_bits) {
    return static_cast<uint32_t>(*r & ((1u << scale_bits) - 1));
}

static inline void Rans64DecAdvance(Rans64State* r, uint32_t** pptr, uint32_t start, uint32_t freq, uint32_t scale_bits) {
    uint64_t mask = (1ull << scale_bits) - 1;
    uint64_t x = *r;
    x = freq * (x >> scale_bits) + (x & mask) - start;
    if (x < RANS64_L) {
        x = (x << 32) | **pptr;
        *pptr += 1;
        Rans64Assert(x >= RANS64_L);
    }
    *r = x;
}

// Support for bypass mode (raw bits) encoding/decoding
inline void Rans64EncPutBits(Rans64State *r, uint32_t **pptr, uint32_t val, uint32_t nbits) {
    uint64_t x = *r;
    uint32_t freq = 1 << (16 - nbits);
    uint64_t x_max = ((RANS64_L >> 16) << 32) * freq;
    if (x >= x_max) {
        *pptr -= 1;
        **pptr = static_cast<uint32_t>(x);
        x >>= 32;
        Rans64Assert(x < x_max);
    }
    *r = (x << nbits) | val;
}

inline uint32_t Rans64DecGetBits(Rans64State *r, uint32_t **pptr, uint32_t n_bits) {
    uint64_t x = *r;
    uint32_t val = static_cast<uint32_t>(x & ((1u << n_bits) - 1));
    x = x >> n_bits;
    if (x < RANS64_L) {
        x = (x << 32) | **pptr;
        *pptr += 1;
        Rans64Assert(x >= RANS64_L);
    }
    *r = x;
    return val;
}

// --------------------------------------------------------------------------
// Data structure for symbol and encoder/decoder classes

struct RansSymbol {
    uint16_t start;
    uint16_t range;
    bool bypass; // flag indicating raw bits (bypass mode)
};

// --------------------------------------------------------------------------

// BufferedRansEncoder collects symbols before performing the actual rANS encode.
// It is used internally by RansEncoder.
class BufferedRansEncoder {
public:
    BufferedRansEncoder() = default;
    BufferedRansEncoder(const BufferedRansEncoder &) = delete;
    BufferedRansEncoder(BufferedRansEncoder &&) = delete;
    BufferedRansEncoder &operator=(const BufferedRansEncoder &) = delete;
    BufferedRansEncoder &operator=(BufferedRansEncoder &&) = delete;

    // The following function assumes:
    // - symbols: 1D int32_t tensor of input symbols.
    // - indexes: 1D int32_t tensor (same length as symbols) indicating which CDF row to use.
    // - cdfs: 2D int32_t tensor with shape (n_cdfs, max_cdf_length).
    // - cdfs_sizes: 1D int32_t tensor of lengths (for each CDF row).
    // - offsets: 1D int32_t tensor of offsets (for each CDF row).
    void encode_with_indexes(const at::Tensor &symbols,
                             const at::Tensor &indexes,
                             const at::Tensor &cdfs,
                             const at::Tensor &cdfs_sizes,
                             const at::Tensor &offsets);
    // Flush the current symbols into a compressed byte stream tensor (dtype uint8)
    at::Tensor flush();

private:
    std::vector<RansSymbol> _syms;
};

// Implementation of BufferedRansEncoder::encode_with_indexes
void BufferedRansEncoder::encode_with_indexes(const at::Tensor &symbols, 
                                              const at::Tensor &indexes,
                                              const at::Tensor &cdfs,
                                              const at::Tensor &cdfs_sizes,
                                              const at::Tensor &offsets) {
    // Extract raw pointers for 1D tensors.
    const int32_t* symbols_ptr = symbols.data_ptr<int32_t>();
    const int32_t* indexes_ptr = indexes.data_ptr<int32_t>();
    const int32_t* cdfs_sizes_ptr = cdfs_sizes.data_ptr<int32_t>();
    const int32_t* offsets_ptr = offsets.data_ptr<int32_t>();

    // For the CDFs, we assume a 2D tensor of shape (n_cdfs, max_cdf_length).
    auto cdfs_accessor = cdfs.accessor<int32_t, 2>();
    int64_t n_sym = symbols.size(0);

    // Iterate over symbols in order (we will later pop from the vector to reverse).
    for (int64_t i = 0; i < n_sym; ++i) {
        int32_t cdf_idx = indexes_ptr[i];
        int32_t cdf_size = cdfs_sizes_ptr[cdf_idx];
        int32_t max_value = cdf_size - 2;
        int32_t offset = offsets_ptr[cdf_idx];
        int32_t value = symbols_ptr[i] - offset;

        int32_t raw_val = 0;
        if (value < 0) {
            raw_val = -2 * value - 1;
            value = max_value;
        } else if (value >= max_value) {
            raw_val = 2 * (value - max_value);
            value = max_value;
        }

        // Note: We assume that each CDF is stored in a row of the 2D tensor.
        // The CDF table is assumed to be non-decreasing.
        uint16_t start = static_cast<uint16_t>( cdfs_accessor[cdf_idx][value] );
        uint16_t next = static_cast<uint16_t>( cdfs_accessor[cdf_idx][value+1] );
        uint16_t range = next - start;
        _syms.push_back({start, range, false});

        // If symbol equals the max_value, then we enter bypass mode.
        if (value == max_value) {
            // Determine the number of bypass (raw bits) groups needed.
            int32_t n_bypass = 0;
            while ((raw_val >> (n_bypass * bypass_precision)) != 0) {
                ++n_bypass;
            }
            // Encode the number of bypass groups.
            int32_t val = n_bypass;
            while (val >= max_bypass_val) {
                _syms.push_back({max_bypass_val, max_bypass_val + 1, true});
                val -= max_bypass_val;
            }
            _syms.push_back({static_cast<uint16_t>(val), static_cast<uint16_t>(val + 1), true});
            // Encode the raw value bits.
            for (int32_t j = 0; j < n_bypass; ++j) {
                int32_t bits = (raw_val >> (j * bypass_precision)) & max_bypass_val;
                _syms.push_back({static_cast<uint16_t>(bits), static_cast<uint16_t>(bits + 1), true});
            }
        }
    }
}

// Implementation of BufferedRansEncoder::flush
//
// This function performs the actual rANS encoding of the symbols stored in _syms.
// It writes 32-bit words in reverse order. The final encoded stream is returned as a
// torch::Tensor of type uint8.
at::Tensor BufferedRansEncoder::flush() {
    Rans64State rans;
    Rans64EncInit(&rans);

    // Reserve enough space. In worst-case, we may need more space than _syms.size().
    // Here we allocate one uint32_t per symbol plus some extra.
    std::vector<uint32_t> output(_syms.size() + 10, 0xCC);
    uint32_t* ptr = output.data() + output.size();  // pointer at the end

    // Process the symbols in reverse order.
    while (! _syms.empty()) {
        RansSymbol sym = _syms.back();
        _syms.pop_back();
        if (!sym.bypass) {
            Rans64EncPut(&rans, &ptr, sym.start, sym.range, precision);
        } else {
            Rans64EncPutBits(&rans, &ptr, sym.start, bypass_precision);
        }
    }

    Rans64EncFlush(&rans, &ptr);
    // Compute number of uint32_t words that were written.
    int n_words = static_cast<int>((output.data() + output.size()) - ptr);
    int nbytes = n_words * static_cast<int>(sizeof(uint32_t));

    // Create an output tensor (uint8) and copy the encoded data.
    auto options = at::TensorOptions().dtype(at::kByte);
    at::Tensor encoded_tensor = at::empty({nbytes}, options);
    // Copy data from ptr into the tensor.
    std::memcpy(encoded_tensor.data_ptr<uint8_t>(), ptr, nbytes);
    return encoded_tensor;
}

// --------------------------------------------------------------------------

// RansEncoder provides a one‐shot interface to encode symbols with their CDFs.
class RansEncoder : public torch::CustomClassHolder {
public:
    RansEncoder() = default;

    at::Tensor encode_with_indexes(const at::Tensor &symbols,
                                   const at::Tensor &indexes,
                                   const at::Tensor &cdfs,
                                   const at::Tensor &cdfs_sizes,
                                   const at::Tensor &offsets);
};

// Implementation of RansEncoder::encode_with_indexes
at::Tensor RansEncoder::encode_with_indexes(const at::Tensor &symbols,
                                            const at::Tensor &indexes,
                                            const at::Tensor &cdfs,
                                            const at::Tensor &cdfs_sizes,
                                            const at::Tensor &offsets) {
    BufferedRansEncoder buffered_enc;
    buffered_enc.encode_with_indexes(symbols, indexes, cdfs, cdfs_sizes, offsets);
    return buffered_enc.flush();
}

// --------------------------------------------------------------------------

// RansDecoder provides decoding functions. Note that the “stream” here is the
// encoded tensor of type uint8. To keep the interface stateless for one‐shot
// decoding we provide a decode_with_indexes method that takes the encoded stream,
// decodes and returns the output symbols.
class RansDecoder : public torch::CustomClassHolder {
public:
    RansDecoder() = default;

    // Decodes from a uint8 tensor containing the encoded rANS stream.
    at::Tensor decode_with_indexes(const at::Tensor &encoded,
                                   const at::Tensor &indexes,
                                   const at::Tensor &cdfs,
                                   const at::Tensor &cdfs_sizes,
                                   const at::Tensor &offsets);
};

// Implementation of RansDecoder::decode_with_indexes
//
// The encoded stream is provided as a tensor of dtype uint8.
// The CDFs, cdfs_sizes, and offsets use the same conventions as in the encoder.
at::Tensor RansDecoder::decode_with_indexes(const at::Tensor &encoded,
                                            const at::Tensor &indexes,
                                            const at::Tensor &cdfs,
                                            const at::Tensor &cdfs_sizes,
                                            const at::Tensor &offsets) {
    // Assume output tensor is same length as indexes.
    int64_t n_sym = indexes.size(0);
    auto output = at::empty({n_sym}, at::kInt);

    // Convert the encoded tensor (of uint8) into a pointer to uint32_t.
    // We require that the encoded data is aligned to 4 bytes.
    uint8_t* encoded_bytes = encoded.data_ptr<uint8_t>();
    uint32_t* ptr = reinterpret_cast<uint32_t*>(encoded_bytes);

    Rans64State rans;
    Rans64DecInit(&rans, &ptr);

    // Set up pointer access for the other tensors.
    const int32_t* indexes_ptr = indexes.data_ptr<int32_t>();
    const int32_t* cdfs_sizes_ptr = cdfs_sizes.data_ptr<int32_t>();
    const int32_t* offsets_ptr = offsets.data_ptr<int32_t>();
    auto cdfs_accessor = cdfs.accessor<int32_t, 2>();
    int32_t* output_ptr = output.data_ptr<int32_t>();

    for (int64_t i = 0; i < n_sym; ++i) {
        int32_t cdf_idx = indexes_ptr[i];
        int32_t cdf_size = cdfs_sizes_ptr[cdf_idx];
        int32_t max_value = cdf_size - 2;
        int32_t offset = offsets_ptr[cdf_idx];

        // Look up the current cumulative frequency.
        uint32_t cum_freq = Rans64DecGet(&rans, precision);

        // Perform a search on the CDF table (stored in row cdf_idx).
        int32_t found = -1;
        int32_t cdf_length = cdf_size;
        for (int j = 0; j < cdf_length - 1; ++j) {
            // Find j such that cdfs[row][j] <= cum_freq < cdfs[row][j+1]
            if (cdfs_accessor[cdf_idx][j] <= static_cast<int32_t>(cum_freq) &&
                static_cast<int32_t>(cum_freq) < cdfs_accessor[cdf_idx][j+1])
            {
                found = j;
                break;
            }
        }
        if (found < 0) {
            throw std::runtime_error("CDF lookup failed during decoding");
        }
        int s = found;
        uint16_t start = static_cast<uint16_t>( cdfs_accessor[cdf_idx][s] );
        uint16_t next = static_cast<uint16_t>( cdfs_accessor[cdf_idx][s+1] );
        uint16_t range = next - start;
        Rans64DecAdvance(&rans, &ptr, start, range, precision);
        int32_t value = s;
        if (value == max_value) {
            // Bypass decoding mode.
            int32_t val = Rans64DecGetBits(&rans, &ptr, bypass_precision);
            int32_t n_bypass = val;
            while (val == max_bypass_val) {
                val = Rans64DecGetBits(&rans, &ptr, bypass_precision);
                n_bypass += val;
            }
            int32_t raw_val = 0;
            for (int j = 0; j < n_bypass; ++j) {
                val = Rans64DecGetBits(&rans, &ptr, bypass_precision);
                raw_val |= (val << (j * bypass_precision));
            }
            value = raw_val >> 1;
            if (raw_val & 1)
                value = -value - 1;
            else
                value += max_value;
        }
        output_ptr[i] = value + offset;
    }

    return output;
}

// --------------------------------------------------------------------------
// Define a namespace for your operators (e.g., "rans_ops")
TORCH_LIBRARY(cpp_exts, m) {
    // Register the RansEncoder class
    m.class_<RansEncoder>("RansEncoder")
        // Register the constructor (__init__)
        .def(torch::init<>())
        // Register the encode_with_indexes method
        // Format: .def("python_method_name", &ClassName::MethodName)
        .def("encode_with_indexes", &RansEncoder::encode_with_indexes);

    // Register the RansDecoder class
    m.class_<RansDecoder>("RansDecoder")
        // Register the constructor (__init__)
        .def(torch::init<>())
        // Register the decode_with_indexes method
        .def("decode_with_indexes", &RansDecoder::decode_with_indexes);
}

} // namespace cpp_exts

// --------------------------------------------------------------------------

extern "C" {
    /* Creates a dummy empty _C module that can be imported from Python.
       The import from Python will load the .so consisting of this file in this
       extension, so that the TORCH_LIBRARY static initializers below are run. */
    PyObject* PyInit__C(void) {
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "_C",   /* name of module */
            NULL,   /* module documentation, may be NULL */
            -1,     /* size of per-interpreter state of the module,
                       or -1 if the module keeps state in global variables. */
            NULL,   /* methods */
        };
        return PyModule_Create(&module_def);
    }
}