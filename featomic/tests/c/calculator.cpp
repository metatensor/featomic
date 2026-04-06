#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>

#include "featomic.h"
#include "catch.hpp"
#include "helpers.hpp"

// ============================================================================
// Helpers for the new opaque mts_labels_t API
// ============================================================================

// Extract a double* data pointer from an mts_array_t via DLPack
static double* dlpack_data_ptr(mts_array_t& array) {
    DLManagedTensorVersioned* dl = nullptr;
    DLDevice cpu_device = {kDLCPU, 0};
    DLPackVersion max_ver = {1, 0};
    auto status = array.as_dlpack(array.ptr, &dl, cpu_device, nullptr, max_ver);
    if (status != MTS_SUCCESS || dl == nullptr) {
        return nullptr;
    }
    auto* ptr = static_cast<double*>(dl->dl_tensor.data);
    if (dl->deleter) {
        dl->deleter(dl);
    }
    return ptr;
}

// Managed DLPack handle that cleans up on destruction
struct DLPackHandle {
    DLManagedTensorVersioned* dl = nullptr;
    ~DLPackHandle() { if (dl && dl->deleter) dl->deleter(dl); }
    // non-copyable
    DLPackHandle() = default;
    DLPackHandle(const DLPackHandle&) = delete;
    DLPackHandle& operator=(const DLPackHandle&) = delete;
    DLPackHandle(DLPackHandle&& o) noexcept : dl(o.dl) { o.dl = nullptr; }
    DLPackHandle& operator=(DLPackHandle&& o) noexcept { if (dl && dl->deleter) dl->deleter(dl); dl = o.dl; o.dl = nullptr; return *this; }
};

// Extract an int32_t* data pointer from an mts_array_t via DLPack.
// The returned DLPackHandle must outlive the pointer.
static int32_t* dlpack_i32_data_ptr(mts_array_t& array, DLPackHandle& handle) {
    DLDevice cpu_device = {kDLCPU, 0};
    DLPackVersion max_ver = {1, 0};
    auto status = array.as_dlpack(array.ptr, &handle.dl, cpu_device, nullptr, max_ver);
    if (status != MTS_SUCCESS || handle.dl == nullptr) {
        return nullptr;
    }
    return static_cast<int32_t*>(handle.dl->dl_tensor.data);
}

// Wrapper struct holding int32 data + shape for creating mts_array_t
struct I32ArrayData {
    std::vector<int32_t> values;
    std::vector<uintptr_t> shape;
    // For DLPack export
    std::vector<int64_t> dl_shape;
};

static mts_array_t make_i32_array(std::vector<int32_t> data, uintptr_t count, uintptr_t size) {
    mts_array_t array = {};

    auto* holder = new I32ArrayData();
    holder->values = std::move(data);
    holder->shape = {count, size};
    holder->dl_shape = {static_cast<int64_t>(count), static_cast<int64_t>(size)};

    array.ptr = holder;

    array.destroy = [](void* ptr) {
        delete static_cast<I32ArrayData*>(ptr);
    };

    array.origin = [](const void*, mts_data_origin_t* origin) -> mts_status_t {
        mts_register_data_origin("c-tests-i32-array", origin);
        return MTS_SUCCESS;
    };

    array.shape = [](const void* ptr, const uintptr_t** shape, uintptr_t* shape_count) -> mts_status_t {
        auto* h = static_cast<const I32ArrayData*>(ptr);
        *shape = h->shape.data();
        *shape_count = h->shape.size();
        return MTS_SUCCESS;
    };

    array.device = [](const void*, DLDevice* device) -> mts_status_t {
        device->device_type = kDLCPU;
        device->device_id = 0;
        return MTS_SUCCESS;
    };

    array.dtype = [](const void*, DLDataType* dtype) -> mts_status_t {
        dtype->code = kDLInt;
        dtype->bits = 32;
        dtype->lanes = 1;
        return MTS_SUCCESS;
    };

    array.as_dlpack = [](void* ptr, DLManagedTensorVersioned** out, DLDevice, const int64_t*, DLPackVersion) -> mts_status_t {
        auto* h = static_cast<I32ArrayData*>(ptr);

        auto* managed = static_cast<DLManagedTensorVersioned*>(std::calloc(1, sizeof(DLManagedTensorVersioned)));
        managed->version = {1, 0};
        managed->dl_tensor.data = h->values.data();
        managed->dl_tensor.device = {kDLCPU, 0};
        managed->dl_tensor.ndim = static_cast<int32_t>(h->dl_shape.size());
        managed->dl_tensor.shape = h->dl_shape.data();
        managed->dl_tensor.strides = nullptr;
        managed->dl_tensor.byte_offset = 0;
        managed->dl_tensor.dtype = {kDLInt, 32, 1};
        managed->manager_ctx = nullptr;
        managed->deleter = [](DLManagedTensorVersioned* self) { std::free(self); };

        *out = managed;
        return MTS_SUCCESS;
    };

    array.copy = [](const void* ptr, mts_array_t* new_array) -> mts_status_t {
        auto* h = static_cast<const I32ArrayData*>(ptr);
        *new_array = make_i32_array(h->values, h->shape[0], h->shape[1]);
        return MTS_SUCCESS;
    };

    return array;
}

// Create an mts_labels_t* from names + int32 values
static mts_labels_t* create_labels(
    const std::vector<const char*>& names,
    const std::vector<int32_t>& values,
    uintptr_t count
) {
    auto array = make_i32_array(values, count, names.size());
    return mts_labels_create(names.data(), names.size(), array);
}

// Query labels: get dimension count (old .size), entry count (old .count), names, and values
struct LabelsInfo {
    uintptr_t size;       // number of dimensions
    uintptr_t count;      // number of entries
    const char* const* names;
    std::vector<int32_t> values; // copied out
};

static LabelsInfo query_labels(const mts_labels_t* labels) {
    LabelsInfo info;
    info.size = 0;
    info.count = 0;
    info.names = nullptr;

    mts_labels_dimensions(labels, &info.names, &info.size);

    mts_array_t values_array = {};
    mts_labels_values(labels, &values_array);

    const uintptr_t* shape = nullptr;
    uintptr_t shape_count = 0;
    values_array.shape(values_array.ptr, &shape, &shape_count);

    if (shape_count >= 2) {
        info.count = shape[0];
        // shape[1] should equal info.size
    } else if (shape_count == 1) {
        info.count = shape[0];
    }

    // Extract values via DLPack -- keep handle alive during copy
    DLPackHandle handle;
    DLDevice cpu_device = {kDLCPU, 0};
    DLPackVersion max_ver = {1, 0};
    auto status = values_array.as_dlpack(values_array.ptr, &handle.dl, cpu_device, nullptr, max_ver);
    if (status == MTS_SUCCESS && handle.dl != nullptr) {
        auto* ptr = static_cast<int32_t*>(handle.dl->dl_tensor.data);
        info.values.assign(ptr, ptr + info.count * info.size);
    }

    return info;
}


// ============================================================================
// Forward declaration
// ============================================================================

static void check_block(
    mts_tensormap_t* descriptor,
    size_t block_id,
    const std::vector<int32_t>& samples,
    const std::vector<int32_t>& properties,
    const std::vector<double>& values,
    const std::vector<int32_t>& gradient_samples,
    const std::vector<double>& gradients
);

TEST_CASE("calculator name") {
    SECTION("dummy_calculator") {
        const char* HYPERS_JSON = R"({
            "cutoff": 3.5,
            "delta": 25,
            "name": "bar"
        })";
        auto* calculator = featomic_calculator("dummy_calculator", HYPERS_JSON);
        REQUIRE(calculator != nullptr);

        char buffer[256] = {};
        CHECK_SUCCESS(featomic_calculator_name(calculator, buffer, sizeof(buffer)));
        CHECK(buffer == std::string("dummy test calculator with cutoff: 3.5 - delta: 25 - name: bar"));

        featomic_calculator_free(calculator);
    }

    SECTION("long strings") {
        auto name = std::string(2048, 'b');
        auto HYPERS_JSON = R"({
            "cutoff": 3.5,
            "delta": 25,
            "name": ")" + name + "\"}";

        auto* calculator = featomic_calculator("dummy_calculator", HYPERS_JSON.c_str());
        REQUIRE(calculator != nullptr);

        char* buffer = new char[4096];
        auto status = featomic_calculator_name(calculator, buffer, 256);
        CHECK(status == FEATOMIC_BUFFER_SIZE_ERROR);

        CHECK_SUCCESS(featomic_calculator_name(calculator, buffer, 4096));
        std::string expected = "dummy test calculator with cutoff: 3.5 - delta: 25 - ";
        expected += "name: " + name;
        CHECK(buffer == expected);

        delete[] buffer;

        featomic_calculator_free(calculator);
    }
}

TEST_CASE("calculator parameters") {
    SECTION("dummy_calculator") {
        std::string HYPERS_JSON = R"({
            "cutoff": 3.5,
            "delta": 25,
            "name": "bar"
        })";
        auto* calculator = featomic_calculator("dummy_calculator", HYPERS_JSON.c_str());
        REQUIRE(calculator != nullptr);

        char buffer[256] = {};
        CHECK_SUCCESS(featomic_calculator_parameters(calculator, buffer, sizeof(buffer)));
        CHECK(buffer == HYPERS_JSON);

        featomic_calculator_free(calculator);
    }

    SECTION("long strings") {
        auto name = std::string(2048, 'b');
        auto HYPERS_JSON = R"({
            "cutoff": 3.5,
            "delta": 25,
            "name": ")" + name + "\"}";

        auto* calculator = featomic_calculator("dummy_calculator", HYPERS_JSON.c_str());
        REQUIRE(calculator != nullptr);

        char* buffer = new char[4096];
        auto status = featomic_calculator_parameters(calculator, buffer, 256);
        CHECK(status == FEATOMIC_BUFFER_SIZE_ERROR);

        CHECK_SUCCESS(featomic_calculator_parameters(calculator, buffer, 4096));
        CHECK(buffer == HYPERS_JSON);

        delete[] buffer;

        featomic_calculator_free(calculator);
    }

    SECTION("cutoffs") {
        std::string HYPERS_JSON = R"({
            "cutoff": 3.5,
            "delta": 25,
            "name": "bar"
        })";
        auto* calculator = featomic_calculator("dummy_calculator", HYPERS_JSON.c_str());
        REQUIRE(calculator != nullptr);

        const double* cutoffs = nullptr;
        uintptr_t cutoffs_count = 0;
        CHECK_SUCCESS(featomic_calculator_cutoffs(calculator, &cutoffs, &cutoffs_count));
        CHECK(cutoffs_count == 1);
        CHECK(cutoffs[0] == 3.5);

        featomic_calculator_free(calculator);
    }
}

TEST_CASE("calculator creation errors") {
    const char* HYPERS_JSON = R"({
        "cutoff": "532",
        "delta": 25,
        "name": "bar"
    })";
    auto *calculator = featomic_calculator("dummy_calculator", HYPERS_JSON);
    CHECK(calculator == nullptr);

    CHECK(std::string(featomic_last_error()) == "json error: invalid type: string \"532\", expected f64 at line 2 column 23");
}

TEST_CASE("Compute descriptor") {
    const char* HYPERS_JSON = R"({
        "cutoff": 3.0,
        "delta": 4,
        "name": ""
    })";

    SECTION("Full compute") {
        auto system = simple_system();

        featomic_calculation_options_t options;
        std::memset(&options, 0, sizeof(featomic_calculation_options_t));

        const char* gradients_list[] = {"positions"};
        options.gradients = gradients_list;
        options.gradients_count = 1;
        auto* calculator = featomic_calculator("dummy_calculator", HYPERS_JSON);
        REQUIRE(calculator != nullptr);

        mts_tensormap_t* descriptor = nullptr;
        auto status = featomic_calculator_compute(
            calculator, &descriptor, &system, 1, options
        );
        CHECK_SUCCESS(status);

        auto* keys = mts_tensormap_keys(descriptor);
        REQUIRE(keys != nullptr);
        auto keys_info = query_labels(keys);

        CHECK(keys_info.size == 1);
        CHECK(keys_info.names[0] == std::string("center_type"));
        CHECK(keys_info.count == 2);
        CHECK(keys_info.values[0] == 1);
        CHECK(keys_info.values[1] == 6);
        mts_labels_free(keys);

        auto samples = std::vector<int32_t>{
            0, 1, /**/ 0, 2, /**/ 0, 3,
        };
        auto properties = std::vector<int32_t>{
            1, 0, /**/ 0, 1,
        };
        auto values = std::vector<double>{
            5, 39, /**/ 6, 18, /**/ 7, 15,
        };
        auto gradient_samples = std::vector<int32_t>{
            0, 0, 0, /**/ 0, 0, 1, /**/ 0, 0, 2,
            1, 0, 1, /**/ 1, 0, 2, /**/ 1, 0, 3,
            2, 0, 2, /**/ 2, 0, 3,
        };
        auto gradients = std::vector<double>{
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0
        };

        // H block
        check_block(descriptor, 0, samples, properties, values, gradient_samples, gradients);

        samples = std::vector<int32_t>{
            0, 0,
        };
        values = std::vector<double>{
            4, 33,
        };
        gradient_samples = std::vector<int32_t>{
            0, 0, 0, /**/ 0, 0, 1,
        };
        gradients = std::vector<double>{
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0
        };

        // C block
        check_block(descriptor, 1, samples, properties, values, gradient_samples, gradients);

        mts_tensormap_free(descriptor);
        featomic_calculator_free(calculator);
    }

    SECTION("Partial compute -- samples") {
        auto selected_sample_values = std::vector<int32_t>{
            0, 1, /**/ 0, 3,
        };
        auto selected_sample_names = std::vector<const char*>{
            "system", "atom"
        };

        auto* selected_samples = create_labels(selected_sample_names, selected_sample_values, 2);
        REQUIRE(selected_samples != nullptr);

        auto system = simple_system();

        featomic_calculation_options_t options;
        std::memset(&options, 0, sizeof(featomic_calculation_options_t));

        const char* gradients_list[] = {"positions"};
        options.gradients = gradients_list;
        options.gradients_count = 1;
        options.selected_samples.subset = selected_samples;
        auto* calculator = featomic_calculator("dummy_calculator", HYPERS_JSON);
        REQUIRE(calculator != nullptr);

        mts_tensormap_t* descriptor = nullptr;
        auto status = featomic_calculator_compute(
            calculator, &descriptor, &system, 1, options
        );

        CHECK_SUCCESS(status);

        auto* keys = mts_tensormap_keys(descriptor);
        REQUIRE(keys != nullptr);
        auto keys_info = query_labels(keys);

        CHECK(keys_info.size == 1);
        CHECK(keys_info.names[0] == std::string("center_type"));
        CHECK(keys_info.count == 2);
        CHECK(keys_info.values[0] == 1);
        CHECK(keys_info.values[1] == 6);
        mts_labels_free(keys);

        auto samples = std::vector<int32_t>{
            0, 1, /**/ 0, 3,
        };
        auto properties = std::vector<int32_t>{
            1, 0, /**/ 0, 1,
        };
        auto values = std::vector<double>{
            5, 39, /**/ 7, 15,
        };
        auto gradient_samples = std::vector<int32_t>{
            0, 0, 0, /**/ 0, 0, 1, /**/ 0, 0, 2,
            1, 0, 2, /**/ 1, 0, 3,
        };
        auto gradients = std::vector<double>{
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0
        };

        // H block
        check_block(descriptor, 0, samples, properties, values, gradient_samples, gradients);

        samples = std::vector<int32_t>{};
        values = std::vector<double>{};
        gradient_samples = std::vector<int32_t>{};
        gradients = std::vector<double>{};

        // C block
        check_block(descriptor, 1, samples, properties, values, gradient_samples, gradients);

        mts_labels_free(selected_samples);
        mts_tensormap_free(descriptor);
        featomic_calculator_free(calculator);
    }

    SECTION("Partial compute -- features") {
        auto selected_property_values = std::vector<int32_t>{
            0, 1,
        };
        auto selected_property_names = std::vector<const char*>{
            "index_delta", "x_y_z"
        };

        auto* selected_properties = create_labels(selected_property_names, selected_property_values, 1);
        REQUIRE(selected_properties != nullptr);

        auto system = simple_system();

        featomic_calculation_options_t options;
        std::memset(&options, 0, sizeof(featomic_calculation_options_t));

        const char* gradients_list[] = {"positions"};
        options.gradients = gradients_list;
        options.gradients_count = 1;
        options.selected_properties.subset = selected_properties;
        auto* calculator = featomic_calculator("dummy_calculator", HYPERS_JSON);
        REQUIRE(calculator != nullptr);

        mts_tensormap_t* descriptor = nullptr;
        auto status = featomic_calculator_compute(
            calculator, &descriptor, &system, 1, options
        );
        CHECK_SUCCESS(status);

        auto* keys = mts_tensormap_keys(descriptor);
        REQUIRE(keys != nullptr);
        auto keys_info = query_labels(keys);

        CHECK(keys_info.size == 1);
        CHECK(keys_info.names[0] == std::string("center_type"));
        CHECK(keys_info.count == 2);
        CHECK(keys_info.values[0] == 1);
        CHECK(keys_info.values[1] == 6);
        mts_labels_free(keys);

        auto samples = std::vector<int32_t>{
            0, 1, /**/ 0, 2, /**/ 0, 3,
        };
        auto properties = std::vector<int32_t>{
            0, 1,
        };
        auto values = std::vector<double>{
            39, /**/ 18, /**/ 15,
        };
        auto gradient_samples = std::vector<int32_t>{
            0, 0, 0, /**/ 0, 0, 1, /**/ 0, 0, 2,
            1, 0, 1, /**/ 1, 0, 2, /**/ 1, 0, 3,
            2, 0, 2, /**/ 2, 0, 3,
        };
        auto gradients = std::vector<double>{
            1.0, /**/ 1.0, /**/ 1.0,
            1.0, /**/ 1.0, /**/ 1.0,
            1.0, /**/ 1.0, /**/ 1.0,
            1.0, /**/ 1.0, /**/ 1.0,
            1.0, /**/ 1.0, /**/ 1.0,
            1.0, /**/ 1.0, /**/ 1.0,
            1.0, /**/ 1.0, /**/ 1.0,
            1.0, /**/ 1.0, /**/ 1.0
        };

        // H block
        check_block(descriptor, 0, samples, properties, values, gradient_samples, gradients);

        samples = std::vector<int32_t>{
            0, 0,
        };
        values = std::vector<double>{
            33,
        };
        gradient_samples = std::vector<int32_t>{
            0, 0, 0, /**/ 0, 0, 1,
        };
        gradients = std::vector<double>{
            1.0, /**/ 1.0, /**/ 1.0,
            1.0, /**/ 1.0, /**/ 1.0
        };

        // C block
        check_block(descriptor, 1, samples, properties, values, gradient_samples, gradients);

        mts_labels_free(selected_properties);
        mts_tensormap_free(descriptor);
        featomic_calculator_free(calculator);
    }

    SECTION("Partial compute -- preselected") {
        auto sample_names = std::vector<const char*>{
            "system", "atom"
        };
        auto property_names = std::vector<const char*>{
            "index_delta", "x_y_z"
        };

        auto h_sample_values = std::vector<int32_t>{
            0, 3,
        };
        auto h_property_values = std::vector<int32_t>{
            0, 1,
        };

        mts_block_t* blocks[2] = {nullptr, nullptr};

        // mts_block takes ownership of labels
        auto* h_samples = create_labels(sample_names, h_sample_values, 1);
        REQUIRE(h_samples != nullptr);
        auto* h_properties = create_labels(property_names, h_property_values, 1);
        REQUIRE(h_properties != nullptr);

        blocks[0] = mts_block(empty_array({1, 1}), h_samples, nullptr, 0, h_properties);
        REQUIRE(blocks[0] != nullptr);

        auto c_sample_values = std::vector<int32_t>{
            0, 0,
        };
        auto c_property_values = std::vector<int32_t>{
            1, 0,
        };

        auto* c_samples = create_labels(sample_names, c_sample_values, 1);
        REQUIRE(c_samples != nullptr);
        auto* c_properties = create_labels(property_names, c_property_values, 1);
        REQUIRE(c_properties != nullptr);

        blocks[1] = mts_block(empty_array({1, 1}), c_samples, nullptr, 0, c_properties);
        REQUIRE(blocks[1] != nullptr);

        auto keys_names = std::vector<const char*>{"center_type"};
        auto keys_values = std::vector<int32_t>{1, 6};

        // mts_tensormap takes ownership of keys
        auto* tm_keys = create_labels(keys_names, keys_values, 2);
        REQUIRE(tm_keys != nullptr);

        auto* predefined = mts_tensormap(tm_keys, blocks, 2);
        REQUIRE(predefined != nullptr);

        auto system = simple_system();
        featomic_calculation_options_t options = {};
        const char* gradients_list[] = {"positions"};
        options.gradients = gradients_list;
        options.gradients_count = 1;
        options.selected_samples.predefined = predefined;
        options.selected_properties.predefined = predefined;
        auto* calculator = featomic_calculator("dummy_calculator", HYPERS_JSON);
        REQUIRE(calculator != nullptr);

        mts_tensormap_t* descriptor = nullptr;
        auto status = featomic_calculator_compute(
            calculator, &descriptor, &system, 1, options
        );
        CHECK_SUCCESS(status);

        auto* keys = mts_tensormap_keys(descriptor);
        REQUIRE(keys != nullptr);
        auto keys_info = query_labels(keys);

        CHECK(keys_info.size == 1);
        CHECK(keys_info.names[0] == std::string("center_type"));
        CHECK(keys_info.count == 2);
        CHECK(keys_info.values[0] == 1);
        CHECK(keys_info.values[1] == 6);
        mts_labels_free(keys);

        auto samples = std::vector<int32_t>{
            0, 3,
        };
        auto properties = std::vector<int32_t>{
            0, 1,
        };
        auto values = std::vector<double>{
            15,
        };
        auto gradient_samples = std::vector<int32_t>{
            0, 0, 2, /**/ 0, 0, 3,
        };
        auto gradients = std::vector<double>{
            1.0, /**/ 1.0, /**/ 1.0,
            1.0, /**/ 1.0, /**/ 1.0
        };

        // H block
        check_block(descriptor, 0, samples, properties, values, gradient_samples, gradients);

        samples = std::vector<int32_t>{
            0, 0,
        };
        properties = std::vector<int32_t>{
            1, 0,
        };
        values = std::vector<double>{
            4,
        };
        gradient_samples = std::vector<int32_t>{
            0, 0, 0, /**/ 0, 0, 1,
        };
        gradients = std::vector<double>{
            0.0, /**/ 0.0, /**/ 0.0,
            0.0, /**/ 0.0, /**/ 0.0,
        };

        // C block
        check_block(descriptor, 1, samples, properties, values, gradient_samples, gradients);

        mts_tensormap_free(predefined);
        mts_tensormap_free(descriptor);
        featomic_calculator_free(calculator);
    }

    SECTION("Partial compute -- key selection") {
        // check key selection: we add a non-existing key (12) and remove an
        // existing one (1) from the default set of keys. We also put the keys
        // in a different order than what would be the default (6, 12).

        const char* key_names[] = {"center_type"};
        int32_t key_values[] = {12, 6};
        auto* selected_keys = create_labels(
            std::vector<const char*>{key_names[0]},
            std::vector<int32_t>{key_values[0], key_values[1]},
            2
        );
        REQUIRE(selected_keys != nullptr);

        auto system = simple_system();

        featomic_calculation_options_t options = {};
        const char* gradients_list[] = {"positions"};
        options.gradients = gradients_list;
        options.gradients_count = 1;
        options.selected_keys = selected_keys;

        auto* calculator = featomic_calculator("dummy_calculator", HYPERS_JSON);
        REQUIRE(calculator != nullptr);

        mts_tensormap_t* descriptor = nullptr;
        auto status = featomic_calculator_compute(
            calculator, &descriptor, &system, 1, options
        );
        CHECK_SUCCESS(status);

        auto* keys = mts_tensormap_keys(descriptor);
        REQUIRE(keys != nullptr);
        auto keys_info = query_labels(keys);

        CHECK(keys_info.size == 1);
        CHECK(keys_info.names[0] == std::string("center_type"));
        CHECK(keys_info.count == 2);
        CHECK(keys_info.values[0] == 12);
        CHECK(keys_info.values[1] == 6);
        mts_labels_free(keys);

        auto samples = std::vector<int32_t>{};
        auto properties = std::vector<int32_t>{
            1, 0, 0, 1
        };
        auto values = std::vector<double>{};
        auto gradient_samples = std::vector<int32_t>{};
        auto gradients = std::vector<double>{};

        // empty block, center_type=12 is not present in the system
        check_block(descriptor, 0, samples, properties, values, gradient_samples, gradients);

        samples = std::vector<int32_t>{
            0, 0,
        };
        properties = std::vector<int32_t>{
            1, 0, 0, 1
        };
        values = std::vector<double>{
            4, 33
        };
        gradient_samples = std::vector<int32_t>{
            0, 0, 0, /**/ 0, 0, 1,
        };
        gradients = std::vector<double>{
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0
        };

        // C block
        check_block(descriptor, 1, samples, properties, values, gradient_samples, gradients);

        mts_labels_free(selected_keys);
        mts_tensormap_free(descriptor);
        featomic_calculator_free(calculator);
    }
}

void check_block(
    mts_tensormap_t* descriptor,
    size_t block_id,
    const std::vector<int32_t>& samples,
    const std::vector<int32_t>& properties,
    const std::vector<double>& values,
    const std::vector<int32_t>& gradient_samples,
    const std::vector<double>& gradients
) {
    mts_block_t* block = nullptr;

    auto status = mts_tensormap_block_by_id(descriptor, &block, block_id);
    CHECK_SUCCESS(status);

    /**************************************************************************/
    auto* labels = mts_block_labels(block, 0);
    REQUIRE(labels != nullptr);
    auto info = query_labels(labels);

    CHECK(info.size == 2);
    CHECK(info.names[0] == std::string("system"));
    CHECK(info.names[1] == std::string("atom"));
    auto n_samples = info.count;

    CHECK(info.values == samples);
    mts_labels_free(labels);

    /**************************************************************************/
    labels = mts_block_labels(block, 1);
    REQUIRE(labels != nullptr);
    info = query_labels(labels);

    CHECK(info.size == 2);
    CHECK(info.names[0] == std::string("index_delta"));
    CHECK(info.names[1] == std::string("x_y_z"));
    auto n_properties = info.count;

    CHECK(info.values == properties);
    mts_labels_free(labels);

    /**************************************************************************/
    mts_array_t array = {};
    status = mts_block_data(block, &array);
    CHECK_SUCCESS(status);


    const uintptr_t* shape = nullptr;
    uintptr_t shape_count = 0;
    status = array.shape(array.ptr, &shape, &shape_count);
    CHECK_SUCCESS(status);

    double* values_ptr = dlpack_data_ptr(array);
    CHECK(values_ptr != nullptr);

    CHECK(shape_count == 2);
    CHECK(shape[0] == n_samples);
    CHECK(shape[1] == n_properties);

    auto actual_values = std::vector<double>(
        values_ptr, values_ptr + n_samples * n_properties
    );
    CHECK(actual_values == values);

    /**************************************************************************/
    mts_block_t* gradients_block = nullptr;
    status = mts_block_gradient(block, "positions", &gradients_block);
    CHECK_SUCCESS(status);

    labels = mts_block_labels(gradients_block, 0);
    REQUIRE(labels != nullptr);
    info = query_labels(labels);

    CHECK(info.size == 3);
    CHECK(info.names[0] == std::string("sample"));
    CHECK(info.names[1] == std::string("system"));
    CHECK(info.names[2] == std::string("atom"));
    auto n_gradient_samples = info.count;

    CHECK(info.values == gradient_samples);
    mts_labels_free(labels);

    /**************************************************************************/
    status = mts_block_data(gradients_block, &array);
    CHECK_SUCCESS(status);

    status = array.shape(array.ptr, &shape, &shape_count);
    CHECK_SUCCESS(status);
    values_ptr = dlpack_data_ptr(array);
    CHECK(values_ptr != nullptr);

    CHECK(shape_count == 3);
    CHECK(shape[0] == n_gradient_samples);
    CHECK(shape[1] == 3);
    CHECK(shape[2] == n_properties);

    auto actual_gradients = std::vector<double>(
        values_ptr, values_ptr + n_gradient_samples * 3 * n_properties
    );
    CHECK(actual_gradients == gradients);
}
