#include <vector>
#include <string>
#include <cstring>

#include "featomic.h"
#include "metatensor.hpp"
#include "catch.hpp"
#include "helpers.hpp"

static void check_labels(
    const mts_labels_t* labels,
    const std::vector<std::string>& expected_names,
    const std::vector<int32_t>& expected_values
);

static const mts_labels_t* create_labels(
    std::vector<const char*> names,
    std::vector<int32_t> values
);

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


        const mts_labels_t* keys = mts_tensormap_keys(descriptor);
        REQUIRE(keys != nullptr);
        check_labels(keys, {"center_type"}, {{1, 6}});
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
        const mts_labels_t* selected_samples = create_labels(
            {"system", "atom"},
            {0, 1, /**/ 0, 3,}
        );
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

        const mts_labels_t* keys = mts_tensormap_keys(descriptor);
        REQUIRE(keys != nullptr);
        check_labels(keys, {"center_type"}, {{1, 6}});
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

        mts_tensormap_free(descriptor);
        mts_labels_free(selected_samples);
        featomic_calculator_free(calculator);
    }

    SECTION("Partial compute -- features") {
        const mts_labels_t* selected_properties = create_labels(
            {"index_delta", "x_y_z"},
            {0, 1}
        );
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

        const mts_labels_t* keys = mts_tensormap_keys(descriptor);
        REQUIRE(keys != nullptr);
        check_labels(keys, {"center_type"}, {{1, 6}});
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

        mts_tensormap_free(descriptor);
        mts_labels_free(selected_properties);
        featomic_calculator_free(calculator);
    }

    SECTION("Partial compute -- preselected") {
        mts_block_t* blocks[2] = {nullptr, nullptr};

        const mts_labels_t* h_samples = create_labels(
            {"system", "atom"},
            {0, 3}
        );

        const mts_labels_t* h_properties = create_labels(
            {"index_delta", "x_y_z"},
            {0, 1}
        );

        blocks[0] = mts_block(empty_array({1, 1}), h_samples, nullptr, 0, h_properties);
        REQUIRE(blocks[0] != nullptr);


        auto c_sample_values = std::vector<int32_t>{
            0, 0,
        };
        auto c_property_values = std::vector<int32_t>{
            1, 0,
        };

        const mts_labels_t* c_samples = create_labels(
            {"system", "atom"},
            {0, 0}
        );

        const mts_labels_t* c_properties = create_labels(
            {"index_delta", "x_y_z"},
            {1, 0}
        );

        blocks[1] = mts_block(empty_array({1, 1}), c_samples, nullptr, 0, c_properties);
        REQUIRE(blocks[1] != nullptr);

        const mts_labels_t* keys = create_labels(
            {"center_type"},
            {1, 6}
        );
        REQUIRE(keys != nullptr);

        auto* predefined = mts_tensormap(keys, blocks, 2);
        REQUIRE(predefined != nullptr);
        mts_labels_free(keys);

        mts_labels_free(h_samples);
        mts_labels_free(h_properties);
        mts_labels_free(c_samples);
        mts_labels_free(c_properties);

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

        keys = mts_tensormap_keys(descriptor);
        REQUIRE(keys != nullptr);

        check_labels(keys, {"center_type"}, {{1, 6}});
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

        const mts_labels_t* selected_keys = create_labels(
            {"center_type"},
            {12, 6}
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

        const mts_labels_t* keys = mts_tensormap_keys(descriptor);
        REQUIRE(keys != nullptr);

        check_labels(keys, {"center_type"}, {{12, 6}});
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

        mts_tensormap_free(descriptor);
        mts_labels_free(selected_keys);
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
    const mts_labels_t* labels = mts_block_labels(block, 0);
    REQUIRE(labels != nullptr);

    check_labels(labels, {"system", "atom"}, samples);
    mts_labels_free(labels);

    auto n_samples = samples.size() / 2;

    /**************************************************************************/
    labels = mts_block_labels(block, 1);
    REQUIRE(labels != nullptr);

    check_labels(labels, {"index_delta", "x_y_z"}, properties);
    mts_labels_free(labels);

    auto n_properties = properties.size() / 2;

    /**************************************************************************/
    mts_array_t array = {};
    status = mts_block_data(block, &array);
    CHECK_SUCCESS(status);

    const uintptr_t* shape = nullptr;
    uintptr_t shape_count = 0;
    status = array.shape(array.ptr, &shape, &shape_count);
    CHECK_SUCCESS(status);

    CHECK(shape_count == 2);
    CHECK(shape[0] == n_samples);
    CHECK(shape[1] == n_properties);

    DLManagedTensorVersioned* dl_tensor = nullptr;
    status = array.as_dlpack(
        array.ptr,
        &dl_tensor,
        DLDevice { kDLCPU, 0 },
        nullptr,
        DLPackVersion{ DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION }
    );
    CHECK_SUCCESS(status);

    auto* values_ptr = static_cast<double*>(dl_tensor->dl_tensor.data) + dl_tensor->dl_tensor.byte_offset / sizeof(double);
    auto actual_values = std::vector<double>(
        values_ptr, values_ptr + n_samples * n_properties
    );
    CHECK(actual_values == values);

    if (dl_tensor != nullptr && dl_tensor->deleter != nullptr) {
        dl_tensor->deleter(dl_tensor);
    }

    /**************************************************************************/
    mts_block_t* gradients_block = nullptr;
    status = mts_block_gradient(block, "positions", &gradients_block);
    CHECK_SUCCESS(status);

    labels = mts_block_labels(gradients_block, 0);
    REQUIRE(labels != nullptr);
    check_labels(labels, {"sample", "system", "atom"}, gradient_samples);
    mts_labels_free(labels);

    auto n_gradient_samples = gradient_samples.size() / 3;

    /**************************************************************************/
    status = mts_block_data(gradients_block, &array);
    CHECK_SUCCESS(status);

    status = array.shape(array.ptr, &shape, &shape_count);
    CHECK_SUCCESS(status);

    CHECK(shape_count == 3);
    CHECK(shape[0] == n_gradient_samples);
    CHECK(shape[1] == 3);
    CHECK(shape[2] == n_properties);

    dl_tensor = nullptr;
    status = array.as_dlpack(
        array.ptr,
        &dl_tensor,
        DLDevice { kDLCPU, 0 },
        nullptr,
        DLPackVersion{ DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION }
    );
    CHECK_SUCCESS(status);

    values_ptr = static_cast<double*>(dl_tensor->dl_tensor.data) + dl_tensor->dl_tensor.byte_offset / sizeof(double);
    auto actual_gradients = std::vector<double>(
        values_ptr, values_ptr + n_gradient_samples * 3 * n_properties
    );
    CHECK(actual_gradients == gradients);

    if (dl_tensor != nullptr && dl_tensor->deleter != nullptr) {
        dl_tensor->deleter(dl_tensor);
    }
}

void check_labels(
    const mts_labels_t* labels,
    const std::vector<std::string>& expected_names,
    const std::vector<int32_t>& expected_values
) {
    const char* const* names = nullptr;
    size_t size = 0;
    mts_status_t status = mts_labels_dimensions(labels, &names, &size);
    REQUIRE(status == MTS_SUCCESS);

    CHECK(size == expected_names.size());
    for (size_t i = 0; i < size; ++i) {
        CHECK(names[i] == expected_names[i]);
    }

    int32_t const* keys_values = nullptr;
    size_t count = 0;
    status = mts_labels_values_cpu(labels, &keys_values, &count, &size);
    REQUIRE(status == MTS_SUCCESS);

    CHECK(size == expected_names.size());
    CHECK(count == expected_values.size() / size);
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < size; ++j) {
            CHECK(keys_values[i * size + j] == expected_values[i * size + j]);
        }
    }
}

const mts_labels_t* create_labels(
    std::vector<const char*> names,
    std::vector<int32_t> values
) {
    REQUIRE(values.size() % names.size() == 0);

    auto shape = std::vector<uintptr_t>{values.size() / names.size(), names.size()};
    auto array = std::make_unique<metatensor::SimpleDataArray<int32_t>>(
        metatensor::SimpleDataArray<int32_t>(std::move(shape), std::move(values))
    );

    auto mts_array = metatensor::DataArrayBase::to_mts_array(std::move(array));

    return mts_labels(
        names.data(),
        names.size(),
        std::move(mts_array).release()
    );
}
