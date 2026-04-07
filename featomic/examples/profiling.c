#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include <metatensor.h>
#include <featomic.h>

#include "common/systems.h"

// Minimal scalar fill_value array
static double SCALAR_FILL_DATA_P = 0.0;
static uintptr_t SCALAR_FILL_SHAPE_P = 1;

static mts_status_t fill_shape_p(const void* p, const uintptr_t** shape, uintptr_t* count) {
    (void)p; *shape = &SCALAR_FILL_SHAPE_P; *count = 1; return MTS_SUCCESS;
}
static mts_status_t fill_origin_p(const void* p, mts_data_origin_t* o) {
    (void)p; mts_register_data_origin("c-scalar-fill", o); return MTS_SUCCESS;
}
static mts_status_t fill_device_p(const void* p, DLDevice* d) {
    (void)p; d->device_type = kDLCPU; d->device_id = 0; return MTS_SUCCESS;
}
static mts_status_t fill_dtype_p(const void* p, DLDataType* dt) {
    (void)p; dt->code = kDLFloat; dt->bits = 64; dt->lanes = 1; return MTS_SUCCESS;
}

static mts_array_t scalar_fill_value(void) {
    mts_array_t array;
    memset(&array, 0, sizeof(array));
    array.ptr = &SCALAR_FILL_DATA_P;
    array.shape = fill_shape_p;
    array.origin = fill_origin_p;
    array.device = fill_device_p;
    array.dtype = fill_dtype_p;
    return array;
}

/// Compute SOAP power spectrum, this is the same code as the 'compute-soap'
/// example
static mts_tensormap_t* compute_soap(const char* path);

int main(int argc, char* argv[]) {
    featomic_status_t status = FEATOMIC_SUCCESS;
    char* buffer = NULL;
    size_t buffer_size = 8192;
    bool got_error = true;

    if (argc < 2) {
        printf("error: expected a command line argument");
        goto cleanup;
    }

    // enable collection of profiling data
    status = featomic_profiling_enable(true);
    if (status != FEATOMIC_SUCCESS) {
        printf("Error: %s\n", featomic_last_error());
        goto cleanup;
    }

    // clear any existing collected data
    status = featomic_profiling_clear();
    if (status != FEATOMIC_SUCCESS) {
        printf("Error: %s\n", featomic_last_error());
        goto cleanup;
    }

    mts_tensormap_t* descriptor = compute_soap(argv[1]);
    if (descriptor == NULL) {
        goto cleanup;
    }

    buffer = calloc(buffer_size, sizeof(char));
    if (buffer == NULL) {
        printf("Error: failed to allocate memory\n");
        goto cleanup;
    }

    // Get the profiling data as a table to display it directly
    status = featomic_profiling_get("short_table", buffer, buffer_size);
    if (status != FEATOMIC_SUCCESS) {
        printf("Error: %s\n", featomic_last_error());
        goto cleanup;
    }
    printf("%s\n", buffer);

    // Or save this data as json for future usage
    status = featomic_profiling_get("json", buffer, buffer_size);
    if (status != FEATOMIC_SUCCESS) {
        printf("Error: %s\n", featomic_last_error());
        goto cleanup;
    }
    printf("%s\n", buffer);

    got_error = false;
cleanup:
    free(buffer);
    mts_tensormap_free(descriptor);

    if (got_error) {
        return 1;
    } else {
        return 0;
    }
}

static mts_tensormap_t* move_keys_to_samples(mts_tensormap_t* descriptor, const char* keys_to_move[], size_t keys_to_move_len);
static mts_tensormap_t* move_keys_to_properties(mts_tensormap_t* descriptor, const char* keys_to_move[], size_t keys_to_move_len);

// this is the same function as in the compute-soap.c example
mts_tensormap_t* compute_soap(const char* path) {
    int status = FEATOMIC_SUCCESS;
    featomic_calculator_t* calculator = NULL;
    featomic_system_t* systems = NULL;
    uintptr_t n_systems = 0;
    const double* values = NULL;
    const uintptr_t* shape = NULL;
    uintptr_t shape_count = 0;
    bool got_error = true;
    const char* keys_to_samples[] = {"center_type"};
    const char* keys_to_properties[] = {"neighbor_1_type", "neighbor_2_type"};

    // use the default set of options, computing all samples and all features
    featomic_calculation_options_t options = {0};
    const char* gradients_list[] = {"positions"};
    options.gradients = gradients_list;
    options.gradients_count = 1;
    options.use_native_system = true;

    mts_tensormap_t* descriptor = NULL;
    const mts_block_t* block = NULL;
    mts_array_t data = {0};
    (void)0; // removed unused mts_labels_t keys_to_move

    const char* parameters = "{\n"
        "\"cutoff\": {\n"
        "    \"radius\": 5.0,\n"
        "    \"smoothing\": {\"type\": \"ShiftedCosine\", \"width\": 0.5}\n"
        "},\n"
        "\"density\": {\n"
        "    \"type\": \"Gaussian\",\n"
        "    \"width\": 0.3\n"
        "},\n"
        "\"basis\": {\n"
        "    \"type\": \"TensorProduct\",\n"
        "    \"max_angular\": 6,\n"
        "    \"radial\": {\"type\": \"Gto\", \"max_radial\": 6}\n"
        "}\n"
    "}";


    status = read_systems_example(path, &systems, &n_systems);
    if (status != FEATOMIC_SUCCESS) {
        printf("Error: %s\n", featomic_last_error());
        goto cleanup;
    }

    calculator = featomic_calculator("soap_power_spectrum", parameters);
    if (calculator == NULL) {
        printf("Error: %s\n", featomic_last_error());
        goto cleanup;
    }

    status = featomic_calculator_compute(
        calculator, &descriptor, systems, n_systems, options
    );
    if (status != FEATOMIC_SUCCESS) {
        printf("Error: %s\n", featomic_last_error());
        goto cleanup;
    }

    descriptor = move_keys_to_samples(descriptor, keys_to_samples, 1);
    if (descriptor == NULL) {
        printf("Error: %s\n", mts_last_error());
        goto cleanup;
    }

    descriptor = move_keys_to_properties(descriptor, keys_to_properties, 2);
    if (descriptor == NULL) {
        printf("Error: %s\n", mts_last_error());
        goto cleanup;
    }

cleanup:
    featomic_calculator_free(calculator);
    free_systems_example(systems, n_systems);

    return descriptor;
}


mts_tensormap_t* move_keys_to_samples(mts_tensormap_t* descriptor, const char* keys_to_move[], size_t keys_to_move_len) {
    mts_tensormap_t* moved_descriptor = NULL;

    mts_array_t empty_values = {0};
    mts_labels_t* keys = mts_labels_create(keys_to_move, keys_to_move_len, empty_values);
    if (keys == NULL) {
        mts_tensormap_free(descriptor);
        return NULL;
    }

    mts_array_t fill_value = scalar_fill_value();
    moved_descriptor = mts_tensormap_keys_to_samples(descriptor, keys, fill_value, true);
    mts_labels_free(keys);
    mts_tensormap_free(descriptor);

    return moved_descriptor;
}


mts_tensormap_t* move_keys_to_properties(mts_tensormap_t* descriptor, const char* keys_to_move[], size_t keys_to_move_len) {
    mts_tensormap_t* moved_descriptor = NULL;

    mts_array_t empty_values = {0};
    mts_labels_t* keys = mts_labels_create(keys_to_move, keys_to_move_len, empty_values);
    if (keys == NULL) {
        mts_tensormap_free(descriptor);
        return NULL;
    }

    mts_array_t fill_value = scalar_fill_value();
    moved_descriptor = mts_tensormap_keys_to_properties(descriptor, keys, fill_value, true);
    mts_labels_free(keys);
    mts_tensormap_free(descriptor);

    return moved_descriptor;
}


#include "common/systems.c"
