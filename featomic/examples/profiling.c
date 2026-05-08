#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

#include <metatensor.h>
#include <featomic.h>

#include "common/systems.h"
#include "common/utils.h"

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

static const char* get_mts_last_error() {
    const char* error = NULL;
    mts_status_t status = mts_last_error(&error, NULL, NULL);

    if (status != MTS_SUCCESS) {
        return "Unknown error";
    } else {
        return error;
    }
}

// this is the same function as in the compute-soap.c example
mts_tensormap_t* compute_soap(const char* path) {
    int status = FEATOMIC_SUCCESS;
    featomic_calculator_t* calculator = NULL;
    featomic_system_t* systems = NULL;
    uintptr_t n_systems = 0;
    const uintptr_t* shape = NULL;
    uintptr_t shape_count = 0;
    bool got_error = true;
    const char* keys_to_samples[] = {"center_type"};
    const char* keys_to_properties[] = {"neighbor_1_type", "neighbor_2_type"};
    // use the default set of options, computing all samples and all features,
    // and including gradients with respect to positions
    featomic_calculation_options_t options = {0};
    const char* gradients_list[] = {"positions"};
    options.gradients = gradients_list;
    options.gradients_count = 1;
    options.use_native_system = true;

    mts_tensormap_t* descriptor = NULL;
    mts_block_t* block = NULL;
    mts_array_t array = {0};

    DLManagedTensorVersioned* dlpack_tensor = NULL;
    DLDevice dl_device = {kDLCPU, 0};
    DLPackVersion dl_version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};

    // hyper-parameters for the calculation as JSON
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

    // load systems from command line arguments
    status = read_systems_example(path, &systems, &n_systems);
    if (status != 0) {
        goto cleanup;
    }

    // create the calculator with its name and parameters
    calculator = featomic_calculator("soap_power_spectrum", parameters);
    if (calculator == NULL) {
        printf("Error: %s\n", featomic_last_error());
        goto cleanup;
    }

    // run the calculation
    status = featomic_calculator_compute(
        calculator, &descriptor, systems, n_systems, options
    );
    if (status != FEATOMIC_SUCCESS) {
        printf("Error: %s\n", featomic_last_error());
        goto cleanup;
    }

    // The descriptor is a metatensor `TensorMap`, containing multiple blocks.
    // We can transform it to a single block containing a dense representation,
    // with one sample for each atom-centered environment.
    descriptor = move_keys_to_samples(descriptor, keys_to_samples, 1);
    if (descriptor == NULL) {
        printf("Error: %s\n", get_mts_last_error());
        goto cleanup;
    }

    descriptor = move_keys_to_properties(descriptor, keys_to_properties, 2);
    if (descriptor == NULL) {
        printf("Error: %s\n", get_mts_last_error());
        goto cleanup;
    }

    // extract the unique block and corresponding values from the descriptor
    status = mts_tensormap_block_by_id(descriptor, &block, 0);
    if (status != MTS_SUCCESS) {
        printf("Error: %s\n", get_mts_last_error());
        goto cleanup;
    }

    status = mts_block_data(block, &array);
    if (status != MTS_SUCCESS) {
        printf("Error: %s\n", get_mts_last_error());
        goto cleanup;
    }

    // Call the function to get the data as a dlpack array.
    status = array.as_dlpack(array.ptr, &dlpack_tensor, dl_device, NULL, dl_version);
    if (status != MTS_SUCCESS) {
        printf("Error: %s\n", get_mts_last_error());
        goto cleanup;
    }

    assert(dlpack_tensor->dl_tensor.ndim == 2);
    // you can now use `dlpack_tensor` as the input of a machine learning algorithm
    printf("the value array shape is %lld x %lld\n", dlpack_tensor->dl_tensor.shape[0], dlpack_tensor->dl_tensor.shape[1]);

    got_error = false;
cleanup:
    featomic_calculator_free(calculator);

    free_systems_example(systems, n_systems);
    if (dlpack_tensor != NULL && dlpack_tensor->deleter != NULL) {
        dlpack_tensor->deleter(dlpack_tensor);
    }

    if (got_error) {
        return NULL;
    } else {
        return descriptor;
    }
}

mts_tensormap_t* move_keys_to_samples(mts_tensormap_t* descriptor, const char* keys_to_move[], size_t keys_to_move_len) {
    const mts_labels_t* keys = NULL;
    mts_tensormap_t* moved_descriptor = NULL;
    mts_array_t fill_value;
    mts_array_t empty_values;

    empty_values = create_empty_array(keys_to_move_len);
    keys = mts_labels(keys_to_move, keys_to_move_len, empty_values);

    fill_value = create_fill_value(0.0);

    moved_descriptor = mts_tensormap_keys_to_samples(descriptor, keys, fill_value, true);
    mts_tensormap_free(descriptor);
    mts_labels_free(keys);

    return moved_descriptor;
}


mts_tensormap_t* move_keys_to_properties(mts_tensormap_t* descriptor, const char* keys_to_move[], size_t keys_to_move_len) {
    const mts_labels_t* keys = NULL;
    mts_tensormap_t* moved_descriptor = NULL;
    mts_array_t fill_value;
    mts_array_t empty_values;

    empty_values = create_empty_array(keys_to_move_len);
    keys = mts_labels(keys_to_move, keys_to_move_len, empty_values);

    fill_value = create_fill_value(0.0);

    moved_descriptor = mts_tensormap_keys_to_properties(descriptor, keys, fill_value, true);
    mts_tensormap_free(descriptor);
    mts_labels_free(keys);

    return moved_descriptor;
}


#include "common/systems.c"
#include "common/utils.c"
