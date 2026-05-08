#include <assert.h>
#include <string.h>
#include <stdlib.h>

#include "utils.h"

static void array_destroy(void* user_data) {
    free(user_data);
}

static mts_status_t array_device(const void* user_data, DLDevice* device) {
    device->device_type = kDLCPU;
    device->device_id = 0;
    return MTS_SUCCESS;
}

static mts_status_t array_dtype(const void* user_data, DLDataType* dtype) {
    dtype->code = kDLFloat;
    dtype->bits = 64;
    dtype->lanes = 1;
    return MTS_SUCCESS;
}

static void fill_value_dlpack_deleter(DLManagedTensorVersioned* dlpack) {
    free(dlpack);
}

static mts_status_t fill_value_as_dlpack(
    void* user_data,
    DLManagedTensorVersioned** dlpack,
    DLDevice device,
    const int64_t* stream,
    DLPackVersion max_version
) {
    assert(device.device_type == kDLCPU);
    assert(device.device_id == 0);

    *dlpack = malloc(sizeof(DLManagedTensorVersioned));
    memset(*dlpack, 0, sizeof(DLManagedTensorVersioned));

    (*dlpack)->dl_tensor.data = user_data;
    (*dlpack)->dl_tensor.ndim = 0;
    (*dlpack)->dl_tensor.shape = NULL;
    (*dlpack)->dl_tensor.strides = NULL;
    (*dlpack)->dl_tensor.byte_offset = 0;
    (*dlpack)->dl_tensor.dtype.code = kDLFloat;
    (*dlpack)->dl_tensor.dtype.bits = 64;
    (*dlpack)->dl_tensor.dtype.lanes = 1;
    (*dlpack)->dl_tensor.device = device;

    (*dlpack)->version = max_version;
    (*dlpack)->flags = 0;
    (*dlpack)->manager_ctx = NULL;
    (*dlpack)->deleter = fill_value_dlpack_deleter;


    return MTS_SUCCESS;
}

static mts_status_t fill_value_shape(const void* user_data, const uintptr_t** shape, uintptr_t* shape_count) {
    *shape = NULL;
    *shape_count = 0;
    return MTS_SUCCESS;
}

mts_array_t create_fill_value(double value) {
    double* user_data = malloc(sizeof(double));
    *user_data = value;

    // minimal definition of mts_array_t interface only useful for fill_value
    mts_array_t array = {
        .ptr = user_data,
        .device = array_device,
        .dtype = array_dtype,
        .destroy = array_destroy,
        .as_dlpack = fill_value_as_dlpack,
        .shape = fill_value_shape,
        .origin = NULL,
        .reshape = NULL,
        .swap_axes = NULL,
        .move_data = NULL,
        .create = NULL,
    };

    return array;
}

static mts_status_t empty_shape(const void* user_data, const uintptr_t** shape, uintptr_t* shape_count) {
    *shape = user_data;
    *shape_count = 2;
    return MTS_SUCCESS;
}

mts_array_t create_empty_array(size_t size) {
    size_t* user_data = malloc(2 * sizeof(size_t));
    user_data[0] = 0;
    user_data[1] = size;

    // minimal definition of mts_array_t interface only useful for fill_value
    mts_array_t array = {
        .ptr = user_data,
        .device = array_device,
        .dtype = array_dtype,
        .destroy = array_destroy,
        .as_dlpack = NULL,
        .shape = empty_shape,
        .origin = NULL,
        .reshape = NULL,
        .swap_axes = NULL,
        .move_data = NULL,
        .create = NULL,
    };

    return array;
}
