#ifndef FEATOMIC_EXAMPLE_UTILS_H
#define FEATOMIC_EXAMPLE_UTILS_H

#include <metatensor.h>

/// Create an mts_array_t containing a single double value to be used as
/// fill_value
mts_array_t create_fill_value(double value);

/// Create an empty mts_array_t with shape (0, size), to be used as the values
/// for empty labels
mts_array_t create_empty_array(size_t size);

#endif
