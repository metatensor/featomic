/* ============    Automatically generated file, DOT NOT EDIT.    ============ *
 *                                                                             *
 *    This file is automatically generated from the featomic sources,          *
 *    using cbindgen. If you want to make change to this file (including       *
 *    documentation), make the corresponding changes in the rust sources       *
 *    in `featomic/src/c_api/`                                                 *
 * =========================================================================== */

#ifndef FEATOMIC_H
#define FEATOMIC_H

#include <stdbool.h>
#include <stdint.h>
#include <metatensor.h>

/**
 * Status code used when a function succeeded
 */
#define FEATOMIC_SUCCESS 0

/**
 * Status code used when a function got an invalid parameter
 */
#define FEATOMIC_INVALID_PARAMETER_ERROR 1

/**
 * Status code used when there was an error reading or writing JSON
 */
#define FEATOMIC_JSON_ERROR 2

/**
 * Status code used when a string contains non-utf8 data
 */
#define FEATOMIC_UTF8_ERROR 3

/**
 * Status code used for errors coming from the system implementation if we
 * don't have a more specific status
 */
#define FEATOMIC_SYSTEM_ERROR 128

/**
 * Status code used when a memory buffer is too small to fit the requested data
 */
#define FEATOMIC_BUFFER_SIZE_ERROR 254

/**
 * Status code used when there was an internal error, i.e. there is a bug
 * inside featomic
 */
#define FEATOMIC_INTERNAL_ERROR 255

/**
 * The "error" level designates very serious errors
 */
#define FEATOMIC_LOG_LEVEL_ERROR 1

/**
 * The "warn" level designates hazardous situations
 */
#define FEATOMIC_LOG_LEVEL_WARN 2

/**
 * The "info" level designates useful information
 */
#define FEATOMIC_LOG_LEVEL_INFO 3

/**
 * The "debug" level designates lower priority information
 *
 * By default, log messages at this level are disabled in release mode, and
 * enabled in debug mode.
 */
#define FEATOMIC_LOG_LEVEL_DEBUG 4

/**
 * The "trace" level designates very low priority, often extremely verbose,
 * information.
 *
 * By default, featomic disable this level, you can enable it by editing the
 * code.
 */
#define FEATOMIC_LOG_LEVEL_TRACE 5

/**
 * Opaque type representing a `Calculator`
 */
typedef struct featomic_calculator_t featomic_calculator_t;

/**
 * Status type returned by all functions in the C API.
 *
 * The value 0 (`FEATOMIC_SUCCESS`) is used to indicate successful operations.
 * Positive non-zero values are reserved for internal use in featomic.
 * Negative values are reserved for use in user code, in particular to indicate
 * error coming from callbacks.
 */
typedef int32_t featomic_status_t;

/**
 * Callback function type for featomic logging system. Such functions are
 * called when a log event is emitted in the code.
 *
 * The first argument is the log level, one of `FEATOMIC_LOG_LEVEL_ERROR`,
 * `FEATOMIC_LOG_LEVEL_WARN` `FEATOMIC_LOG_LEVEL_INFO`, `FEATOMIC_LOG_LEVEL_DEBUG`,
 * or `FEATOMIC_LOG_LEVEL_TRACE`. The second argument is a NULL-terminated string
 * containing the message associated with the log event.
 */
typedef void (*featomic_logging_callback_t)(int32_t level, const char *message);

/**
 * Pair of atoms coming from a neighbor list
 */
typedef struct featomic_pair_t {
  /**
   * index of the first atom in the pair
   */
  uintptr_t first;
  /**
   * index of the second atom in the pair
   */
  uintptr_t second;
  /**
   * distance between the two atoms
   */
  double distance;
  /**
   * vector from the first atom to the second atom, accounting for periodic
   * boundary conditions. This should be
   * `position[second] - position[first] + H * cell_shift`
   * where `H` is the cell matrix.
   */
  double vector[3];
  /**
   * How many cell shift where applied to the `second` atom to create this
   * pair.
   */
  int32_t cell_shift_indices[3];
} featomic_pair_t;

/**
 * A `featomic_system_t` deals with the storage of atoms and related information,
 * as well as the computation of neighbor lists.
 *
 * This struct contains a manual implementation of a virtual table, allowing to
 * implement the rust `System` trait in C and other languages. Speaking in Rust
 * terms, `user_data` contains a pointer (analog to `Box<Self>`) to the struct
 * implementing the `System` trait; and then there is one function pointers
 * (`Option<unsafe extern "C" fn(XXX)>`) for each function in the `System` trait.
 *
 * The `featomic_status_t` return value for the function is used to communicate
 * error messages. It should be 0/`FEATOMIC_SUCCESS` in case of success, any
 * non-zero value in case of error. The error will be propagated to the
 * top-level caller as a `FEATOMIC_SYSTEM_ERROR`
 *
 * A new implementation of the System trait can then be created in any language
 * supporting a C API (meaning any language for our purposes); by correctly
 * setting `user_data` to the actual data storage, and setting all function
 * pointers to the correct functions. For an example of code doing this, see
 * the `SystemBase` class in the Python interface to featomic.
 *
 * **WARNING**: all function implementations **MUST** be thread-safe, function
 * taking `const` pointer parameters can be called from multiple threads at the
 * same time. The `featomic_system_t` itself might be moved from one thread to
 * another.
 */
typedef struct featomic_system_t {
  /**
   * User-provided data should be stored here, it will be passed as the
   * first parameter to all function pointers below.
   */
  void *user_data;
  /**
   * This function should set `*size` to the number of atoms in this system
   */
  featomic_status_t (*size)(const void *user_data, uintptr_t *size);
  /**
   * This function should set `*types` to a pointer to the first element of
   * a contiguous array containing the atomic types of each atom in the
   * system. Different atomic types should be identified with a different
   * value. These values are usually the atomic number, but don't have to be.
   * The array should contain `featomic_system_t::size()` elements.
   */
  featomic_status_t (*types)(const void *user_data, const int32_t **types);
  /**
   * This function should set `*positions` to a pointer to the first element
   * of a contiguous array containing the atomic cartesian coordinates.
   * `positions[0], positions[1], positions[2]` must contain the x, y, z
   * cartesian coordinates of the first atom, and so on.
   */
  featomic_status_t (*positions)(const void *user_data, const double **positions);
  /**
   * This function should write the unit cell matrix in `cell`, which have
   * space for 9 values. The cell should be written in row major order, i.e.
   * `ax ay az bx by bz cx cy cz`, where a/b/c are the unit cell vectors.
   */
  featomic_status_t (*cell)(const void *user_data, double *cell);
  /**
   * This function should compute the neighbor list with the given cutoff,
   * and store it for later access using `pairs` or `pairs_containing`.
   */
  featomic_status_t (*compute_neighbors)(void *user_data, double cutoff);
  /**
   * This function should set `*pairs` to a pointer to the first element of a
   * contiguous array containing all pairs in this system; and `*count` to
   * the size of the array/the number of pairs.
   *
   * This list of pair should only contain each pair once (and not twice as
   * `i-j` and `j-i`), should not contain self pairs (`i-i`); and should only
   * contains pairs where the distance between atoms is actually bellow the
   * cutoff passed in the last call to `compute_neighbors`. This function is
   * only valid to call after a call to `compute_neighbors`.
   */
  featomic_status_t (*pairs)(const void *user_data,
                             const struct featomic_pair_t **pairs,
                             uintptr_t *count);
  /**
   * This function should set `*pairs` to a pointer to the first element of a
   * contiguous array containing all pairs in this system containing the atom
   * with index `atom`; and `*count` to the size of the array/the number of
   * pairs.
   *
   * The same restrictions on the list of pairs as `featomic_system_t::pairs`
   * applies, with the additional condition that the pair `i-j` should be
   * included both in the return of `pairs_containing(i)` and
   * `pairs_containing(j)`.
   */
  featomic_status_t (*pairs_containing)(const void *user_data,
                                        uintptr_t atom,
                                        const struct featomic_pair_t **pairs,
                                        uintptr_t *count);
} featomic_system_t;

/**
 * Rules to select labels (either samples or properties) on which the user
 * wants to run a calculation
 *
 * To run the calculation for all possible labels, users should set both fields
 * to NULL.
 */
typedef struct featomic_labels_selection_t {
  /**
   * Select a subset of labels, using the same selection criterion for all
   * keys in the final `mts_tensormap_t`.
   *
   * If the `mts_labels_t` instance contains the same variables as the full
   * set of labels, then only entries from the full set that also appear in
   * this selection will be used.
   *
   * If the `mts_labels_t` instance contains a subset of the variables of the
   * full set of labels, then only entries from the full set which match one
   * of the entry in this selection for all of the selection variable will be
   * used.
   */
  const mts_labels_t *subset;
  /**
   * Use a predefined subset of labels, with different entries for different
   * keys of the final `mts_tensormap_t`.
   *
   * For each key, the corresponding labels are fetched out of the
   * `mts_tensormap_t` instance, which must have the same set of keys as the
   * full calculation.
   */
  const mts_tensormap_t *predefined;
} featomic_labels_selection_t;

/**
 * Options that can be set to change how a calculator operates.
 */
typedef struct featomic_calculation_options_t {
  /**
   * @verbatim embed:rst:leading-asterisk
   * Array of NULL-terminated strings containing the gradients to compute.
   * If this field is `NULL` and `gradients_count` is 0, no gradients are
   * computed.
   *
   * The following gradients are available:
   *
   * - ``"positions"``, for gradients of the representation with respect to
   *   atomic positions, with fixed cell matrix parameters. Positions
   *   gradients are computed as
   *
   *   .. math::
   *       \frac{\partial \langle q \vert A_i \rangle}
   *            {\partial \mathbf{r_j}}
   *
   *   where :math:`\langle q \vert A_i \rangle` is the representation around
   *   atom :math:`i` and :math:`\mathbf{r_j}` is the position vector of the
   *   atom :math:`j`.
   *
   *   **Note**: Position gradients of an atom are computed with respect to all
   *   other atoms within the representation. To recover the force one has to
   *   accumulate all pairs associated with atom :math:`i`.
   *
   * - ``"strain"``, for gradients of the representation with respect to
   *   strain. These gradients are typically used to compute the virial, and
   *   from there the pressure acting on a system. To compute them, we
   *   pretend that all the positions :math:`\mathbf r` and unit cell
   *   :math:`\mathbf H` have been scaled by a strain matrix
   *   :math:`\epsilon`:
   *
   *   .. math::
   *      \mathbf r &\rightarrow \mathbf r \left(\mathbb{1} + \epsilon \right)\\
   *      \mathbf H &\rightarrow \mathbf H \left(\mathbb{1} + \epsilon \right)
   *
   *   and then take the gradients of the representation with respect to this
   *   matrix:
   *
   *   .. math::
   *       \frac{\partial \langle q \vert A_i \rangle} {\partial \mathbf{\epsilon}}
   *
   * - ``"cell"``, for gradients of the representation with respect to the
   *   system's cell parameters. These gradients are computed at fixed
   *   positions, and often not what you want when computing gradients
   *   explicitly (they are mainly used in ``featomic.torch`` to integrate
   *   with backward propagation). If you are trying to compute the virial
   *   or the stress, you should use ``"strain"`` gradients instead.
   *
   *   .. math::
   *       \left. \frac{\partial \langle q \vert A_i \rangle}
   *            {\partial \mathbf{H}} \right |_\mathbf{r}
   *
   * @endverbatim
   */
  const char *const *gradients;
  /**
   * Size of the `gradients` array
   */
  uintptr_t gradients_count;
  /**
   * Copy the data from systems into native `SimpleSystem`. This can be
   * faster than having to cross the FFI boundary too often.
   */
  bool use_native_system;
  /**
   * Selection of samples on which to run the computation
   */
  struct featomic_labels_selection_t selected_samples;
  /**
   * Selection of properties to compute for the samples
   */
  struct featomic_labels_selection_t selected_properties;
  /**
   * Selection for the keys to include in the output. Set this parameter to
   * `NULL` to use the default set of keys, as determined by the calculator.
   * Note that this default set of keys can depend on which systems we are
   * running the calculation on.
   */
  const mts_labels_t *selected_keys;
} featomic_calculation_options_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Get the last error message that was created on the current thread.
 *
 * @returns the last error message, as a NULL-terminated string
 */
const char *featomic_last_error(void);

/**
 * Set the given ``callback`` function as the global logging callback. This
 * function will be called on all log events. If a logging callback was already
 * set, it is replaced by the new one.
 */
featomic_status_t featomic_set_logging_callback(featomic_logging_callback_t callback);

/**
 * Create a new calculator with the given `name` and `parameters`.
 *
 * @verbatim embed:rst:leading-asterisk
 *
 * The list of available calculators and the corresponding parameters are in
 * the :ref:`main documentation <userdoc-references>`. The ``parameters`` should
 * be formatted as JSON, according to the requested calculator schema.
 *
 * @endverbatim
 *
 * All memory allocated by this function can be released using
 * `featomic_calculator_free`.
 *
 * @param name name of the calculator as a NULL-terminated string
 * @param parameters hyper-parameters of the calculator, JSON-formatted in a
 *                   NULL-terminated string
 *
 * @returns A pointer to the newly allocated calculator, or a `NULL` pointer in
 *          case of error. In case of error, you can use `featomic_last_error()`
 *          to get the error message.
 */
struct featomic_calculator_t *featomic_calculator(const char *name, const char *parameters);

/**
 * Free the memory associated with a `calculator` previously created with
 * `featomic_calculator`.
 *
 * If `calculator` is `NULL`, this function does nothing.
 *
 * @param calculator pointer to an existing calculator, or `NULL`
 *
 * @returns The status code of this operation. If the status is not
 *          `FEATOMIC_SUCCESS`, you can use `featomic_last_error()` to get the
 *          full error message.
 */
featomic_status_t featomic_calculator_free(struct featomic_calculator_t *calculator);

/**
 * Get a copy of the name of this calculator in the `name` buffer of size
 * `bufflen`.
 *
 * `name` will be NULL-terminated by this function. If the buffer is too small
 * to fit the whole name, this function will return
 * `FEATOMIC_BUFFER_SIZE_ERROR`
 *
 * @param calculator pointer to an existing calculator
 * @param name string buffer to fill with the calculator name
 * @param bufflen number of characters available in the buffer
 *
 * @returns The status code of this operation. If the status is not
 *          `FEATOMIC_SUCCESS`, you can use `featomic_last_error()` to get the full
 *          error message.
 */
featomic_status_t featomic_calculator_name(const struct featomic_calculator_t *calculator,
                                           char *name,
                                           uintptr_t bufflen);

/**
 * Get a copy of the parameters used to create this calculator in the
 * `parameters` buffer of size `bufflen`.
 *
 * `parameters` will be NULL-terminated by this function. If the buffer is too
 * small to fit the whole name, this function will return
 * `FEATOMIC_BUFFER_SIZE_ERROR`.
 *
 * @param calculator pointer to an existing calculator
 * @param parameters string buffer to fill with the parameters used to create
 *                   this calculator
 * @param bufflen number of characters available in the buffer
 *
 * @returns The status code of this operation. If the status is not
 *          `FEATOMIC_SUCCESS`, you can use `featomic_last_error()` to get the full
 *          error message.
 */
featomic_status_t featomic_calculator_parameters(const struct featomic_calculator_t *calculator,
                                                 char *parameters,
                                                 uintptr_t bufflen);

/**
 * Get all radial cutoffs used by this `calculator`'s neighbors lists (which
 * can be an empty list).
 *
 * The `*cutoffs` pointer will be pointing to data inside the `calculator`, and
 * is only valid when the `calculator` itself is.
 *
 * @param calculator pointer to an existing calculator
 * @param cutoffs pointer to be filled with the address of the first element of
 *                an array of cutoffs
 * @param cutoffs_count pointer to be filled with the number of elements in the
 *                      `cutoffs` array
 * @returns The status code of this operation. If the status is not
 *          `FEATOMIC_SUCCESS`, you can use `featomic_last_error()` to get the full
 *          error message.
 */
featomic_status_t featomic_calculator_cutoffs(const struct featomic_calculator_t *calculator,
                                              const double **cutoffs,
                                              uintptr_t *cutoffs_count);

/**
 * Compute the representation of the given list of `systems` with a
 * `calculator`
 *
 * This function allocates a new `mts_tensormap_t` in `*descriptor`, which
 * memory needs to be released by the user with `mts_tensormap_free`.
 *
 * @param calculator pointer to an existing calculator
 * @param descriptor pointer to an `mts_tensormap_t *` that will be allocated
 *                   by this function
 * @param systems pointer to an array of systems implementation
 * @param systems_count number of systems in `systems`
 * @param options options for this calculation
 *
 * @returns The status code of this operation. If the status is not
 *          `FEATOMIC_SUCCESS`, you can use `featomic_last_error()` to get the full
 *          error message.
 */
featomic_status_t featomic_calculator_compute(struct featomic_calculator_t *calculator,
                                              mts_tensormap_t **descriptor,
                                              struct featomic_system_t *systems,
                                              uintptr_t systems_count,
                                              struct featomic_calculation_options_t options);

/**
 * Clear all collected profiling data
 *
 * See also `featomic_profiling_enable` and `featomic_profiling_get`.
 *
 * @returns The status code of this operation. If the status is not
 *          `FEATOMIC_SUCCESS`, you can use `featomic_last_error()` to get the full
 *          error message.
 */
featomic_status_t featomic_profiling_clear(void);

/**
 * Enable or disable profiling data collection. By default, data collection
 * is disabled.
 *
 * Featomic uses the [`time_graph`](https://docs.rs/time-graph/) to collect
 * timing information on the calculations. This profiling code collects the
 * total time spent inside the most important functions, as well as the
 * function call graph (which function called which other function).
 *
 * You can use `featomic_profiling_clear` to reset profiling data to an empty
 * state, and `featomic_profiling_get` to extract the profiling data.
 *
 * @param enabled whether data collection should be enabled or not
 *
 * @returns The status code of this operation. If the status is not
 *          `FEATOMIC_SUCCESS`, you can use `featomic_last_error()` to get the full
 *          error message.
 */
featomic_status_t featomic_profiling_enable(bool enabled);

/**
 * Extract the current set of data collected for profiling.
 *
 * See also `featomic_profiling_enable` and `featomic_profiling_clear`.
 *
 * @param format in which format should the data be provided. `"table"`,
 *              `"short_table"` and `"json"` are currently supported
 * @param buffer pre-allocated buffer in which profiling data will be copied.
 *               If the buffer is too small, this function will return
 *               `FEATOMIC_BUFFER_SIZE_ERROR`
 * @param bufflen size of the `buffer`
 *
 * @returns The status code of this operation. If the status is not
 *          `FEATOMIC_SUCCESS`, you can use `featomic_last_error()` to get the full
 *          error message.
 */
featomic_status_t featomic_profiling_get(const char *format, char *buffer, uintptr_t bufflen);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* FEATOMIC_H */
