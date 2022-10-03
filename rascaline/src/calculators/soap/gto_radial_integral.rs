use std::f64;

use ndarray::{Array2, ArrayViewMut2};

use crate::math::{gamma, DoubleRegularized1F1};
use crate::Error;

use crate::calculators::radial_integral::RadialIntegral;
use crate::calculators::radial_integral::GtoRadialBasis;

const SQRT_PI_OVER_4: f64 = 0.44311346272637897;

/// Parameters controlling the radial integral with GTO radial basis
#[derive(Debug, Clone, Copy)]
pub struct GtoParameters {
    /// Number of radial components
    pub max_radial: usize,
    /// Number of angular components
    pub max_angular: usize,
    /// atomic density gaussian width
    pub atomic_gaussian_width: f64,
    /// cutoff radius
    pub cutoff: f64,
}

impl GtoParameters {
    pub(crate) fn validate(&self) -> Result<(), Error> {
        if self.max_radial == 0 {
            return Err(Error::InvalidParameter(
                "max_radial must be at least 1 for GTO radial integral".into()
            ));
        }

        if self.cutoff < 0.0 || !self.cutoff.is_finite() {
            return Err(Error::InvalidParameter(
                "cutoff must be a positive number for GTO radial integral".into()
            ));
        }

        if self.atomic_gaussian_width < 0.0 || !self.atomic_gaussian_width.is_finite() {
            return Err(Error::InvalidParameter(
                "atomic_gaussian_width must be a positive number for GTO radial integral".into()
            ));
        }

        Ok(())
    }
}

/// Implementation of the SOAP radial integral for GTO radial basis and gaussian
/// atomic density.
#[derive(Debug, Clone)]
pub struct SoapGtoRadialIntegral {
    parameters: GtoParameters,
    /// σ^2, with σ the atomic density gaussian width
    atomic_gaussian_width_2: f64,
    /// 1/2σ^2, with σ the atomic density gaussian width
    atomic_gaussian_constant: f64,
    /// 1/2σ_n^2, with σ_n the GTO gaussian width, i.e. `cutoff * max(√n, 1) / n_max`
    gto_gaussian_constants: Vec<f64>,
    /// `n_max * n_max` matrix to orthonormalize the GTO
    gto_orthonormalization: Array2<f64>,
    /// Implementation of `Gamma(a) / Gamma(b) 1F1(a, b, z)`
    double_regularized_1f1: DoubleRegularized1F1,
}

impl SoapGtoRadialIntegral {
    pub fn new(parameters: GtoParameters) -> Result<SoapGtoRadialIntegral, Error> {
        parameters.validate()?;

        let atomic_gaussian_width_2 = parameters.atomic_gaussian_width * parameters.atomic_gaussian_width;
        let atomic_gaussian_constant = 1.0 / (2.0 * atomic_gaussian_width_2);

        let gto_gaussian_width = GtoRadialBasis::gaussian_widths(parameters.max_radial, parameters.cutoff);
        let gto_gaussian_constants = gto_gaussian_width.into_iter()
            .map(|sigma| 1.0 / (2.0 * sigma * sigma))
            .collect::<Vec<_>>();

        let gto_orthonormalization = GtoRadialBasis::orthonormalization_matrix(
            parameters.max_radial, parameters.cutoff
        );

        return Ok(SoapGtoRadialIntegral {
            parameters: parameters,
            double_regularized_1f1: DoubleRegularized1F1 {
                max_angular: parameters.max_angular,
            },
            atomic_gaussian_width_2: atomic_gaussian_width_2,
            atomic_gaussian_constant: atomic_gaussian_constant,
            gto_gaussian_constants: gto_gaussian_constants,
            gto_orthonormalization: gto_orthonormalization,
        })
    }
}

impl RadialIntegral for SoapGtoRadialIntegral {
    #[time_graph::instrument(name = "SoapGtoRadialIntegral::compute")]
    fn compute(
        &self,
        distance: f64,
        mut values: ArrayViewMut2<f64>,
        mut gradients: Option<ArrayViewMut2<f64>>
    ) {
        let expected_shape = [self.parameters.max_angular + 1, self.parameters.max_radial];
        assert_eq!(
            values.shape(), expected_shape,
            "wrong size for values array, expected [{}, {}] but got [{}, {}]",
            expected_shape[0], expected_shape[1], values.shape()[0], values.shape()[1]
        );

        if let Some(ref gradients) = gradients {
            assert_eq!(
                gradients.shape(), expected_shape,
                "wrong size for gradients array, expected [{}, {}] but got [{}, {}]",
                expected_shape[0], expected_shape[1], gradients.shape()[0], gradients.shape()[1]
            );
        }

        // Define global factor of radial integral arising from two parts:
        // - a global factor of sqrt(pi)/4 from the integral itself
        // - the normalization constant of the atomic Gaussian density.
        //   We use a factor of 1/(pi*sigma^2)^0.75 which leads to
        //   Gaussian densities that are normalized in the L2-sense, i.e.
        //   integral_{R^3} |g(r)|^2 d^3r = 1.
        let atomic_gaussian_normalization = (std::f64::consts::PI * self.atomic_gaussian_width_2).powf(-0.75);
        let global_factor = SQRT_PI_OVER_4 * atomic_gaussian_normalization;

        let c = self.atomic_gaussian_constant;
        let c_rij = c * distance;
        let exp_c_rij = f64::exp(-distance * c_rij);

        for n in 0..self.parameters.max_radial {
            let gto_constant = self.gto_gaussian_constants[n];
            // `global_factor * exp(-c rij^2) * (c * rij)^l`
            let mut factor = global_factor * exp_c_rij;

            let z = c_rij * c_rij / (self.atomic_gaussian_constant + gto_constant);
            self.double_regularized_1f1.compute(
                z, n,
                values.index_axis_mut(ndarray::Axis(1), n),
                gradients.as_mut().map(|g| g.index_axis_mut(ndarray::Axis(1), n))
            );

            for l in 0..(self.parameters.max_angular + 1) {
                let n_l_3_over_2 = 0.5 * (n + l) as f64 + 1.5;
                let c_dn = (c + gto_constant).powf(-n_l_3_over_2);

                values[[l, n]] *= c_dn * factor;
                if let Some(ref mut gradients) = gradients {
                    gradients[[l, n]] *= c_dn * factor * 2.0 * z / distance;
                    gradients[[l, n]] += values[[l, n]] * (l as f64 / distance - 2.0 * c_rij);
                }

                factor *= c_rij;
            }
        }

        // for r = 0, the formula used in the calculations above yield NaN,
        // which in turns breaks the SplinedGto radial integral. From the
        // analytical formula, the gradient is 0 everywhere expect for l=1
        if distance == 0.0 {
            if let Some(ref mut gradients) = gradients {
                gradients.fill(0.0);

                if self.parameters.max_angular >= 1 {
                    let l = 1;
                    for n in 0..self.parameters.max_radial {
                        let gto_constant = self.gto_gaussian_constants[n];
                        let a = 0.5 * (n + l) as f64 + 1.5;
                        let b = 2.5;
                        let c_dn = (c + gto_constant).powf(-a);
                        let factor = global_factor * c * c_dn;

                        gradients[[l, n]] = gamma(a) / gamma(b) * factor;
                    }
                }
            }
        }

        values.assign(&values.dot(&self.gto_orthonormalization));
        if let Some(ref mut gradients) = gradients {
            gradients.assign(&gradients.dot(&self.gto_orthonormalization));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calculators::radial_integral::RadialIntegral;

    use ndarray::Array2;
    use approx::assert_relative_eq;

    #[test]
    #[should_panic = "max_radial must be at least 1"]
    fn invalid_max_radial() {
        SoapGtoRadialIntegral::new(GtoParameters {
            max_radial: 0,
            max_angular: 4,
            cutoff: 3.0,
            atomic_gaussian_width: 0.5
        }).unwrap();
    }

    #[test]
    #[should_panic = "cutoff must be a positive number"]
    fn negative_cutoff() {
        SoapGtoRadialIntegral::new(GtoParameters {
            max_radial: 10,
            max_angular: 4,
            cutoff: -3.0,
            atomic_gaussian_width: 0.5
        }).unwrap();
    }

    #[test]
    #[should_panic = "cutoff must be a positive number"]
    fn infinite_cutoff() {
        SoapGtoRadialIntegral::new(GtoParameters {
            max_radial: 10,
            max_angular: 4,
            cutoff: std::f64::INFINITY,
            atomic_gaussian_width: 0.5
        }).unwrap();
    }

    #[test]
    #[should_panic = "atomic_gaussian_width must be a positive number"]
    fn negative_atomic_gaussian_width() {
        SoapGtoRadialIntegral::new(GtoParameters {
            max_radial: 10,
            max_angular: 4,
            cutoff: 3.0,
            atomic_gaussian_width: -0.5
        }).unwrap();
    }

    #[test]
    #[should_panic = "atomic_gaussian_width must be a positive number"]
    fn infinite_atomic_gaussian_width() {
        SoapGtoRadialIntegral::new(GtoParameters {
            max_radial: 10,
            max_angular: 4,
            cutoff: 3.0,
            atomic_gaussian_width: std::f64::INFINITY,
        }).unwrap();
    }

    #[test]
    #[should_panic = "radial overlap matrix is singular, try with a lower max_radial (current value is 30)"]
    fn ill_conditioned_orthonormalization() {
        SoapGtoRadialIntegral::new(GtoParameters {
            max_radial: 30,
            max_angular: 3,
            cutoff: 5.0,
            atomic_gaussian_width: 0.5,
        }).unwrap();
    }

    #[test]
    #[should_panic = "wrong size for values array, expected [4, 2] but got [3, 2]"]
    fn values_array_size() {
        let gto = SoapGtoRadialIntegral::new(GtoParameters {
            max_radial: 2,
            max_angular: 3,
            cutoff: 5.0,
            atomic_gaussian_width: 0.5,
        }).unwrap();
        let mut values = Array2::from_elem((3, 2), 0.0);

        gto.compute(1.0, values.view_mut(), None);
    }

    #[test]
    #[should_panic = "wrong size for gradients array, expected [4, 2] but got [3, 2]"]
    fn gradient_array_size() {
        let gto = SoapGtoRadialIntegral::new(GtoParameters {
            max_radial: 2,
            max_angular: 3,
            cutoff: 5.0,
            atomic_gaussian_width: 0.5,
        }).unwrap();
        let mut values = Array2::from_elem((4, 2), 0.0);
        let mut gradients = Array2::from_elem((3, 2), 0.0);

        gto.compute(1.0, values.view_mut(), Some(gradients.view_mut()));
    }

    #[test]
    fn gradients_near_zero() {
        let max_radial = 8;
        let max_angular = 8;
        let gto = SoapGtoRadialIntegral::new(GtoParameters {
            max_radial: max_radial,
            max_angular: max_angular,
            cutoff: 5.0,
            atomic_gaussian_width: 0.5,
        }).unwrap();

        let shape = (max_angular + 1, max_radial);
        let mut values = Array2::from_elem(shape, 0.0);
        let mut gradients = Array2::from_elem(shape, 0.0);
        let mut gradients_plus = Array2::from_elem(shape, 0.0);
        gto.compute(0.0, values.view_mut(), Some(gradients.view_mut()));
        gto.compute(1e-12, values.view_mut(), Some(gradients_plus.view_mut()));

        assert_relative_eq!(
            gradients, gradients_plus, epsilon=1e-12, max_relative=1e-6,
        );
    }

    #[test]
    fn finite_differences() {
        let max_radial = 8;
        let max_angular = 8;
        let gto = SoapGtoRadialIntegral::new(GtoParameters {
            max_radial: max_radial,
            max_angular: max_angular,
            cutoff: 5.0,
            atomic_gaussian_width: 0.5,
        }).unwrap();

        let rij = 3.4;
        let delta = 1e-9;

        let shape = (max_angular + 1, max_radial);
        let mut values = Array2::from_elem(shape, 0.0);
        let mut values_delta = Array2::from_elem(shape, 0.0);
        let mut gradients = Array2::from_elem(shape, 0.0);
        gto.compute(rij, values.view_mut(), Some(gradients.view_mut()));
        gto.compute(rij + delta, values_delta.view_mut(), None);

        let finite_differences = (&values_delta - &values) / delta;

        assert_relative_eq!(
            finite_differences, gradients, max_relative=1e-4
        );
    }
}