#![allow(clippy::needless_return)]

use rascaline::calculators::CalculatorBase;
use rascaline::calculators::{SoapPowerSpectrum, PowerSpectrumParameters};
use rascaline::calculators::soap::{RadialBasis, CutoffFunction};

use rascaline::{Descriptor, System};

use criterion::{BenchmarkGroup, Criterion, measurement::WallTime, SamplingMode};
use criterion::{black_box, criterion_group, criterion_main};


fn load_systems(path: &str) -> Vec<Box<dyn System>> {
    let systems = rascaline::systems::read_from_file(&format!("benches/data/{}", path))
        .expect("failed to read file");

    return systems.into_iter()
        .map(|s| Box::new(s) as Box<dyn System>)
        .collect()
}

fn run_soap_power_spectrum(mut group: BenchmarkGroup<WallTime>, path: &str, gradients: bool) {
    let mut systems = load_systems(path);

    let cutoff = 4.0;
    let mut n_centers = 0;
    for system in &mut systems {
        n_centers += system.size().unwrap();
        system.compute_neighbors(cutoff).unwrap();
    }

    for &max_radial in black_box(&[2, 8, 14]) {
        for &max_angular in black_box(&[1, 7, 15]) {
            let parameters = PowerSpectrumParameters {
                max_radial,
                max_angular,
                cutoff,
                gradients,
                atomic_gaussian_width: 0.3,
                radial_basis: RadialBasis::Gto {},
                cutoff_function: CutoffFunction::ShiftedCosine{ width: 0.5 },
            };
            let mut calculator = SoapPowerSpectrum::new(parameters).unwrap();

            let mut descriptor = Descriptor::new();
            if gradients {
                let (samples, gradients) = calculator.samples().with_gradients(&mut systems).unwrap();
                descriptor.prepare_gradients(samples, gradients.unwrap(), calculator.features());
            } else {
                let samples = calculator.samples().indexes(&mut systems).unwrap();
                descriptor.prepare(samples, calculator.features());
            }

            group.bench_function(&format!("n_max = {}, l_max = {}", max_radial, max_angular), |b| b.iter_custom(|repeat| {
                let start = std::time::Instant::now();
                for _ in 0..repeat {
                    calculator.compute(&mut systems, &mut descriptor).unwrap();
                }
                start.elapsed() / n_centers as u32
            }));
        }
    }
}

fn soap_power_spectrum(c: &mut Criterion) {
    let mut group = c.benchmark_group("SOAP power spectrum (per atom)/Bulk Silicon");
    group.noise_threshold(0.05);
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    run_soap_power_spectrum(group, "silicon_bulk.xyz", false);

    let mut group = c.benchmark_group("SOAP power spectrum (per atom) with gradients/Bulk Silicon");
    group.noise_threshold(0.05);
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    run_soap_power_spectrum(group, "silicon_bulk.xyz", true);

    let mut group = c.benchmark_group("SOAP power spectrum (per atom)/Molecular crystals");
    group.noise_threshold(0.05);
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    run_soap_power_spectrum(group, "molecular_crystals.xyz", false);

    let mut group = c.benchmark_group("SOAP power spectrum (per atom) with gradients/Molecular crystals");
    group.noise_threshold(0.05);
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    run_soap_power_spectrum(group, "molecular_crystals.xyz", true);
}


criterion_group!(all, soap_power_spectrum);
criterion_main!(all);
