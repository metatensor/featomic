#![allow(dead_code)]
use std::io::Read;
use std::path::Path;

use serde_json::Value;
use ndarray::ArrayD;
use flate2::read::GzDecoder;

use featomic::{SimpleSystem, System, Matrix3, Vector3D};
use featomic::systems::UnitCell;

type HyperParameters = String;

pub fn load_calculator_input(path: impl AsRef<Path>) -> (Vec<Box<dyn System>>, HyperParameters) {
    let json = std::fs::read_to_string(format!("tests/data/generated/{}", path.as_ref().display()))
        .expect("failed to read input file");

    let data: Value = serde_json::from_str(&json).expect("failed to parse JSON");
    let parameters = data["hyperparameters"].to_string();

    let mut systems = Vec::new();
    for system in data["systems"].as_array().expect("systems must be an array") {
        let cell = read_cell(&system["cell"]);
        let mut simple_system = SimpleSystem::new(cell);

        let types = system["types"].as_array().expect("types must be an array");
        let positions = system["positions"].as_array().expect("positions must be an array");

        for (atomic_type, position) in types.iter().zip(positions) {
            let atomic_type = atomic_type.as_i64().expect("atomic_type must be an integer") as i32;
            let position = position.as_array().expect("position must be an array");
            let position = Vector3D::new(
                position[0].as_f64().unwrap(),
                position[1].as_f64().unwrap(),
                position[2].as_f64().unwrap(),
            );

            simple_system.add_atom(atomic_type, position);
        }

        systems.push(Box::new(simple_system) as Box<dyn System>);
    }

    (systems, parameters)
}

fn read_cell(cell: &Value) -> UnitCell {
    let cell = cell.as_array().expect("cell must be an array");
    let matrix = Matrix3::new([
        [cell[0].as_f64().unwrap(), cell[1].as_f64().unwrap(), cell[2].as_f64().unwrap()],
        [cell[3].as_f64().unwrap(), cell[4].as_f64().unwrap(), cell[5].as_f64().unwrap()],
        [cell[6].as_f64().unwrap(), cell[7].as_f64().unwrap(), cell[8].as_f64().unwrap()],
    ]);

    if matrix == Matrix3::zero() {
        UnitCell::infinite()
    } else {
        UnitCell::from(matrix)
    }
}

pub fn load_expected_values(path: impl AsRef<Path>) -> ArrayD<f64> {
    let file = std::fs::File::open(format!("tests/data/generated/{}", path.as_ref().display()))
        .expect("failed to open file");

    let mut reader = GzDecoder::new(file);

    let mut magic = [0u8; 6];
    reader.read_exact(&mut magic).expect("failed to read magic");
    assert_eq!(&magic, b"\x93NUMPY");

    let mut version = [0u8; 2];
    reader.read_exact(&mut version).expect("failed to read version");

    let mut header_len_bytes = [0u8; 2];
    reader.read_exact(&mut header_len_bytes).expect("failed to read header len");
    let header_len = u16::from_le_bytes(header_len_bytes) as usize;

    let mut header_bytes = vec![0u8; header_len];
    reader.read_exact(&mut header_bytes).expect("failed to read header");
    let header = String::from_utf8(header_bytes).expect("failed to parse header as utf8");

    if !header.contains("'descr': '<f8'") {
        panic!("only float64 is supported, got header: {}", header);
    }

    if !header.contains("'fortran_order': False") {
        panic!("only C-order is supported, got header: {}", header);
    }

    let start = header.find("'shape': (").expect("failed to find shape") + 10;
    let end = header[start..].find(")").expect("failed to find end of shape");
    let shape_str = &header[start..start + end];
    let shape: Vec<usize> = shape_str
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().expect("failed to parse shape"))
        .collect();

    let mut data_bytes = Vec::new();
    reader.read_to_end(&mut data_bytes).expect("failed to read data");

    let data: Vec<f64> = data_bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    ArrayD::from_shape_vec(shape, data).expect("failed to create array")
}
