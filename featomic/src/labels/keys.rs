use std::collections::BTreeSet;

use metatensor::Labels;
use ndarray::Array2;

use crate::{System, Error};


/// Common interface to create a set of metatensor's `TensorMap` keys from systems
pub trait KeysBuilder {
    /// Compute the keys corresponding to these systems
    fn keys(&self, systems: &mut [Box<dyn System>]) -> Result<Labels, Error>;
}

/// Compute a set of keys with a single variable, the central atom type.
pub struct CenterTypesKeys;

impl KeysBuilder for CenterTypesKeys {
    fn keys(&self, systems: &mut [Box<dyn System>]) -> Result<Labels, Error> {
        let mut all_types = BTreeSet::new();
        for system in systems {
            for &atomic_type in system.types()? {
                all_types.insert(atomic_type);
            }
        }

        let values = Array2::from_shape_vec(
            (all_types.len(), 1),
            all_types.into_iter().collect::<Vec<i32>>(),
        ).expect("wrong shape for CenterTypesKeys");

        return Ok(Labels::new_assume_unique(["center_type"], values));
    }
}

/// Compute a set of keys with two variables: the central atom type and a
/// all neighbor atom types within the whole system.
pub struct AllTypesPairsKeys {}

impl KeysBuilder for AllTypesPairsKeys {
    fn keys(&self, systems: &mut [Box<dyn System>]) -> Result<Labels, Error> {

        let mut all_types_pairs = BTreeSet::new();
        for system in systems {
            for &first_type in system.types()? {
                for &second_type in system.types()? {
                    all_types_pairs.insert([first_type, second_type]);
                }
            }
        }

        let values = Array2::from_shape_vec(
            (all_types_pairs.len(), 2),
            all_types_pairs.into_iter().flatten().collect(),
        ).expect("wrong shape for AllTypesPairsKeys");

        return Ok(Labels::new_assume_unique(["center_type", "neighbor_type"], values));
    }
}

/// Compute a set of keys with two variables: the central atom type and a
/// single neighbor atom type within a cutoff around the central atom.
pub struct CenterSingleNeighborsTypesKeys {
    /// Spherical cutoff to use when searching for neighbors around an atom
    pub cutoff: f64,
    /// Should we consider an atom to be it's own neighbor or not?
    pub self_pairs: bool,
}

impl KeysBuilder for CenterSingleNeighborsTypesKeys {
    fn keys(&self, systems: &mut [Box<dyn System>]) -> Result<Labels, Error> {
        assert!(self.cutoff > 0.0 && self.cutoff.is_finite());

        let mut all_types_pairs = BTreeSet::new();
        for system in systems {
            system.compute_neighbors(self.cutoff)?;

            let types = system.types()?;
            for pair in system.pairs()? {
                all_types_pairs.insert([types[pair.first], types[pair.second]]);
                all_types_pairs.insert([types[pair.second], types[pair.first]]);
            }

            if self.self_pairs {
                for &atomic_type in types {
                    all_types_pairs.insert([atomic_type, atomic_type]);
                }
            }
        }

        let values = Array2::from_shape_vec(
            (all_types_pairs.len(), 2),
            all_types_pairs.into_iter().flatten().collect(),
        ).expect("wrong shape for CenterSingleNeighborsTypesKeys");

        return Ok(Labels::new_assume_unique(["center_type", "neighbor_type"], values));
    }
}


/// Compute a set of keys with three variables: the central atom type and two
/// neighbor atom types.
pub struct CenterTwoNeighborsTypesKeys {
    /// Spherical cutoff to use when searching for neighbors around an atom
    pub cutoff: f64,
    /// Should we consider an atom to be it's own neighbor or not?
    pub self_pairs: bool,
    /// Are neighbor atoms keys symmetric with respect to exchange or not?
    pub symmetric: bool,
}

impl KeysBuilder for CenterTwoNeighborsTypesKeys {
    fn keys(&self, systems: &mut [Box<dyn System>]) -> Result<Labels, Error> {
        assert!(self.cutoff > 0.0 && self.cutoff.is_finite());

        let mut keys = BTreeSet::new();
        for system in systems {
            system.compute_neighbors(self.cutoff)?;
            let types = system.types()?;

            for atom in 0..system.size()? {
                let center_type = types[atom];

                // all neighbor types around the current atom
                let mut neighbor_types = BTreeSet::new();
                for pair in system.pairs_containing(atom)? {
                    let neighbor = if pair.first == atom {
                        pair.second
                    } else {
                        debug_assert_eq!(pair.second, atom);
                        pair.first
                    };

                    neighbor_types.insert(types[neighbor]);
                }

                if self.self_pairs {
                    neighbor_types.insert(center_type);
                }

                // create keys
                for &neighbor_1_type in &neighbor_types {
                    for &neighbor_2_type in &neighbor_types {
                        if self.symmetric && neighbor_2_type < neighbor_1_type {
                            continue;
                        }

                        keys.insert([center_type, neighbor_1_type, neighbor_2_type]);
                    }
                }
            }
        }

        let values = Array2::from_shape_vec(
            (keys.len(), 3),
            keys.into_iter().flatten().collect(),
        ).expect("wrong shape for CenterTwoNeighborsTypesKeys");

        return Ok(Labels::new_assume_unique(["center_type", "neighbor_1_type", "neighbor_2_type"], values));
    }
}
