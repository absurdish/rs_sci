use crate::units::*;

#[derive(Debug, Clone)]
pub struct Const;

impl Const {
    // mathematical constants
    pub const PI: f64 = std::f64::consts::PI;
    pub const E: f64 = std::f64::consts::E;
    pub const GOLDEN_RATIO: f64 = 1.618033988749895;
    pub const SQRT_2: f64 = 1.4142135623730951;
    pub const LN_2: f64 = 0.6931471805599453;
    pub const LN_10: f64 = 2.302585092994046;
    pub const EULER_MASCHERONI: f64 = 0.5772156649015329;

    // physical constants
    pub const SPEED_OF_LIGHT: f64 = 299_792_458.0; // m/s
    pub const GRAVITATIONAL_CONSTANT: f64 = 6.674_30e-11; // m^3/kg·s^2
    pub const PLANCK_CONSTANT: f64 = 6.626_070_15e-34; // J·s
    pub const REDUCED_PLANCK_CONSTANT: f64 = Self::PLANCK_CONSTANT / (2.0 * Self::PI); // J·s
    pub const BOLTZMANN_CONSTANT: f64 = 1.380_649e-23; // J/K
    pub const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19; // C
    pub const ELECTRON_MASS: f64 = 9.109_383_701_5e-31; // kg
    pub const PROTON_MASS: f64 = 1.672_621_923_69e-27; // kg
    pub const NEUTRON_MASS: f64 = 1.674_927_471_4e-27; // kg
    pub const AVOGADRO_CONSTANT: f64 = 6.022_140_76e23; // mol^-1
    pub const VACUUM_PERMITTIVITY: f64 = 8.854_187_812_8e-12; // F/m
    pub const VACUUM_PERMEABILITY: f64 = 1.256_637_062_12e-6; // H/m
    pub const FINE_STRUCTURE_CONSTANT: f64 = 7.297_352_5693e-3; // dimensionless
    pub const RYDBERG_CONSTANT: f64 = 10_973_731.568_160; // m^-1
    pub const ATOMIC_MASS_UNIT: f64 = 1.660_539_066_60e-27; // kg
    pub const FARADAY_CONSTANT: f64 = 96_485.332_123_1; // C/mol
    pub const GAS_CONSTANT: f64 = 8.314_462_618_153_24; // J/(mol·K)
    pub const STEFAN_BOLTZMANN_CONSTANT: f64 = 5.670_374_419e-8; // W/(m^2·K^4)
    pub const WIEN_DISPLACEMENT_CONSTANT: f64 = 2.897_771_955e-3; // m·K

    // astronomical constants
    pub const ASTRONOMICAL_UNIT: f64 = 149_597_870_700.0; // m
    pub const LIGHT_YEAR: f64 = 9.460_730_472_580_8e15; // m
    pub const PARSEC: f64 = 3.085_677_581_491_367e16; // m
    pub const SOLAR_MASS: f64 = 1.988_847e30; // kg
    pub const EARTH_MASS: f64 = 5.972_167e24; // kg
    pub const EARTH_RADIUS: f64 = 6.371e6; // m
    pub const SOLAR_RADIUS: f64 = 6.957e8; // m
    pub const HUBBLE_CONSTANT: f64 = 67.4; // (km/s)/Mpc
    pub fn pi(precision: Option<u32>) -> f64 {
        match precision {
            Some(p) => {
                let scale = 10f64.powi(p as i32);
                (Self::PI * scale).round() / scale
            }
            None => Self::PI,
        }
    }

    pub fn e(precision: Option<u32>) -> f64 {
        match precision {
            Some(p) => {
                let scale = 10f64.powi(p as i32);
                (Self::E * scale).round() / scale
            }
            None => Self::E,
        }
    }

    pub fn golden_ratio(precision: Option<u32>) -> f64 {
        match precision {
            Some(p) => {
                let scale = 10f64.powi(p as i32);
                (Self::GOLDEN_RATIO * scale).round() / scale
            }
            None => Self::GOLDEN_RATIO,
        }
    }

    pub fn sqrt_2(precision: Option<u32>) -> f64 {
        match precision {
            Some(p) => {
                let scale = 10f64.powi(p as i32);
                (Self::SQRT_2 * scale).round() / scale
            }
            None => Self::SQRT_2,
        }
    }

    pub fn ln_2(precision: Option<u32>) -> f64 {
        match precision {
            Some(p) => {
                let scale = 10f64.powi(p as i32);
                (Self::LN_2 * scale).round() / scale
            }
            None => Self::LN_2,
        }
    }

    pub fn ln_10(precision: Option<u32>) -> f64 {
        match precision {
            Some(p) => {
                let scale = 10f64.powi(p as i32);
                (Self::LN_10 * scale).round() / scale
            }
            None => Self::LN_10,
        }
    }

    pub fn euler_mascheroni(precision: Option<u32>) -> f64 {
        match precision {
            Some(p) => {
                let scale = 10f64.powi(p as i32);
                (Self::EULER_MASCHERONI * scale).round() / scale
            }
            None => Self::EULER_MASCHERONI,
        }
    }

    // Physical constant methods (existing)
    pub fn speed_of_light(unit: Speed) -> Speed {
        Speed::C.convert_to(unit)
    }

    pub fn gravitational_constant(unit: Force) -> Force {
        Force::Custom(
            Mass::Kg,
            Length::Custom(Self::GRAVITATIONAL_CONSTANT, Box::new(Length::Meter)),
            Time::Second,
        )
        .convert_to(unit)
    }

    pub fn planck_constant(unit: Energy) -> Energy {
        Energy::Custom(
            Force::Custom(
                Mass::Custom(Self::PLANCK_CONSTANT, Box::new(Mass::Kg)),
                Length::Meter,
                Time::Second,
            ),
            Length::Meter,
        )
        .convert_to(unit)
    }

    pub fn reduced_planck_constant(unit: Energy) -> Energy {
        Energy::Custom(
            Force::Custom(
                Mass::Custom(Self::REDUCED_PLANCK_CONSTANT, Box::new(Mass::Kg)),
                Length::Meter,
                Time::Second,
            ),
            Length::Meter,
        )
        .convert_to(unit)
    }

    pub fn boltzmann_constant(unit: Energy) -> Energy {
        Energy::Custom(
            Force::Custom(
                Mass::Custom(Self::BOLTZMANN_CONSTANT, Box::new(Mass::Kg)),
                Length::Meter,
                Time::Second,
            ),
            Length::Meter,
        )
        .convert_to(unit)
    }

    pub fn elementary_charge(unit: ElectricCharge) -> ElectricCharge {
        ElectricCharge::Custom(
            Box::new(Current::Custom(
                ElectricCharge::Coulomb,
                Time::Custom(Self::ELEMENTARY_CHARGE, Box::new(Time::Second)),
            )),
            Time::Second,
        )
        .convert_to(unit)
    }

    pub fn electron_mass(unit: Mass) -> Mass {
        Mass::Custom(Self::ELECTRON_MASS, Box::new(Mass::Kg)).convert_to(unit)
    }

    pub fn proton_mass(unit: Mass) -> Mass {
        Mass::Custom(Self::PROTON_MASS, Box::new(Mass::Kg)).convert_to(unit)
    }

    pub fn avogadro_constant(unit: Frequency) -> Frequency {
        Frequency::Custom(Time::Custom(
            1.0 / Self::AVOGADRO_CONSTANT,
            Box::new(Time::Second),
        ))
        .convert_to(unit)
    }

    pub fn fine_structure_constant() -> f64 {
        Self::FINE_STRUCTURE_CONSTANT
    }

    pub fn rydberg_constant(unit: Frequency) -> Frequency {
        Frequency::Custom(Time::Custom(Self::RYDBERG_CONSTANT, Box::new(Time::Second)))
            .convert_to(unit)
    }

    pub fn atomic_mass_unit(unit: Mass) -> Mass {
        Mass::Custom(Self::ATOMIC_MASS_UNIT, Box::new(Mass::Kg)).convert_to(unit)
    }

    pub fn faraday_constant(unit: ElectricCharge) -> ElectricCharge {
        ElectricCharge::Custom(
            Box::new(Current::Custom(
                ElectricCharge::Coulomb,
                Time::Custom(Self::FARADAY_CONSTANT, Box::new(Time::Second)),
            )),
            Time::Second,
        )
        .convert_to(unit)
    }

    pub fn gas_constant(unit: Energy) -> Energy {
        Energy::Custom(
            Force::Custom(
                Mass::Custom(Self::GAS_CONSTANT, Box::new(Mass::Kg)),
                Length::Meter,
                Time::Second,
            ),
            Length::Meter,
        )
        .convert_to(unit)
    }
}
