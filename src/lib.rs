pub mod consts;
pub mod math;
pub mod units;

#[cfg(test)]
mod tests {
    use crate::consts::Const;
    use crate::math::calculus;
    use crate::math::complex::Complex;
    use crate::math::linear::{Matrix, Vector};
    use crate::math::stats::Stats;
    use crate::units::*;
    use std::f64::consts::PI;

    // consts
    #[test]
    fn test_mathematical_constants() {
        assert_eq!(Const::pi(Some(7)), 3.1415927);
        assert_eq!(Const::e(Some(7)), 2.7182818);
        assert_eq!(Const::golden_ratio(Some(8)), 1.61803399);

        assert!((Const::PI - std::f64::consts::PI).abs() < 1e-10);
        assert!((Const::E - std::f64::consts::E).abs() < 1e-10);
        assert!((Const::GOLDEN_RATIO - 1.618033988749895).abs() < 1e-10);
    }

    #[test]
    fn test_physical_constants() {
        assert!((Const::GRAVITATIONAL_CONSTANT - 6.67430e-11).abs() < 1e-15);
        assert!(
            (Const::REDUCED_PLANCK_CONSTANT - Const::PLANCK_CONSTANT / (2.0 * Const::PI)).abs()
                < 1e-40
        );
        assert!((Const::BOLTZMANN_CONSTANT - 1.380649e-23).abs() < 1e-28);
        assert!((Const::PROTON_MASS - 1.67262192369e-27).abs() < 1e-35);
        assert!((Const::AVOGADRO_CONSTANT - 6.02214076e23).abs() < 1e15);
    }

    #[test]
    fn test_precision_handling() {
        assert_eq!(Const::pi(Some(2)), 3.14);
        assert_eq!(Const::pi(Some(4)), 3.1416);
        assert_eq!(Const::e(Some(2)), 2.72);
        assert_eq!(Const::e(Some(4)), 2.7183);
        assert_eq!(Const::golden_ratio(Some(2)), 1.62);
        assert_eq!(Const::golden_ratio(Some(4)), 1.6180);
    }

    #[test]
    fn test_constant_unit_conversions() {
        let g_in_nm3_kg_s2 = Const::gravitational_constant(Force::Newton);
        assert!(g_in_nm3_kg_s2.to_newtons() > 0.0);

        let h_in_joule_sec = Const::planck_constant(Energy::Joule);
        assert!(h_in_joule_sec.to_joules() > 0.0);

        let kb_in_joule_per_k = Const::boltzmann_constant(Energy::Joule);
        assert!(kb_in_joule_per_k.to_joules() > 0.0);

        let me_in_atomic_units = Const::electron_mass(Mass::Atomic);
        assert!(me_in_atomic_units.to_kg() > 0.0);

        let mp_in_atomic_units = Const::proton_mass(Mass::Atomic);
        assert!(mp_in_atomic_units.to_kg() > 0.0);

        let na_in_hz = Const::avogadro_constant(Frequency::Hertz);
        assert!(na_in_hz.to_hertz() > 0.0);
    }

    #[test]
    fn test_derived_constants() {
        let fine_structure = Const::ELEMENTARY_CHARGE.powi(2)
            / (4.0
                * Const::PI
                * 8.854187812813e-12
                * Const::REDUCED_PLANCK_CONSTANT
                * Const::SPEED_OF_LIGHT);
        assert!((fine_structure - 0.0072973525693).abs() < 1e-10);
    }

    #[test]
    fn test_constant_relationships() {
        let thermal_energy_at_room_temp = Const::BOLTZMANN_CONSTANT * 300.0;
        assert!(thermal_energy_at_room_temp > 0.0);

        let electron_rest_energy = Const::ELECTRON_MASS * Const::SPEED_OF_LIGHT.powi(2);
        assert!(electron_rest_energy > 0.0);

        let proton_electron_mass_ratio = Const::PROTON_MASS / Const::ELECTRON_MASS;
        assert!((proton_electron_mass_ratio - 1836.15267343).abs() < 1e-6);
    }

    #[test]
    fn test_constant_conversions() {
        let c_in_kmh = Const::speed_of_light(Speed::KmH);
        let e_in_ev = Const::elementary_charge(ElectricCharge::ElementaryCharge);
        let m_in_kg = Const::electron_mass(Mass::Kg);
        assert!(c_in_kmh.to_mps() > 0.0);
        assert!(e_in_ev.to_coulombs() > 0.0);
        assert!(m_in_kg.to_kg() > 0.0);
    }

    // units
    #[test]
    fn test_mass_conversions() {
        let kg = Mass::Kg;
        assert!((kg.clone().convert_to(Mass::Pound).to_kg() - 1.0).abs() < 1e-10);
        assert!((kg.convert_to(Mass::Gram).to_kg() - 1.0).abs() < 1e-10);

        let ton_in_kg = Mass::Ton.to_kg();
        assert!((ton_in_kg - 1000.0).abs() < 1e-10);

        assert!((Mass::Pound.convert_to(Mass::Ounce).to_kg() - 0.45359237).abs() < 1e-8);
    }

    #[test]
    fn test_length_conversions() {
        let meter = Length::Meter;
        assert!((meter.convert_to(Length::Foot).to_meters() - 1.0).abs() < 1e-10);
        assert!((Length::Mile.convert_to(Length::Kilometer).to_meters() - 1609.344).abs() < 1e-10);
        assert!((Length::Inch.convert_to(Length::Centimeter).to_meters() - 0.0254).abs() < 1e-10);

        let nautical_mile = Length::NauticalMile;
        let meters = nautical_mile.to_meters();
        assert!((meters - 1852.0).abs() < 1e-10);
    }

    #[test]
    fn test_temperature_conversions() {
        let celsius = Temperature::Celsius;
        assert!((celsius.to_kelvin(0.0) - 273.15).abs() < 1e-10);
        assert!((celsius.to_fahrenheit(100.0) - 212.0).abs() < 1e-10);
        assert!((Temperature::Fahrenheit.to_celsius(32.0) - 0.0).abs() < 1e-10);
        assert!((Temperature::Kelvin.to_celsius(273.15) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_time_conversions() {
        let second = Time::Second;
        assert!((second.convert_to(Time::Millisecond).to_seconds() - 1.0).abs() < 1e-10);
        assert!((Time::Hour.convert_to(Time::Minute).to_seconds() - 3600.0).abs() < 1e-10);
        assert!((Time::Day.convert_to(Time::Hour).to_seconds() - 86400.0).abs() < 1e-10);

        assert!((Time::JulianYear.convert_to(Time::Day).to_seconds() - 31557600.0).abs() < 1e-10);
        assert!((Time::SiderealDay.convert_to(Time::Hour).to_seconds() - 86164.0905).abs() < 1e-6);
        assert!(
            (Time::SiderealYear.convert_to(Time::Day).to_seconds() - 31558149.504).abs() < 1e-6
        );
    }

    #[test]
    fn test_length_astronomical() {
        let light_year = Length::LightYear;
        assert!((light_year.convert_to(Length::Parsec).to_meters() - 9.461e15).abs() < 1e12);
        assert!(
            (Length::AstroUnit.convert_to(Length::Kilometer).to_meters() - 149597870700.0).abs()
                < 1e5
        );
        assert!((Length::Parsec.convert_to(Length::LightYear).to_meters() - 3.086e16).abs() < 1e13);
    }

    #[test]
    fn test_length_microscopic() {
        assert!((Length::Angstrom.convert_to(Length::Nanometer).to_meters() - 1e-10).abs() < 1e-15);
        assert!((Length::Fermi.convert_to(Length::Angstrom).to_meters() - 1e-15).abs() < 1e-20);
        assert!((Length::Micron.convert_to(Length::Millimeter).to_meters() - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_area_conversions() {
        let hectare = Area::Hectare;
        assert!((hectare.to_square_meters() - 10000.0).abs() < 1e-10);
        assert!((Area::Acre.to_square_meters() - 4046.8564224).abs() < 1e-6);
        assert!((Area::SquareMile.to_hectares() - 258.9988110336).abs() < 1e-6);
    }

    #[test]
    fn test_volume_conversions() {
        assert!((Volume::Liter.to_cubic_meters() - 0.001).abs() < 1e-10);
        assert!((Volume::Gallon.to_liters() - 3.785411784).abs() < 1e-6);
        assert!((Volume::CubicFoot.to_cubic_meters() - 0.028316846592).abs() < 1e-10);

        let custom_volume = Length::Meter.cubed();
        assert!((custom_volume.to_liters() - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_speed_conversions() {
        assert!((Speed::C.to_mps() - 299792458.0).abs() < 1e-6);
        assert!((Speed::Mach.to_kmh() - 1234.8).abs() < 1e-1);
        assert!((Speed::Knot.to_mps() - 0.514444).abs() < 1e-6);
    }

    #[test]
    fn test_force_and_energy() {
        let newton = Force::Newton;
        assert!((newton.to_pounds() - 0.224809).abs() < 1e-6);
        assert!((Force::Kg.to_newtons() - 9.80665).abs() < 1e-6);

        let joule = Energy::Joule;
        assert!((joule.to_calories() - 0.239006).abs() < 1e-6);
        assert!((Energy::Ev.to_joules() - 1.602176634e-19).abs() < 1e-25);
    }

    #[test]
    fn test_pressure_conversions() {
        assert!((Pressure::Bar.to_pascal() - 100000.0).abs() < 1e-6);
        assert!((Pressure::Atmospheric.to_pascal() - 101325.0).abs() < 1e-6);
        assert!((Pressure::Psi.to_pascal() - 6894.757293168361).abs() < 1e-6);
        assert!((Pressure::Torr.to_pascal() - 133.322).abs() < 1e-3);
    }

    #[test]
    fn test_electrical_units() {
        let coulomb = ElectricCharge::Coulomb;
        assert!((coulomb.to_elementary_charges() - 6.241509074e18).abs() < 1e12);

        let ampere = Current::Ampere;
        assert!((ampere.to_milliamperes() - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_frequency_and_acceleration() {
        let hz = Frequency::Hertz;
        assert!((hz.to_kilohertz() - 0.001).abs() < 1e-10);
        assert!((Frequency::Gigahertz.to_hertz() - 1e9).abs() < 1e-6);

        let g = Acceleration::G;
        assert!((g.to_mps2() - 9.80665).abs() < 1e-6);
        assert!((Acceleration::MetersPerSecondSquared.to_g() - 0.101972).abs() < 1e-6);
    }

    #[test]
    fn test_compound_unit_conversions() {
        let work = Energy::Custom(Force::Newton, Length::Meter);
        assert!((work.to_joules() - 1.0).abs() < 1e-10);

        let power = Energy::Joule.per_time(Time::Second);
        assert!((power.to_watts() - 1.0).abs() < 1e-10);

        let accel = Length::Meter.per_time_squared(Time::Second, Time::Second);
        assert!((accel.to_mps2() - 1.0).abs() < 1e-10);

        let freq = Time::Second.frequency();
        assert!((freq.to_hertz() - 1.0).abs() < 1e-10);
    }

    // linear algebra
    #[test]
    fn test_vector_operations() {
        let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
        let v2 = Vector::new(vec![4.0, 5.0, 6.0]);
        let v3 = Vector::new(vec![2.0, 1.0, -1.0]);

        let cross = v1.cross(&v2).unwrap();
        assert!((cross[0] + 3.0).abs() < 1e-10);
        assert!((cross[1] - 6.0).abs() < 1e-10);
        assert!((cross[2] - (-3.0)).abs() < 1e-10);

        let v4 = Vector::new(vec![1.0, 2.0]);
        assert!(v1.dot(&v4).is_err());
        assert!(v1.cross(&v4).is_err());

        let normalized = v3.normalize();
        assert!((normalized.magnitude() - 1.0).abs() < 1e-10);

        let sum = (v1 + v2).unwrap();
        assert!((sum[0] - 5.0).abs() < 1e-10);
        assert!((sum[1] - 7.0).abs() < 1e-10);
        assert!((sum[2] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_operations() {
        let m = Matrix::new(vec![
            vec![4.0, -2.0, 1.0],
            vec![-2.0, 4.0, -2.0],
            vec![1.0, -2.0, 4.0],
        ])
        .unwrap();

        let (l, u) = m.lu_decomposition().unwrap();
        let lu_product = (l * u).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!((m[(i, j)] - lu_product[(i, j)]).abs() < 1e-10);
            }
        }

        let (q, r) = m.qr_decomposition().unwrap();
        let qr_product = (q * r).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!((m[(i, j)] - qr_product[(i, j)]).abs() < 1e-10);
            }
        }

        assert!(m.is_symmetric());
        assert!(m.is_positive_definite());
        assert!((m.trace().unwrap() - 12.0).abs() < 1e-10);

        let l = m.cholesky().unwrap();
        let l_transpose = l.transpose();
        let ll_product = (l * l_transpose).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!((m[(i, j)] - ll_product[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_matrix_transformations() {
        let m = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        // Test matrix-vector multiplication
        let v = Vector::new(vec![1.0, 2.0]);
        let mv = (m.clone() * &v).unwrap();
        assert!((mv[0] - 5.0).abs() < 1e-10);
        assert!((mv[1] - 11.0).abs() < 1e-10);

        // Test matrix inverse
        let inv = m.inverse().unwrap();
        let identity = (m.clone() * inv.clone()).unwrap();
        assert!((identity[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((identity[(1, 1)] - 1.0).abs() < 1e-10);
        assert!(identity[(0, 1)].abs() < 1e-10);
        assert!(identity[(1, 0)].abs() < 1e-10);

        // Test matrix rank
        assert_eq!(m.rank().unwrap(), 2);

        let singular_matrix = Matrix::new(vec![vec![1.0, 2.0], vec![2.0, 4.0]]).unwrap();
        assert_eq!(singular_matrix.rank().unwrap(), 1);
    }

    #[test]
    fn test_matrix_special_cases() {
        assert!(Matrix::new(vec![]).is_err());

        let rect_matrix = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
        assert!(rect_matrix.determinant().is_err());
        assert!(rect_matrix.inverse().is_err());
        assert!(rect_matrix.lu_decomposition().is_err());

        let singular = Matrix::new(vec![vec![1.0, 1.0], vec![1.0, 1.0]]).unwrap();
        assert!(singular.inverse().is_err());

        let non_pd = Matrix::new(vec![vec![1.0, 2.0], vec![2.0, 1.0]]).unwrap();
        assert!(!non_pd.is_positive_definite());
    }

    #[test]
    fn test_matrix_decompositions() {
        let m = Matrix::new(vec![vec![4.0, 2.0], vec![2.0, 4.0]]).unwrap();

        let (l, u) = m.lu_decomposition().unwrap();
        assert!((l[(1, 0)] * u[(0, 1)] + l[(1, 1)] * u[(1, 1)] - m[(1, 1)]).abs() < 1e-10);

        let inverse = m.inverse().unwrap();
        assert!((inverse[(0, 0)] * m[(0, 0)] + inverse[(0, 1)] * m[(1, 0)] - 1.0).abs() < 1e-10);
    }

    // statistics
    #[test]
    fn test_central_tendency() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert!((Stats::mean(&data).unwrap() - 3.0).abs() < 1e-10);
        assert!((Stats::median(&data).unwrap() - 3.0).abs() < 1e-10);

        let data_with_mode = vec![1.0, 2.0, 2.0, 3.0, 4.0];
        let mode = Stats::mode(&data_with_mode).unwrap();
        assert_eq!(mode, vec![2.0]);
    }

    #[test]
    fn test_dispersion() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

        assert!((Stats::variance(&data).unwrap() - 4.571428571428571).abs() < 1e-10);
        assert!((Stats::std_dev(&data).unwrap() - 2.138089935299395).abs() < 1e-10);
        assert!((Stats::range(&data).unwrap() - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_distribution_characteristics() {
        let data = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0];

        let quartiles = Stats::quartiles(&data).unwrap();
        assert!((quartiles.0 - 2.0).abs() < 1e-10); // Q1
        assert!((quartiles.1 - 3.0).abs() < 1e-10); // Q2 (median)
        assert!((quartiles.2 - 4.0).abs() < 1e-10); // Q3
    }

    #[test]
    fn test_effect_size() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];

        let d = Stats::cohens_d(&data1, &data2).unwrap();
        assert!(d < 0.0); // data1 mean < data2 mean
    }

    #[test]
    fn test_regression() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let (slope, intercept) = Stats::linear_regression(&x, &y).unwrap();
        assert!((slope - 2.0).abs() < 1e-10);
        assert!(intercept.abs() < 1e-10);
    }

    #[test]
    fn test_time_series() {
        let data = vec![1.0, 2.0, 1.0, 2.0, 1.0];
        let ac1 = Stats::autocorrelation(&data, 1).unwrap();
        assert!(ac1 < 0.0);

        let ac2 = Stats::autocorrelation(&data, 2).unwrap();
        assert!(ac2 > 0.0);
    }

    // complex numbers
    #[test]
    fn test_complex_arithmetic() {
        let z1 = Complex::new(1.0, 2.0);
        let z2 = Complex::new(3.0, 4.0);

        let sum = z1 + z2;
        assert!((sum.re() as f32 - 4.0).abs() < 1e-10);
        assert!((sum.im() as f32 - 6.0).abs() < 1e-10);

        let product = z1 * z2;
        assert!((product.re() as f32 - (-5.0)).abs() < 1e-10);
        assert!((product.im() as f32 - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_complex_properties() {
        let z = Complex::new(3.0, 4.0);

        assert!((z.modulus() - 5.0).abs() < 1e-10);
        assert!((z.argument() - (4.0_f64).atan2(3.0)).abs() < 1e-10);

        let conjugate = z.conjugate();
        assert!((conjugate.re() - 3.0).abs() < 1e-10);
        assert!((conjugate.im() + 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_complex_functions() {
        let z = Complex::new(0.0, PI);

        let exp_z = z.exp();
        assert!((exp_z.re() + 1.0).abs() < 1e-10);
        assert!(exp_z.im().abs() < 1e-10);

        let z1 = Complex::new(1.0, 1.0);
        let ln_z = z1.ln();

        let expected_re = (2.0_f64).sqrt().ln();
        let expected_im = PI / 4.0;

        assert!((ln_z.re() - expected_re).abs() < 1e-10);
        assert!((ln_z.im() - expected_im).abs() < 1e-10);

        let z2 = Complex::new(1.0, 0.0);
        let ln_z2 = z2.ln();
        assert!(ln_z2.re().abs() < 1e-10);
        assert!(ln_z2.im().abs() < 1e-10);
    }

    // calc
    #[test]
    fn test_derivatives() {
        let f = |x: f64| x * x;
        assert!((calculus::Calculus::derivative(&f, 2.0, 1e-6) - 4.0).abs() < 1e-4);
        assert!((calculus::Calculus::derivative(&f, -2.0, 1e-6) + 4.0).abs() < 1e-4);
        assert!((calculus::Calculus::second_derivative(&f, 0.0, 1e-6) - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_integrals() {
        let f = |x: f64| x * x;

        let simpson = calculus::Calculus::integrate_simpson(&f, 0.0, 1.0, 1000);
        assert!((simpson - 1.0 / 3.0).abs() < 1e-4);

        let trapezoid = calculus::Calculus::integrate_trapezoid(&f, 0.0, 1.0, 1000);
        assert!((trapezoid - 1.0 / 3.0).abs() < 1e-3);

        let rectangle = calculus::Calculus::integrate_rectangle(&f, 0.0, 1.0, 1000);
        assert!((rectangle - 1.0 / 3.0).abs() < 1e-3);
    }

    #[test]
    fn test_differential_equations() {
        let f = |_x: f64, y: f64| y;

        let euler = calculus::Calculus::euler(&f, 0.0, 1.0, 0.01, 100);
        assert!((euler.last().unwrap().1 - f64::exp(1.0)).abs() < 0.1);

        let rk4 = calculus::Calculus::runge_kutta4(&f, 0.0, 1.0, 0.01, 100);
        assert!((rk4.last().unwrap().1 - f64::exp(1.0)).abs() < 0.01);
    }
}
