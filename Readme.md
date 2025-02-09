# rs-sci

An advanced scientific computing crate, providing tools for math, physics and engineering calculations.

## Features

- **Linear Algebra**: Fully implemented vector and matrix algebra.
- **Complex Numbers**: Out-of-the-Box complex number support.
- **Calculus**: Integrate, differentiate, expand series and solve differential equations.
- **Statistics**: Everything you might need.
- **Constants**: 30+ mathematical and physical units, precision and unit convertable.
- **Units**: SI, US, Astronomical, Compound, Physical... All kinds of units.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
rs-sci = "0.1.0"
```

Or simply:

```sh
cargo add rs-sci
```

### Examples

```rust
use rs_sci::complex::Complex;
use rs_sci::consts::Const;
use rs_sci::linear::{Matrix, Vector};
use rs_sci::units::*;

// complex number operations
let z1 = Complex::new(1.0, 2.0);
let z2 = z1.exp();

// physical constants with units
let c = Const::speed_of_light(Speed::MeterPerSecond);
let h = Const::planck_constant(Energy::Joule);

// linear algebra
let m = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
let v = Vector::new(vec![1.0, 2.0]);
let mv = (&m * &v).unwrap();

// unit conversions
let meters = Length::Foot.convert_to(Length::Meter);
```

## Documentation

Too early for documenting since much more will be implemented in the near future

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
