use rs_sci::math::complex::Complex;
use rs_sci::*;

fn main() {
    let a: Complex<f32> = complex!(2.0, 3.0);
    let b: Complex<f32> = complex!(2.0) + im!(3.0);
    println!("{}={}", a, b);
}
