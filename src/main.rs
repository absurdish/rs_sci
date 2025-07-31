use rs_sci::math::linear::Matrix;
use rs_sci::matrix;

fn main() {
    let A: Matrix = matrix!([[1.0, 0.0], [0.0, -1.0]]);
    let B: Matrix = Matrix::new(vec![vec![0.0, -1.0], vec![1.0, 0.0]]).unwrap();


    println!("x = {}", (A + B).unwrap());
}

// - complex matrices