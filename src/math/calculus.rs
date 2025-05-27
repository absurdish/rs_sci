use std::f64;

pub struct Calculus;

impl Calculus {
    // numerical differentiation

    /// calculates first derivative via central difference method
    ///
    /// #### Example
    /// ```txt
    /// let f = |x| x.powi(2);
    /// let derivative = Calculus::derivative(f, 2.0, 0.0001); // ≈ 4.0
    /// ```
    /// ---
    /// provides better accuracy than forward/backward difference by using points on both sides
    pub fn derivative<F: Fn(f64) -> f64>(f: F, x: f64, h: f64) -> f64 {
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    /// computes second derivative using central difference
    ///
    /// #### Example
    /// ```txt
    /// let f = |x| x.powi(3);
    /// let second_deriv = Calculus::second_derivative(f, 2.0, 0.0001);
    /// ```
    /// ---
    /// approximates second derivative with three points
    pub fn second_derivative<F: Fn(f64) -> f64>(f: F, x: f64, h: f64) -> f64 {
        (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)
    }

    /// finds partial derivative wrt x using central difference
    ///
    /// #### Example
    /// ```txt
    /// let f = |x, y| x*x + y*y;
    /// let dx = Calculus::partial_x(f, 1.0, 2.0, 0.0001); // ≈ 2.0
    /// ```
    /// ---
    /// keeps y constant while differentiating in x direction
    pub fn partial_x<F: Fn(f64, f64) -> f64>(f: F, x: f64, y: f64, h: f64) -> f64 {
        (f(x + h, y) - f(x - h, y)) / (2.0 * h)
    }

    /// finds partial derivative wrt y using central difference
    ///
    /// #### Example
    /// ```txt
    /// let f = |x, y| x*x + y*y;
    /// let dy = Calculus::partial_y(f, 1.0, 2.0, 0.0001); // ≈ 4.0
    /// ```
    /// ---
    /// keeps x constant while differentiating in y direction
    pub fn partial_y<F: Fn(f64, f64) -> f64>(f: F, x: f64, y: f64, h: f64) -> f64 {
        (f(x, y + h) - f(x, y - h)) / (2.0 * h)
    }

    //numerical integration

    /// computes definite integral using rectangle method
    ///
    /// #### Example
    /// ```txt
    /// let f = |x| x.powi(2);
    /// let integral = Calculus::integrate_rectangle(f, 0.0, 1.0, 1000); // ≈ 0.333
    /// ```
    /// ---
    /// approximates area using sum of rectangles with equal width
    pub fn integrate_rectangle<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
        let dx = (b - a) / n as f64;
        let mut sum = 0.0;

        for i in 0..n {
            let x = a + dx * i as f64;
            sum += f(x);
        }

        sum * dx
    }

    /// computes definite integral using trapezoidal method
    ///
    /// #### Example
    /// ```txt
    /// let f = |x| x.powi(2);
    /// let integral = Calculus::integrate_trapezoid(f, 0.0, 1.0, 1000); // ≈ 0.333
    /// ```
    /// ---
    /// uses linear approximation between points for better accuracy than rectangle method
    pub fn integrate_trapezoid<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
        let dx = (b - a) / n as f64;
        let mut sum = (f(a) + f(b)) / 2.0;

        for i in 1..n {
            let x = a + dx * i as f64;
            sum += f(x);
        }

        sum * dx
    }

    /// computes definite integral using simpson's method
    ///
    /// #### Example
    /// ```txt
    /// let f = |x| x.powi(2);
    /// let integral = Calculus::integrate_simpson(f, 0.0, 1.0, 1000); // ≈ 0.333
    /// ```
    /// ---
    /// uses quadratic approximation for higher accuracy than trapezoid method
    pub fn integrate_simpson<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
        if n % 2 != 0 {
            panic!("n must be even for Simpson's rule");
        }

        let dx = (b - a) / n as f64;
        let mut sum = f(a) + f(b);

        for i in 1..n {
            let x = a + dx * i as f64;
            sum += if i % 2 == 0 { 2.0 } else { 4.0 } * f(x);
        }

        sum * dx / 3.0
    }

    // differentials

    /// solves first-order ODE using euler's method
    ///
    /// #### Example
    /// ```txt
    /// let f = |x, y| x + y;
    /// let solution = Calculus::euler(f, 0.0, 1.0, 0.1, 10);
    /// ```
    /// ---
    /// basic numerical method for ODEs using linear approximation
    pub fn euler<F: Fn(f64, f64) -> f64>(
        f: F,         // dy/dx = f(x, y)
        x0: f64,      // initial x
        y0: f64,      // initial y
        h: f64,       // step size
        steps: usize, // number of steps
    ) -> Vec<(f64, f64)> {
        let mut result = vec![(x0, y0)];
        let mut x = x0;
        let mut y = y0;

        for _ in 0..steps {
            y = y + h * f(x, y);
            x = x + h;
            result.push((x, y));
        }

        result
    }

    /// solves ODE using 4th order runge-kutta method
    ///
    /// #### Example
    /// ```txt
    /// let f = |x, y| x + y;
    /// let solution = Calculus::runge_kutta4(f, 0.0, 1.0, 0.1, 10);
    /// ```
    /// ---
    /// more accurate than euler's method by using weighted average of slopes
    pub fn runge_kutta4<F: Fn(f64, f64) -> f64>(
        f: F,
        x0: f64,
        y0: f64,
        h: f64,
        steps: usize,
    ) -> Vec<(f64, f64)> {
        let mut result = vec![(x0, y0)];
        let mut x = x0;
        let mut y = y0;

        for _ in 0..steps {
            let k1 = h * f(x, y);
            let k2 = h * f(x + h / 2.0, y + k1 / 2.0);
            let k3 = h * f(x + h / 2.0, y + k2 / 2.0);
            let k4 = h * f(x + h, y + k3);

            y = y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
            x = x + h;
            result.push((x, y));
        }

        result
    }

    // series and limits

    /// approximates function using taylor series expansion
    ///
    /// #### Example
    /// ```txt
    /// let f = |x| x.exp();
    /// let df = [|x| x.exp()]; // derivatives
    /// let approx = Calculus::taylor_series(f, &df, 0.0, 0.1, 2);
    /// ```
    /// ---
    /// uses function and its derivatives to create polynomial approximation

    pub fn taylor_series<F: Fn(f64) -> f64>(
        f: F,         // function
        df: &[F],     // array of derivative functions
        a: f64,       // expansion point
        x: f64,       // evaluation point
        terms: usize, // number of terms
    ) -> f64 {
        let mut sum = f(a);
        let mut factorial = 1.0;
        let mut power = 1.0;

        for (n, derivative) in df.iter().take(terms - 1).enumerate() {
            factorial *= (n + 1) as f64;
            power *= x - a;
            sum += derivative(a) * power / factorial;
        }

        sum
    }

    // vector calc

    /// computes gradient of 2D scalar field
    ///
    /// #### Example
    /// ```txt
    /// let f = |x, y| x*x + y*y;
    /// let grad = Calculus::gradient(f, 1.0, 1.0, 0.0001);
    /// ```
    /// ---
    /// returns vector of partial derivatives (∂f/∂x, ∂f/∂y)

    pub fn gradient<F: Fn(f64, f64) -> f64>(f: F, x: f64, y: f64, h: f64) -> (f64, f64) {
        (Self::partial_x(&f, x, y, h), Self::partial_y(&f, x, y, h))
    }

    /// calculates divergence of 2D vector field
    ///
    /// #### Example
    /// ```txt
    /// let f = |x, y| (x*y, x+y);
    /// let div = Calculus::divergence(f, 1.0, 1.0, 0.0001);
    /// ```
    /// ---
    /// computes sum of partial derivatives ∂P/∂x + ∂Q/∂y

    pub fn divergence<F: Fn(f64, f64) -> (f64, f64)>(f: F, x: f64, y: f64, h: f64) -> f64 {
        let dx = |x, y| (f(x + h, y).0 - f(x - h, y).0) / (2.0 * h);
        let dy = |x, y| (f(x, y + h).1 - f(x, y - h).1) / (2.0 * h);

        dx(x, y) + dy(x, y)
    }

    /// finds z-component of curl for 2D vector field
    ///
    /// #### Example
    /// ```txt
    /// let f = |x, y| (x*y, x+y);
    /// let curl = Calculus::curl_z(f, 1.0, 1.0, 0.0001);
    /// ```
    /// ---
    /// computes \partial Q/\partial x - \partial P/\partial y

    pub fn curl_z<F: Fn(f64, f64) -> (f64, f64)>(f: F, x: f64, y: f64, h: f64) -> f64 {
        let dy_dx = (f(x + h, y).1 - f(x - h, y).1) / (2.0 * h);
        let dx_dy = (f(x, y + h).0 - f(x, y - h).0) / (2.0 * h);

        dy_dx - dx_dy
    }

    // optimization

    /// finds local minimum using gradient descent
    ///
    /// #### Example
    /// ```txt
    /// let f = |x, y| x*x + y*y;
    /// let min = Calculus::gradient_descent(f, 1.0, 1.0, 0.1, 1e-6, 1000);
    /// ```
    /// ---
    /// iteratively moves in direction of steepest descent
    pub fn gradient_descent<F: Fn(f64, f64) -> f64>(
        f: F,
        mut x: f64,
        mut y: f64,
        learning_rate: f64,
        tolerance: f64,
        max_iterations: usize,
    ) -> (f64, f64) {
        for _ in 0..max_iterations {
            let (dx, dy) = Self::gradient(&f, x, y, 1e-6);
            let new_x = x - learning_rate * dx;
            let new_y = y - learning_rate * dy;

            if (new_x - x).abs() < tolerance && (new_y - y).abs() < tolerance {
                break;
            }

            x = new_x;
            y = new_y;
        }

        (x, y)
    }
}
