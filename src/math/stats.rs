use crate::consts::Const;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Stats;

impl Stats {
    // Central Tendency Measures
    pub fn mean(data: &[f64]) -> Option<f64> {
        if data.is_empty() {
            return None;
        }
        Some(data.iter().sum::<f64>() / data.len() as f64)
    }

    pub fn median(data: &[f64]) -> Option<f64> {
        if data.is_empty() {
            return None;
        }
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            Some((sorted[mid - 1] + sorted[mid]) / 2.0)
        } else {
            Some(sorted[mid])
        }
    }

    pub fn mode(data: &[f64]) -> Option<Vec<f64>> {
        if data.is_empty() {
            return None;
        }
        let mut freq_map: HashMap<String, usize> = HashMap::new();

        // Convert to strings to handle floating point comparison
        for value in data {
            let key = format!("{:.10}", value);
            *freq_map.entry(key).or_insert(0) += 1;
        }

        let max_freq = freq_map.values().max()?;
        let modes: Vec<f64> = freq_map
            .iter()
            .filter(|(_, &count)| count == *max_freq)
            .map(|(key, _)| key.parse::<f64>().unwrap())
            .collect();

        Some(modes)
    }

    // Dispersion Measures
    pub fn variance(data: &[f64]) -> Option<f64> {
        if data.len() < 2 {
            return None;
        }
        let mean = Stats::mean(data)?;
        let squared_diff_sum: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum();
        Some(squared_diff_sum / (data.len() - 1) as f64)
    }

    pub fn std_dev(data: &[f64]) -> Option<f64> {
        Some(Stats::variance(data)?.sqrt())
    }

    pub fn range(data: &[f64]) -> Option<f64> {
        if data.is_empty() {
            return None;
        }
        let min = data.iter().min_by(|a, b| a.partial_cmp(b).unwrap())?;
        let max = data.iter().max_by(|a, b| a.partial_cmp(b).unwrap())?;
        Some(max - min)
    }

    // Quartiles and Percentiles
    pub fn quartiles(data: &[f64]) -> Option<(f64, f64, f64)> {
        if data.is_empty() {
            return None;
        }
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let q2 = Stats::median(&sorted)?;
        let (lower, upper) = sorted.split_at(sorted.len() / 2);
        let q1 = Stats::median(lower)?;
        let q3 = Stats::median(if sorted.len() % 2 == 0 {
            upper
        } else {
            &upper[1..]
        })?;

        Some((q1, q2, q3))
    }

    pub fn percentile(data: &[f64], p: f64) -> Option<f64> {
        if data.is_empty() || p < 0.0 || p > 100.0 {
            return None;
        }
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let rank = p / 100.0 * (sorted.len() - 1) as f64;
        let lower_idx = rank.floor() as usize;
        let upper_idx = rank.ceil() as usize;

        if lower_idx == upper_idx {
            Some(sorted[lower_idx])
        } else {
            let weight = rank - lower_idx as f64;
            Some(sorted[lower_idx] * (1.0 - weight) + sorted[upper_idx] * weight)
        }
    }

    // Distribution Characteristics
    pub fn skewness(data: &[f64]) -> Option<f64> {
        if data.len() < 3 {
            return None;
        }
        let mean = Stats::mean(data)?;
        let std_dev = Stats::std_dev(data)?;
        let n = data.len() as f64;

        let m3 = data.iter().map(|&x| (x - mean).powi(3)).sum::<f64>() / n;

        Some(m3 / std_dev.powi(3))
    }

    pub fn kurtosis(data: &[f64]) -> Option<f64> {
        if data.len() < 4 {
            return None;
        }
        let mean = Stats::mean(data)?;
        let std_dev = Stats::std_dev(data)?;
        let n = data.len() as f64;

        let m4 = data.iter().map(|&x| (x - mean).powi(4)).sum::<f64>() / n;

        Some(m4 / std_dev.powi(4) - 3.0) // Excess kurtosis
    }

    // Correlation and Covariance
    pub fn covariance(x: &[f64], y: &[f64]) -> Option<f64> {
        if x.len() != y.len() || x.is_empty() {
            return None;
        }
        let mean_x = Stats::mean(x)?;
        let mean_y = Stats::mean(y)?;
        let n = x.len() as f64;

        let sum = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>();

        Some(sum / (n - 1.0))
    }

    pub fn correlation(x: &[f64], y: &[f64]) -> Option<f64> {
        let cov = Stats::covariance(x, y)?;
        let std_x = Stats::std_dev(x)?;
        let std_y = Stats::std_dev(y)?;

        Some(cov / (std_x * std_y))
    }

    // Z-scores and Standardization
    pub fn z_scores(data: &[f64]) -> Option<Vec<f64>> {
        let mean = Stats::mean(data)?;
        let std_dev = Stats::std_dev(data)?;

        Some(data.iter().map(|&x| (x - mean) / std_dev).collect())
    }

    // Robust Statistics
    pub fn mad(data: &[f64]) -> Option<f64> {
        // Median Absolute Deviation
        let median = Stats::median(data)?;
        let deviations: Vec<f64> = data.iter().map(|&x| (x - median).abs()).collect();
        Stats::median(&deviations)
    }

    pub fn winsorized_mean(data: &[f64], trim: f64) -> Option<f64> {
        if data.is_empty() || trim < 0.0 || trim > 0.5 {
            return None;
        }
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted.len();
        let k = (n as f64 * trim).floor() as usize;

        let lower = sorted[k];
        let upper = sorted[n - k - 1];

        let sum: f64 = sorted.iter().map(|&x| x.max(lower).min(upper)).sum();

        Some(sum / n as f64)
    }

    pub fn moving_average(data: &[f64], window: usize) -> Option<Vec<f64>> {
        if window == 0 || window > data.len() {
            return None;
        }

        let mut result = Vec::with_capacity(data.len() - window + 1);
        for i in 0..=data.len() - window {
            let mean = Stats::mean(&data[i..i + window])?;
            result.push(mean);
        }
        Some(result)
    }

    pub fn exponential_smoothing(data: &[f64], alpha: f64) -> Option<Vec<f64>> {
        if data.is_empty() || alpha < 0.0 || alpha > 1.0 {
            return None;
        }

        let mut result = Vec::with_capacity(data.len());
        result.push(data[0]);

        for i in 1..data.len() {
            let smoothed = alpha * data[i] + (1.0 - alpha) * result[i - 1];
            result.push(smoothed);
        }
        Some(result)
    }

    // Multiple Testing Corrections
    pub fn bonferroni_correction(p_values: &[f64]) -> Option<Vec<f64>> {
        if p_values.is_empty() {
            return None;
        }
        let n = p_values.len() as f64;
        Some(p_values.iter().map(|&p| (p * n).min(1.0)).collect())
    }

    pub fn benjamini_hochberg(p_values: &[f64]) -> Option<Vec<f64>> {
        if p_values.is_empty() {
            return None;
        }

        // Create indexed p-values
        let mut indexed_p: Vec<(usize, &f64)> = p_values.iter().enumerate().collect();
        indexed_p.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());

        let n = p_values.len();
        let mut adjusted = vec![0.0; n];

        // Calculate adjusted p-values
        let mut prev = 1.0;
        for (rank, (orig_idx, &p)) in indexed_p.iter().enumerate().rev() {
            let adj_p = (p * n as f64 / (rank + 1) as f64).min(prev);
            adjusted[*orig_idx] = adj_p;
            prev = adj_p;
        }

        Some(adjusted)
    }

    // Kernel Density Estimation
    pub fn kernel_density_estimation(data: &[f64], x: f64, bandwidth: f64) -> Option<f64> {
        if data.is_empty() || bandwidth <= 0.0 {
            return None;
        }

        let n = data.len() as f64;
        let density: f64 = data
            .iter()
            .map(|&xi| {
                let u = (x - xi) / bandwidth;
                Stats::gaussian_kernel(u) / bandwidth
            })
            .sum::<f64>()
            / n;

        Some(density)
    }

    fn gaussian_kernel(u: f64) -> f64 {
        (-0.5 * u * u).exp() / (2.0 * Const::PI).sqrt()
    }

    // Bayesian Statistics
    pub fn bayesian_update(prior: f64, likelihood: f64, evidence: f64) -> Option<f64> {
        if evidence == 0.0 || prior < 0.0 || prior > 1.0 || likelihood < 0.0 {
            return None;
        }
        Some(prior * likelihood / evidence)
    }

    // Information Theory Metrics
    pub fn entropy(probabilities: &[f64]) -> Option<f64> {
        if probabilities.is_empty()
            || probabilities.iter().any(|&p| p < 0.0 || p > 1.0)
            || (probabilities.iter().sum::<f64>() - 1.0).abs() > 1e-10
        {
            return None;
        }

        Some(
            -probabilities
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| p * p.log2())
                .sum::<f64>(),
        )
    }

    pub fn kullback_leibler_divergence(p: &[f64], q: &[f64]) -> Option<f64> {
        if p.len() != q.len() || p.is_empty() {
            return None;
        }

        Some(
            p.iter()
                .zip(q.iter())
                .filter(|(&pi, &qi)| pi > 0.0 && qi > 0.0)
                .map(|(&pi, &qi)| pi * (pi / qi).log2())
                .sum(),
        )
    }
}

// Summary Statistics
pub fn summary_statistics(data: &[f64]) -> Option<SummaryStats> {
    Some(SummaryStats {
        count: data.len(),
        mean: Stats::mean(data)?,
        median: Stats::median(data)?,
        mode: Stats::mode(data)?,
        variance: Stats::variance(data)?,
        std_dev: Stats::std_dev(data)?,
        skewness: Stats::skewness(data)?,
        kurtosis: Stats::kurtosis(data)?,
        range: Stats::range(data)?,
        quartiles: Stats::quartiles(data)?,
    })
}

#[derive(Debug)]
pub struct SummaryStats {
    pub count: usize,
    pub mean: f64,
    pub median: f64,
    pub mode: Vec<f64>,
    pub variance: f64,
    pub std_dev: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub range: f64,
    pub quartiles: (f64, f64, f64),
}

// Probability Distributions
impl Stats {
    pub fn normal_pdf(x: f64, mean: f64, std_dev: f64) -> f64 {
        let exponent = -(x - mean).powi(2) / (2.0 * std_dev.powi(2));
        (1.0 / (std_dev * (2.0 * Const::PI).sqrt())) * exponent.exp()
    }

    // Effect Size Measures
    pub fn cohens_d(data1: &[f64], data2: &[f64]) -> Option<f64> {
        let mean1 = Stats::mean(data1)?;
        let mean2 = Stats::mean(data2)?;
        let var1 = Stats::variance(data1)?;
        let var2 = Stats::variance(data2)?;
        let pooled_std = ((var1 + var2) / 2.0).sqrt();
        Some((mean1 - mean2) / pooled_std)
    }

    // Regression Analysis
    pub fn linear_regression(x: &[f64], y: &[f64]) -> Option<(f64, f64)> {
        // Returns (slope, intercept)
        if x.len() != y.len() || x.is_empty() {
            return None;
        }

        let mean_x = Stats::mean(x)?;
        let mean_y = Stats::mean(y)?;
        let cov = Stats::covariance(x, y)?;
        let var_x = Stats::variance(x)?;

        let slope = cov / var_x;
        let intercept = mean_y - slope * mean_x;

        Some((slope, intercept))
    }

    // Time Series Analysis
    pub fn autocorrelation(data: &[f64], lag: usize) -> Option<f64> {
        if lag >= data.len() {
            return None;
        }

        let mean = Stats::mean(data)?;
        let n = data.len() - lag;

        let numerator: f64 = (0..n)
            .map(|i| (data[i] - mean) * (data[i + lag] - mean))
            .sum();
        let denominator: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum();

        Some(numerator / denominator)
    }

    pub fn binomial_pmf(k: u64, n: u64, p: f64) -> f64 {
        if p < 0.0 || p > 1.0 {
            return 0.0;
        }
        let combinations = Stats::combinations(n, k);
        combinations as f64 * p.powi(k as i32) * (1.0 - p).powi((n - k) as i32)
    }

    fn combinations(n: u64, k: u64) -> u64 {
        if k > n {
            return 0;
        }
        let k = k.min(n - k);
        let mut result = 1;
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        result
    }
}
