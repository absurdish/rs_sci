use std::collections::HashMap;

use crate::consts::Const;

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
        let squared_diff_sum: f64 = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum();
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
        let q3 = Stats::median(if sorted.len() % 2 == 0 { upper } else { &upper[1..] })?;
        
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
        
        let m3 = data.iter()
            .map(|&x| (x - mean).powi(3))
            .sum::<f64>() / n;
            
        Some(m3 / std_dev.powi(3))
    }

    pub fn kurtosis(data: &[f64]) -> Option<f64> {
        if data.len() < 4 {
            return None;
        }
        let mean = Stats::mean(data)?;
        let std_dev = Stats::std_dev(data)?;
        let n = data.len() as f64;
        
        let m4 = data.iter()
            .map(|&x| (x - mean).powi(4))
            .sum::<f64>() / n;
            
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
        
        let sum = x.iter().zip(y.iter())
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
        
        Some(data.iter()
            .map(|&x| (x - mean) / std_dev)
            .collect())
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

    pub fn binomial_pmf(k: u64, n: u64, p: f64) -> f64 {
        if p < 0.0 || p > 1.0 {
            return 0.0;
        }
        let combinations = Stats::combinations(n, k);
        combinations as f64 * p.powi(k as i32) * (1.0 - p).powi((n - k) as i32)
    }

    // Helper function for combinations
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