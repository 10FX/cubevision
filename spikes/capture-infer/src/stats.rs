//! Timing summaries

/// The `p` percentile (0..100) by nearest rank; sorts `values`
pub fn percentile(values: &mut [f64], p: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    let rank = ((p / 100.0) * values.len() as f64).ceil() as usize;
    values[rank.clamp(1, values.len()) - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_of_1_to_100() {
        let mut values: Vec<f64> = (1..=100).map(f64::from).collect();
        assert!((percentile(&mut values, 95.0) - 95.0).abs() <= 1.0);
        assert!((percentile(&mut values, 50.0) - 50.0).abs() <= 1.0);
    }
}
