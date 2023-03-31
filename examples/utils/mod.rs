use csv::Reader;
use ndarray::prelude::*;
use std::{error::Error, path::Path};

pub struct CsvFormat {
    pub x: Array2<f64>,
    pub y: Array2<f64>,
}

/// Helper function for reading a CSV containing a header (which is skipped) and `f64` columns.
/// The last column is assumed to be the target variable `y` while the rest are inputs `x`.
pub fn read_csv(path: &Path) -> Result<CsvFormat, Box<dyn Error>> {
    let mut reader = Reader::from_path(path)?;
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for record in reader.records() {
        let record = record?;
        let row: Vec<f64> = record
            .iter()
            .filter_map(|x| x.parse::<f64>().ok())
            .collect();
        let (y, x) = row.split_last().unwrap();
        xs.push(x.to_vec());
        ys.push(*y);
    }

    let xs = Array2::from_shape_fn((xs.len(), xs[0].len()), |(i, j)| xs[i][j]);
    let ys = Array2::from_shape_vec((ys.len(), 1), ys)?;

    Ok(CsvFormat { x: xs, y: ys })
}
