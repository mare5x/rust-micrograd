use std::{error::Error, path::Path};

use csv::Reader;
use ndarray::prelude::*;
use rust_micrograd::{
    nn::{self, Module},
    Value,
};

pub struct CsvFormat {
    pub x: Array2<f64>,
    pub y: Array2<f64>,
}

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

fn main() -> Result<(), Box<dyn Error>> {
    let mut csv = read_csv(Path::new("./examples/housing.csv"))?;
    let mu = csv.x.mean_axis(Axis(0)).unwrap();
    let sigma = csv.x.std_axis(Axis(0), 0.0);
    csv.x = (&csv.x - &mu) / &sigma;

    let vx = Value::from_ndarray(&csv.x);
    let vy = Value::from_ndarray(&csv.y);

    let lin = nn::Linear::new(vx.shape()[1], vy.shape()[1], true);
    let lr = 0.1;
    for _it in 0..50 {
        let y_pred = lin.forward(&vx);
        let mut loss = nn::mse(&y_pred, &vy);
        println!("Loss: {}", loss.data());

        lin.zero_grad();
        loss.backward();
        for param in lin.parameters() {
            param.inner_mut().data -= lr * param.grad();
        }
    }

    println!(
        "w={:?}",
        lin.weights.iter().map(|x| x.data()).collect::<Vec<_>>()
    );
    println!(
        "b={:?}",
        lin.biases.map_or(vec![0.0], |b| b
            .iter()
            .map(|x| x.data())
            .collect::<Vec<_>>())
    );

    Ok(())
}
