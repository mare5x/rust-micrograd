mod utils;

use std::{error::Error, path::Path};

use ndarray::prelude::*;
use rust_micrograd::{
    nn::{self, Module},
    Value,
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut csv = utils::read_csv(Path::new("./examples/housing.csv"))?;
    let mu = csv.x.mean_axis(Axis(0)).unwrap();
    let sigma = csv.x.std_axis(Axis(0), 0.0);
    csv.x = (&csv.x - &mu) / &sigma;

    let vx = Value::from_ndarray(&csv.x);
    let vy = Value::from_ndarray(&csv.y);

    let model = nn::MLP::new(&[vx.shape()[1], 8, vy.shape()[1]], nn::ActivationFunc::RELU);

    let lr = 0.01;
    for _it in 0..50 {
        let y_pred = model.forward(&vx);
        let mut loss = nn::mse(&y_pred, &vy);
        println!("Loss: {}", loss.data());

        model.zero_grad();
        loss.backward();
        for param in model.parameters() {
            param.inner_mut().data -= lr * param.grad();
        }
    }

    Ok(())
}
