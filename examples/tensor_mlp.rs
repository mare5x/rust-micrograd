mod utils;

use std::{error::Error, fs, path::Path, time::Instant};

use ndarray::prelude::*;
use rust_micrograd::{
    tensor::nn::{self as nn, Module},
    tensor::Tensor,
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut csv = utils::read_csv(Path::new("./examples/housing.csv"))?;
    let mu = csv.x.mean_axis(Axis(0)).unwrap();
    let sigma = csv.x.std_axis(Axis(0), 0.0);
    csv.x = (&csv.x - &mu) / &sigma;

    let model = nn::MLP::new(
        &[csv.x.shape()[1], 8, csv.y.shape()[1]],
        nn::ActivationFunc::RELU,
    );
    let vx = Tensor::from(csv.x.clone());
    let vy = Tensor::from(csv.y.clone());

    let lr = 0.01;
    let tic = Instant::now();
    for _it in 0..1000 {
        let y_pred = model.forward(&vx);
        let mut loss = nn::mse(&y_pred, &vy);
        println!("Loss: {}", loss.item());

        model.zero_grad();
        loss.backward();
        for param in model.parameters() {
            let data = &mut *param.inner_mut();
            data.data.scaled_add(-lr, &data.grad);
        }
    }
    println!("Took {:?}", tic.elapsed());

    // Write the computation graph to graphviz dot format.
    let mut loss = nn::mse(&model.forward(&vx), &vy);
    loss.backward();
    fs::write("./examples/tensor_mlp.dot", loss.to_graphviz())?;

    Ok(())
}
