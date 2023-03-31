mod utils;

use std::{error::Error, fs, ops::Deref, path::Path};

use ndarray::prelude::*;
use rust_micrograd::{
    tensor::Tensor,
    tensor_nn::{self as nn, Module},
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut csv = utils::read_csv(Path::new("./examples/housing.csv"))?;
    let mu = csv.x.mean_axis(Axis(0)).unwrap();
    let sigma = csv.x.std_axis(Axis(0), 0.0);
    csv.x = (&csv.x - &mu) / &sigma;

    let model = nn::Linear::new(csv.x.shape()[1], csv.y.shape()[1], true);
    let vx = Tensor::from(csv.x.clone());
    let vy = Tensor::from(csv.y.clone());

    let lr = 0.1;
    for _it in 0..50 {
        let y_pred = model.forward(&vx);
        let mut loss = nn::mse(&y_pred, &vy);
        println!("Loss: {}", loss.item());

        model.zero_grad();
        loss.backward();
        for param in model.parameters() {
            // N.B. `param.data_mut().scaled_add(-lr, &param.grad().deref());` doesn't work
            // because https://stackoverflow.com/questions/47060266/error-while-trying-to-borrow-2-fields-from-a-struct-wrapped-in-refcell
            // So we have to first mutably borrow the whole struct, and then we can get an immutable reference to grad.
            let data = &mut *param.inner_mut();
            data.data.scaled_add(-lr, &data.grad);
        }
    }

    println!("w={:?}", model.weights.data().deref());
    println!(
        "b={:?}",
        model
            .biases
            .as_ref()
            .map_or(arr2(&[[0.0]]), |b| b.data().deref().clone())
    );

    // Write the computation graph to graphviz dot format.
    let mut loss = nn::mse(&model.forward(&vx), &vy);
    loss.backward();
    fs::write("./tensor_linreg.dot", loss.to_graphviz())?;

    Ok(())
}
