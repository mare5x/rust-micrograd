use ndarray::array;
use rust_micrograd::{nn::{self, Module}, Value};

fn main() {
    let x = array![
        0.13312549122349107,
        0.39185329092401716,
        0.3173269343374667,
        0.5854101935932261,
        0.6048632706555529,
        0.7988494681347201,
        0.8290469506480111
    ];
    let n = x.dim();
    let x = x.into_shape((n, 1)).unwrap();
    let vx = Value::from_ndarray(&x);

    let y = array![
        0.2271480569496033,
        0.29348561939561973,
        0.5046767515252639,
        0.4950156627242872,
        0.6529935498747048,
        0.691360626363368,
        0.8684963618384874,
    ];
    let y = y.into_shape((n, 1)).unwrap();
    let vy = Value::from_ndarray(&y);

    let lin = nn::Linear::new(1, 1, true);
    let lr = 0.5;

    for _it in 0..100 {
        lin.zero_grad();
        let y_pred = lin.forward(&vx);
        let mut loss = nn::mse(&y_pred, &vy);
        loss.backward();
        println!("Loss: {}", loss.data());

        for param in lin.parameters() {
            param.inner_mut().data -= lr * param.grad();
        }
    }

    println!("w={:?}", lin.weights.iter().map(|x| x.data()).collect::<Vec<_>>());
    println!("b={:?}", lin.biases.map_or(vec![0.0], |b| b.iter().map(|x| x.data()).collect::<Vec<_>>()));
}
