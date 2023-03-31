pub use ndarray::prelude::*;
use ndarray_rand::{rand_distr::Normal, RandomExt};

use crate::tensor::Tensor;

pub fn mse(y_pred: &Tensor, y_true: &Tensor) -> Tensor {
    let out = y_pred - y_true;
    let out = &out * &out;
    out.mean()
}

pub trait Module {
    fn parameters(&self) -> Vec<&Tensor>;

    fn forward(&self, x: &Tensor) -> Tensor;

    fn zero_grad(&self) {
        for p in self.parameters().iter_mut() {
            p.inner_mut().grad.fill(0.0);
        }
    }
}

pub struct Linear {
    pub weights: Tensor,
    pub biases: Option<Tensor>,
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize, include_bias: bool) -> Linear {
        let weights = Array2::random((in_dim, out_dim), Normal::new(0.0, 1.0).unwrap());
        let biases = if include_bias {
            Some(Array::zeros((1, out_dim)))
        } else {
            None
        };
        Linear {
            weights: Tensor::from(weights),
            biases: biases.map(|x| Tensor::from(x)),
        }
    }
}

impl Module for Linear {
    fn parameters(&self) -> Vec<&Tensor> {
        let mut p: Vec<_> = vec![&self.weights];
        if let Some(biases) = &self.biases {
            p.push(biases)
        }
        p
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let out = x.dot(&self.weights);
        if let Some(b) = &self.biases {
            &out + b
        } else {
            out
        }
    }
}

pub enum ActivationFunc {
    RELU,
}

pub struct MLP {
    pub layers: Vec<Linear>,
    pub act: ActivationFunc,
}

impl MLP {
    pub fn new(sizes: &[usize], act: ActivationFunc) -> MLP {
        assert!(sizes.len() >= 2);
        let mut layers = Vec::new();
        for i in 0..sizes.len() - 1 {
            let bias = i < sizes.len() - 2;
            let layer = Linear::new(sizes[i], sizes[i + 1], bias);
            layers.push(layer);
        }
        MLP { layers, act }
    }
}

impl Module for MLP {
    fn parameters(&self) -> Vec<&Tensor> {
        self.layers
            .iter()
            .map(|layer| layer.parameters())
            .flatten()
            .collect()
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let mut x = x.clone();
        for layer in self.layers.iter() {
            x = layer.forward(&x);
            x = match &self.act {
                ActivationFunc::RELU => x.relu(),
            }
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear() {
        let lin = Linear::new(3, 2, true);
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let v = Tensor::from(a);
        let out = lin.forward(&v);
        assert_eq!(out.data().dim(), (2, 2));
    }

    #[test]
    fn test_mse() {
        let a = array![[1.0, 2.0, 3.0]];
        let va = Tensor::from(a.clone());
        let b = array![[1.0, 1.0, 2.0]];
        let vb = Tensor::from(b.clone());
        let mut vmse = mse(&va, &vb);
        assert!(vmse.data()[[0, 0]] == (&a - &b).mapv(|x| x * x).mean().unwrap());

        // Check gradient computation (dLoss w.r.t. da)
        vmse.backward();
        for (i, vai) in va.grad().iter().enumerate() {
            assert_eq!(
                *vai,
                1.0 / (a.len() as f64) * 2.0 * (&a[[0, i]] - &b[[0, i]])
            );
        }
    }

    #[test]
    fn mlp() {
        let mlp = MLP::new(&[3, 2, 1], ActivationFunc::RELU);
        assert_eq!(mlp.layers.len(), 2);

        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let v = Tensor::from(a);
        let out = mlp.forward(&v);
        assert_eq!(out.data().dim(), (2, 1));
    }
}
