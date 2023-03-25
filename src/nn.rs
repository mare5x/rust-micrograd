pub use ndarray::prelude::*;
use ndarray_rand::{rand_distr::Normal, RandomExt};

use crate::engine::Value;

/// Matrix multiply rectangular Value arrays.
fn dot(a: &Array2<Value>, b: &Array2<Value>) -> Array2<Value> {
    let (n1, _m1) = a.dim();
    let (_n2, m2) = b.dim();
    Array::from_shape_fn((n1, m2), |(i, j)| {
        let a_i = a.index_axis(Axis(0), i);
        let b_j = b.index_axis(Axis(1), j);
        (&a_i * &b_j).sum()
    })
}

pub fn mse<D>(y_pred: &Array<Value, D>, y_true: &Array<Value, D>) -> Value
where
    D: Dimension,
{
    let out = y_pred - y_true;
    let out = out.mapv(|x| &x * &x);
    out.mean().unwrap()
}

pub trait Module {
    fn parameters(&self) -> Vec<&Value>;

    fn forward(&self, x: &Array2<Value>) -> Array2<Value>;

    fn zero_grad(&self) {
        for p in self.parameters().iter_mut() {
            p.inner_mut().grad = 0.0;
        }
    }
}

pub struct Linear {
    pub weights: Array2<Value>,
    pub biases: Option<Array1<Value>>,
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize, include_bias: bool) -> Linear {
        let weights = Array2::random((in_dim, out_dim), Normal::new(0.0, 1.0).unwrap());
        let biases = if include_bias {
            Some(Array::zeros(out_dim))
        } else {
            None
        };
        Linear {
            weights: Value::from_ndarray(&weights),
            biases: biases.map(|x| Value::from_ndarray(&x)),
        }
    }
}

impl Module for Linear {
    fn parameters(&self) -> Vec<&Value> {
        let mut p: Vec<_> = self.weights.iter().collect();
        if let Some(biases) = &self.biases {
            p.extend(biases.iter())
        }
        p
    }

    fn forward(&self, x: &Array2<Value>) -> Array2<Value> {
        let out = dot(x, &self.weights);
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
    fn parameters(&self) -> Vec<&Value> {
        self.layers
            .iter()
            .map(|layer| layer.parameters())
            .flatten()
            .collect()
    }

    fn forward(&self, x: &Array2<Value>) -> Array2<Value> {
        let mut x = x.clone();
        for layer in self.layers.iter() {
            x = layer.forward(&x);
            x = match &self.act {
                ActivationFunc::RELU => x.mapv(|x| x.relu()),
            }
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_ndarray() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let v = a.map(|x| Value::from(*x));
        for (idx, x) in a.indexed_iter() {
            assert_eq!(v[idx].data(), *x);
        }
    }

    #[test]
    fn value_ndarray_sum() {
        // Element wise matrix operations work correctly with Value :)

        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let v = a.map(|x| Value::from(*x));
        let sum_v = &v + &v;
        let sum_a = &a + &a;
        for (idx, x) in sum_a.indexed_iter() {
            assert_eq!(sum_v[idx].data(), *x);
        }

        let sum_v = v.sum_axis(Axis(0)).map(|x| x.data());
        assert_eq!(sum_v, a.sum_axis(Axis(0)));

        assert_eq!((&v * &v).map(|x| x.data()), &a * &a);
    }

    #[test]
    fn value_dot() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let v = Value::from_ndarray(&a);
        let v_t = Value::from_ndarray(&a.t().to_owned());
        let v_dot = dot(&v_t, &v);
        assert_eq!(v_dot.map(|x| x.data()), a.t().dot(&a));
    }

    #[test]
    fn linear() {
        let lin = Linear::new(3, 2, true);
        assert_eq!(lin.parameters().len(), 3 * 2 + 2);

        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let v = Value::from_ndarray(&a);
        let out = lin.forward(&v);
        assert_eq!(out.dim(), (2, 2));
    }

    #[test]
    fn test_mse() {
        let a = array![1.0, 2.0, 3.0];
        let va = Value::from_ndarray(&a);
        let b = array![1.0, 1.0, 2.0];
        let vb = Value::from_ndarray(&b);
        let mut vmse = mse(&va, &vb);
        assert_eq!(vmse.data(), (&a - &b).mapv(|x| x * x).mean().unwrap());

        // Check gradient computation (dLoss w.r.t. da)
        vmse.backward();
        for (i, vai) in va.iter().enumerate() {
            assert_eq!(vai.grad(), 1.0 / (a.len() as f64) * 2.0 * (&a[i] - &b[i]));
        }
    }

    #[test]
    fn mlp() {
        let mlp = MLP::new(&[3, 2, 1], ActivationFunc::RELU);
        assert_eq!(mlp.layers.len(), 2);
        assert_eq!(mlp.parameters().len(), 3 * 2 + 2 + 2 * 1);

        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let v = Value::from_ndarray(&a);
        let out = mlp.forward(&v);
        assert_eq!(out.dim(), (2, 1));
    }
}
