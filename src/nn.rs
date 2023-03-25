use ndarray::prelude::*;
use ndarray_rand::{rand_distr::{Normal, num_traits}, RandomExt};

use crate::engine::Value;

// Implement some traits for use with ndarray.
// This is for array.sum_axis
impl num_traits::Zero for Value {
    fn zero() -> Self {
        Value::from(0.0)
    }

    fn is_zero(&self) -> bool {
        self.data().is_zero()
    }
}

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

trait Module {
    fn parameters(&self) -> Vec<&Value>;

    fn forward(&self, x: &Array2<Value>) -> Array2<Value>;

    fn zero_grad(&self) {
        for p in self.parameters().iter_mut() {
            p.inner_mut().grad = 0.0;
        }
    }
}

struct Linear {
    weights: Array2<Value>,
    biases: Option<Array1<Value>>,
}

impl Linear {
    fn new(in_dim: usize, out_dim: usize, include_bias: bool) -> Linear {
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
        assert_eq!(
            v_dot.map(|x| x.data()),
            a.t().dot(&a));
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
}
