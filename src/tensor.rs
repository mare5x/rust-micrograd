use std::{
    cell::{Ref, RefCell, RefMut},
    collections::{HashMap, HashSet},
    fmt,
    ops::{Add, Deref, DerefMut, Div, Mul, Sub},
    rc::Rc,
    sync::atomic::AtomicUsize,
};

use ndarray::prelude::*;

// TODO don't do this
// N.B. This was only necessary for topological sorting...
static VAL_CNT: AtomicUsize = AtomicUsize::new(0);

/// Holds a closure that backward propagates the gradient to the children nodes.
/// The `name` field is only for debugging convenience.
pub struct GradFn {
    name: String,
    grad_fn: Box<dyn FnMut(Array2<f64>) -> ()>,
}

impl fmt::Debug for GradFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GradFn[{}]", self.name)
    }
}

impl GradFn {
    fn new(name: &str, grad_fn: impl FnMut(Array2<f64>) -> () + 'static) -> Self {
        Self {
            name: String::from(name),
            grad_fn: Box::new(grad_fn),
        }
    }

    fn empty() -> GradFn {
        Self::new(" ", |_| ())
    }
}

#[derive(Debug)]
pub struct TensorData {
    pub data: Array2<f64>,
    pub grad: Array2<f64>,
    prev: Vec<Tensor>,
    grad_fn: GradFn,
    id: usize, // Unique id; for sorting purposes.
}

impl TensorData {
    fn new(data: Array2<f64>) -> TensorData {
        let id = VAL_CNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let dim = data.raw_dim();
        TensorData {
            data,
            grad: Array::zeros(dim),
            prev: vec![],
            grad_fn: GradFn::empty(),
            id,
        }
    }

    /// Initiate local backward pass computation, updating the `grad`
    /// of the children in the node's computation graph.
    fn backward(&mut self) {
        (self.grad_fn.grad_fn)(self.grad.clone())
    }
}

#[derive(Debug, Clone)]
pub struct Tensor(Rc<RefCell<TensorData>>);

impl Tensor {
    fn new(value: TensorData) -> Tensor {
        Tensor(Rc::new(RefCell::new(value)))
    }

    pub fn from(arr: Array2<f64>) -> Tensor {
        Tensor::new(TensorData::new(arr))
    }

    pub fn from_f64(data: f64) -> Tensor {
        Tensor::new(TensorData::new(arr2(&[[data]])))
    }

    /// Create a GraphViz DOT format string representation of the computation graph.
    pub fn to_graphviz(&self) -> String {
        let mut gradfn_colors = HashMap::new();
        gradfn_colors.insert("add", 1);
        gradfn_colors.insert("sub", 1);
        gradfn_colors.insert("mul", 2);
        gradfn_colors.insert("div", 2);
        gradfn_colors.insert("relu", 3);
        gradfn_colors.insert("matmul", 4);
        gradfn_colors.insert("sum", 5);

        fn inner(node: &Tensor, colors: &HashMap<&str, i32>) -> String {
            let id = node.inner().id;
            let mut s = format!(
                "{} [label=\"{{{} | {:?}}}\", color={}];\n",
                id,
                node.inner().grad_fn.name,
                node.data().shape(),
                // node.data(),
                // node.grad(),
                colors
                    .get(&node.inner().grad_fn.name.as_str())
                    .unwrap_or(&0)
            );
            for prev in node.inner().prev.iter() {
                s.push_str(&inner(&prev, colors));
                s.push_str(&format!("{} -- {};\n", id, prev.inner().id));
            }
            s
        }

        let mut s = format!("strict graph {{\n");
        s.push_str("rankdir=RL;\n");
        s.push_str("node [shape=record,colorscheme=set28];\n");
        s.push_str(&inner(&self, &gradfn_colors));
        s.push_str("}\n");
        s
    }

    pub fn inner(&self) -> Ref<TensorData> {
        (*self.0).borrow()
    }

    pub fn inner_mut(&self) -> RefMut<TensorData> {
        (*self.0).borrow_mut()
    }

    pub fn data(&self) -> impl Deref<Target = Array2<f64>> + '_ {
        Ref::map((*self.0).borrow(), |mi| &mi.data)
    }

    pub fn data_mut(&self) -> impl DerefMut<Target = Array2<f64>> + '_ {
        RefMut::map((*self.0).borrow_mut(), |mi| &mut mi.data)
    }

    pub fn grad(&self) -> impl Deref<Target = Array2<f64>> + '_ {
        Ref::map((*self.0).borrow(), |mi| &mi.grad)
    }

    pub fn grad_mut(&self) -> impl DerefMut<Target = Array2<f64>> + '_ {
        RefMut::map((*self.0).borrow_mut(), |mi| &mut mi.grad)
    }

    pub fn item(&self) -> f64 {
        self.data()[[0, 0]]
    }

    fn topological_sort(root: &Tensor) -> Vec<Tensor> {
        fn build(root: &Tensor, visited: &mut HashSet<usize>, out: &mut Vec<Tensor>) {
            visited.insert(root.inner().id);
            for v in root.inner().prev.iter() {
                if !visited.contains(&v.inner().id) {
                    build(v, visited, out);
                }
            }
            out.push(root.clone());
        }
        let mut visited = HashSet::new();
        let mut out = Vec::new();
        build(root, &mut visited, &mut out);
        out.reverse();
        out
    }

    /// Initiate gradient backpropagation, calculating the gradient
    /// w.r.t. to this node's value.
    pub fn backward(&mut self) {
        // We must first topologically sort the nodes in the computation
        // graph so that a node doesn't start backpropagating its gradient
        // before its own gradient is calculated.
        let order = Self::topological_sort(self);

        // d(self) w.r.t. self = 1.0
        self.grad_mut().fill(1.0);
        for v in order.iter() {
            v.inner_mut().backward();
        }
    }

    pub fn relu(&self) -> Tensor {
        let v = self.clone();
        let grad_fn = GradFn::new("relu", move |grad| {
            let dv = v.data().map(|x| if *x > 0.0 { 1.0 } else { 0.0 });
            v.grad_mut().scaled_add(1.0, &(grad * dv));
        });

        let new = self.data().map(|x| if *x > 0.0 { *x } else { 0.0 });
        let mut v = TensorData::new(new);
        v.prev.push(self.clone());
        v.grad_fn = grad_fn;

        Tensor::new(v)
    }

    pub fn sum(&self) -> Tensor {
        let v = self.clone();
        let grad_fn = GradFn::new("sum", move |grad| {
            v.grad_mut().scaled_add(1.0, &grad);
        });
        let mut v = TensorData::new(arr2(&[[self.data().sum()]]));
        v.prev.push(self.clone());
        v.grad_fn = grad_fn;

        Tensor::new(v)
    }

    pub fn mean(&self) -> Tensor {
        let n = self.data().len();
        (1.0 / n as f64) * self.sum()
    }

    pub fn dot(&self, rhs: &Tensor) -> Tensor {
        let lhs = self.clone();
        let rhs1 = rhs.clone();

        let grad_fn = GradFn::new("matmul", move |grad| {
            let da = grad.dot(&rhs1.data().t());
            let db = lhs.data().t().dot(&grad);
            lhs.grad_mut().scaled_add(1.0, &da);
            rhs1.grad_mut().scaled_add(1.0, &db);
        });

        let c = self.data().dot(rhs.data().deref());
        let mut v = TensorData::new(c);
        v.prev.push(self.clone());
        v.prev.push(rhs.clone());
        v.grad_fn = grad_fn;

        Tensor::new(v)
    }
}

/// Macro to implement binary operation traits on `Tensor`s
/// including `f64` expansion.
macro_rules! binary_op {
    [ $trait:ident, $op_name:ident, $op:tt ] => {
        impl $trait for &Tensor {
            type Output = Tensor;

            fn $op_name(self, rhs: Self) -> Self::Output {
                self.clone() $op rhs.clone()
            }
        }

        // N.B. For scalars values we could simplify the code, but
        // we would have to split on whether the operation is commutative or not.
        impl $trait<f64> for Tensor {
            type Output = Tensor;

            fn $op_name(self, rhs: f64) -> Self::Output {
                self $op <Tensor>::from_f64(rhs)
            }
        }

        impl $trait<f64> for &Tensor {
            type Output = Tensor;

            fn $op_name(self, rhs: f64) -> Self::Output {
                self.clone() $op rhs
            }
        }

        impl $trait<Tensor> for f64 {
            type Output = Tensor;

            fn $op_name(self, rhs: Tensor) -> Self::Output {
                <Tensor>::from_f64(self) $op rhs
            }
        }

        impl $trait<&Tensor> for f64 {
            type Output = Tensor;

            fn $op_name(self, rhs: &Tensor) -> Self::Output {
                self $op rhs.clone()
            }
        }
    };

    [ $trait:ident, $op_name:ident, $op:tt, $update_grad:expr ] => {
        impl $trait for Tensor {
            type Output = Tensor;

            fn $op_name(self, rhs: Self) -> Self::Output {
                let v1 = self.clone();
                let v2 = rhs.clone();

                let grad_fn = GradFn::new(stringify!($op_name), move |grad| {
                    let (dv1, dv2) = $update_grad(&grad, v1.data().deref(), v2.data().deref());
                    // If an operand is a scalar, we have to sum the individual derivative contributions together.
                    // Otherwise, the resulting gradient shape wouldn't be correct.
                    // E.g. d(2*[a,b,c])/d(2) = [a,b,c] -> a+b+c
                    let dv1 = match v1.grad().dim() {
                        (1, 1) => arr2(&[[dv1.sum()]]),
                        (_, _) => dv1,
                    };
                    let dv2 = match v2.grad().dim() {
                        (1, 1) => arr2(&[[dv2.sum()]]),
                        (_, _) => dv2,
                    };
                    v1.grad_mut().scaled_add(1.0, &dv1);
                    v2.grad_mut().scaled_add(1.0, &dv2);
                });

                let mut v = TensorData::new(self.data().deref() $op rhs.data().deref());
                v.prev.push(self.clone());
                v.prev.push(rhs.clone());
                v.grad_fn = grad_fn;

                Tensor::new(v)
            }
        }

        binary_op![$trait, $op_name, $op];
    };
}

// N.B. The purpose of `grad * 1.0` is to get a new Array2 so that we have the same result type
// in all macro cases. Otherwise, using just `grad` would give a reference &Array2.
binary_op![Add, add, +, |grad, _a, _b| { (grad * 1.0, grad * 1.0) }];
binary_op![Sub, sub, -, |grad, _a, _b| { (grad * 1.0, grad * -1.0) }];
binary_op![Mul, mul, *, |grad, a, b| { (grad * b, grad * a) }];
binary_op![Div, div, /, |grad, a, b| { (grad * 1.0 / b, grad * -1.0 * a / (b * b)) }];

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_vec<D: ndarray::Dimension>(a: &Array<f64, D>) -> Vec<f64> {
        a.iter().cloned().collect()
    }

    #[test]
    fn make_tensor() {
        let v = TensorData::new(arr2(&[[5.0]]));
        assert_eq!(flat_vec(&v.data), vec![5.0]);
    }

    #[test]
    fn add_values() {
        let v1 = Tensor::from_f64(5.0);
        let v2 = Tensor::from_f64(1.0);
        let v = &v1 + &v2;
        assert_eq!(flat_vec(&v.data()), vec![6.0]);
        assert_eq!(v.inner().grad_fn.name, "add");

        let mut v = v.inner_mut();
        v.backward();
        assert_eq!(v1.inner().grad, v.grad);
        assert_eq!(v2.grad().deref(), v.grad);

        let v1 = Tensor::from(array![[1.0, 2.0]]);
        let v2 = Tensor::from(array![[3.0, 4.0]]);
        let mut v = &v1 + &v2;
        assert_eq!(v.data().deref(), array![[4.0, 6.0]]);
        v.backward();
        assert_eq!(v1.inner().grad, v.grad().deref());
        assert_eq!(v2.grad().deref(), v.grad().deref());
    }

    #[test]
    fn multiply() {
        let v1 = Tensor::from_f64(5.0);
        let v2 = Tensor::from_f64(2.0);
        let mut v = &v1 * &v2;
        assert_eq!(flat_vec(&v.data()), vec![10.0]);
        assert_eq!(v.inner().grad_fn.name, "mul");
        v.backward();
        assert_eq!(flat_vec(&v1.grad()), vec![2.0]);
        assert_eq!(flat_vec(&v2.grad()), vec![5.0]);

        let v1 = Tensor::from(array![[2.0, 3.0]]);
        let v2 = Tensor::from(array![[4.0, 6.0]]);
        let mut v = &v1 * &v2;
        assert_eq!(flat_vec(&v.data()), vec![8.0, 18.0]);
        v.backward();
        assert_eq!(flat_vec(&v1.grad()), vec![4.0, 6.0]);
        assert_eq!(flat_vec(&v2.grad()), vec![2.0, 3.0]);
    }

    #[test]
    fn scalar_multiply() {
        let v1 = Tensor::from_f64(5.0);
        let mut v = 2.0 * &v1;
        assert_eq!(flat_vec(&v.data()), vec![10.0]);
        assert_eq!(v.inner().grad_fn.name, "mul");
        assert_eq!(v.inner().prev.len(), 2);
        v.backward();
        assert_eq!(flat_vec(&v1.grad()), vec![2.0]);

        let v1 = Tensor::from(array![[2.0, 3.0]]);
        let mut v = 2.0 * &v1;
        assert_eq!(flat_vec(&v.data()), vec![4.0, 6.0]);
        v.backward();
        assert_eq!(flat_vec(&v1.grad()), vec![2.0, 2.0]);

        let v1 = Tensor::from_f64(2.0);
        let v2 = Tensor::from(array![[2.0, 1.0]]);
        let mut v = &v2 / &v1;
        assert_eq!(flat_vec(&v.data()), vec![1.0, 0.5]);
        v.backward();
        assert_eq!(flat_vec(&v2.grad()), vec![0.5, 0.5]);
        assert_eq!(v1.grad()[[0, 0]], -0.75);
    }

    #[test]
    fn add_sub() {
        let v1 = Tensor::from_f64(5.0);
        let v2 = &v1 - 10.0;
        let v3 = &v1 - &v1;
        assert_eq!(flat_vec(&v2.data()), vec![-5.0]);
        assert_eq!(flat_vec(&v3.data()), vec![0.0]);
    }

    #[test]
    fn expr_2d() {
        let v1 = Tensor::from(array![[5.0, 6.0]]);
        let v2 = Tensor::from(array![[-3.0, -2.0]]);
        let v3 = Tensor::from_f64(0.5);
        let v4 = &v1 + &v2;
        let v5 = &v4 * &v3;
        let mut v6 = &v5 / 2.0;
        v6.backward();
        assert!(v1.grad().deref() == array![[0.25, 0.25]]);
        assert!(v2.grad().deref() == array![[0.25, 0.25]]);
        assert!(v3.grad().deref() == array![[3.0]]);
    }

    #[test]
    fn div() {
        let v1 = Tensor::from_f64(5.0);
        let mut v2 = 2.0 / &v1;
        assert_eq!(flat_vec(&v2.data()), vec![2.0 / 5.0]);
        v2.backward();
        assert_eq!(flat_vec(&v1.grad()), vec![-2.0 / 25.0]);
    }

    #[test]
    fn topological_order() {
        let v1 = Tensor::from_f64(5.0);
        let v2 = Tensor::from_f64(1.0);
        let v3 = &v1 + &v2; // 6
        let v4 = 2.0 * &v3; // 12
        let v5 = 3.0 * &v3; // 18
        let v6 = &v4 * &v5; // 216

        let order = Tensor::topological_sort(&v6);
        assert_eq!(order.len(), 8);
    }

    #[test]
    fn backward() {
        let v1 = Tensor::from_f64(5.0);
        let v2 = Tensor::from_f64(1.0);
        let v3 = &v1 + &v2; // 6
        let v4 = 2.0 * &v3; // 12
        let v5 = 3.0 * &v3; // 18
        let mut v6 = &v4 * &v5; // 216

        v6.backward();
        assert_eq!(
            v1.grad().deref(),
            12.0 * (v1.data().deref() + v2.data().deref())
        );
        assert_eq!(
            v2.grad().deref(),
            12.0 * (v1.data().deref() + v2.data().deref())
        );
    }

    #[test]
    fn relu() {
        let v1 = Tensor::from_f64(5.0);
        let mut v2 = v1.relu();
        assert_eq!(flat_vec(&v2.data()), vec![5.0]);
        v2.backward();
        assert_eq!(flat_vec(&v1.grad()), vec![1.0]);

        let v1 = Tensor::from(array![[-2.0, 2.0]]);
        let mut v2 = v1.relu();
        assert_eq!(flat_vec(&v2.data()), vec![0.0, 2.0]);
        v2.backward();
        assert_eq!(flat_vec(&v1.grad()), vec![0.0, 1.0]);
    }

    #[test]
    fn sum() {
        let v1 = Tensor::from(array![[1.0, 2.0], [3.0, 4.0]]);
        let mut v = v1.sum();
        assert_eq!(flat_vec(&v.data()), vec![10.0]);
        v.backward();
        assert_eq!(flat_vec(&v1.grad()), vec![1.0; 4]);
    }

    #[test]
    fn mean() {
        let v1 = Tensor::from(array![[1.0, 2.0], [3.0, 4.0]]);
        let mut v = v1.mean();
        assert_eq!(flat_vec(&v.data()), vec![10.0 / 4.0]);
        v.backward();
        assert_eq!(flat_vec(&v1.grad()), vec![1.0 / 4.0; 4]);
    }

    #[test]
    fn dot() {
        let v1 = Tensor::from(array![[1.0, 1.0], [2.0, 3.0]]);
        let v2 = Tensor::from(array![[0.0, 1.0], [1.0, 0.0]]);
        let mut v = v1.dot(&v2);
        assert!(v.data().deref() == array![[1.0, 1.0], [3.0, 2.0]]);
        v.backward();
        assert!(v1.grad().deref() == array![[1.0, 1.0], [1.0, 1.0]]);
        assert!(v2.grad().deref() == array![[3.0, 3.0], [4.0, 4.0]]);
    }

    #[test]
    fn to_graphviz() {
        let v1 = Tensor::from(array![[5.0, 6.0]]);
        // let mut v6 = &v1 * &v1;
        let v2 = Tensor::from(array![[-3.0, -2.0]]);
        let v3 = &v1 + &v2;
        let v4 = 2.0 * &v3;
        let v5 = 3.0 * &v3;
        let mut v6 = &v4 * &v5;
        v6.backward();

        println!("{}", v6.to_graphviz());
    }
}
