use std::{
    cell::{Ref, RefCell, RefMut},
    collections::{HashMap, HashSet},
    fmt,
    ops::{Add, Div, Mul, Sub},
    rc::Rc,
    sync::atomic::AtomicUsize,
};

use ndarray_rand::rand_distr::num_traits;

// TODO don't do this
// N.B. This was only necessary for topological sorting...
static VAL_CNT: AtomicUsize = AtomicUsize::new(0);

/// Holds a closure that backward propagates the gradient to the children nodes.
/// The `name` field is only for debugging convenience.
pub struct GradFn {
    name: String,
    grad_fn: Box<dyn FnMut(f64) -> ()>,
}

impl fmt::Debug for GradFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GradFn[{}]", self.name)
    }
}

impl GradFn {
    fn new(name: &str, grad_fn: impl FnMut(f64) -> () + 'static) -> Self {
        Self {
            name: String::from(name),
            grad_fn: Box::new(grad_fn),
        }
    }
}

/// Underlying `Value` data holding single `f64` values for the data and gradient.
/// # Notes
/// - This struct is wrapped into `Value` with `Rc<RefCell<_>>` because
/// we are building a computation graph/tree.
#[derive(Debug)]
pub struct ValueData {
    pub data: f64,
    pub grad: f64,
    prev: Vec<Value>,
    grad_fn: Option<GradFn>,
    id: usize, // Unique id; for sorting purposes.
}

impl ValueData {
    fn new(data: f64) -> ValueData {
        let id = VAL_CNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        ValueData {
            data,
            grad: 0.0,
            prev: vec![],
            grad_fn: None,
            id,
        }
    }

    /// Initiate local backward pass computation, updating the `grad`
    /// of the children in the node's computation graph.
    fn backward(&mut self) {
        if let Some(grad_fn) = &mut self.grad_fn {
            (grad_fn.grad_fn)(self.grad)
        }
    }
}

/// `ValueData` wrapper that builds a computation graph for gradient backpropagation.
/// # Notes
/// - `Value::Clone` is used to make a clone of the pointer to the underlying data.
#[derive(Debug, Clone)]
pub struct Value(Rc<RefCell<ValueData>>);

impl Value {
    fn new(value: ValueData) -> Value {
        Value(Rc::new(RefCell::new(value)))
    }

    pub fn from(data: f64) -> Value {
        Value::new(ValueData::new(data))
    }

    pub fn from_ndarray<D: ndarray::Dimension>(
        arr: &ndarray::Array<f64, D>,
    ) -> ndarray::Array<Value, D> {
        arr.map(|x| Self::from(*x))
    }

    pub fn inner(&self) -> Ref<ValueData> {
        (*self.0).borrow()
    }

    pub fn inner_mut(&self) -> RefMut<ValueData> {
        (*self.0).borrow_mut()
    }

    pub fn data(&self) -> f64 {
        (*self.0).borrow().data
    }

    pub fn grad(&self) -> f64 {
        (*self.0).borrow().grad
    }

    /// Create a GraphViz DOT format string representation of the computation graph.
    pub fn to_graphviz(&self) -> String {
        let mut gradfn_colors = HashMap::new();
        gradfn_colors.insert("add", 1);
        gradfn_colors.insert("sub", 1);
        gradfn_colors.insert("mul", 2);
        gradfn_colors.insert("div", 2);
        gradfn_colors.insert("relu", 3);

        fn inner(node: &Value, colors: &HashMap<&str, i32>) -> String {
            let id = node.inner().id;
            let mut s = format!(
                "{} [label=\"{{{} {:.2} | {:.2}}}\", color={}];\n",
                id,
                node.inner()
                    .grad_fn
                    .as_ref()
                    .map_or(String::from(""), |g| format!("{} | ", g.name)),
                node.data(),
                node.grad(),
                node.inner()
                    .grad_fn
                    .as_ref()
                    .map_or(0, |g| *colors.get(&g.name.as_str()).unwrap_or(&0)),
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

    fn topological_sort(root: &Value) -> Vec<Value> {
        fn build(root: &Value, visited: &mut HashSet<usize>, out: &mut Vec<Value>) {
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
        self.inner_mut().grad = 1.0;
        for v in order.iter() {
            v.inner_mut().backward();
        }
    }
}

/// Value operations supporting `backward`.
impl Value {
    pub fn relu(&self) -> Value {
        let v = self.clone();
        let x = self.data();

        let grad_fn = GradFn::new("relu", move |grad| {
            let dv = if x > 0.0 { 1.0 } else { 0.0 };
            v.inner_mut().grad += grad * dv;
        });

        let mut v = ValueData::new(if x > 0.0 { x } else { 0.0 });
        v.prev.push(self.clone());
        v.grad_fn = Some(grad_fn);

        Value::new(v)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value({}, grad={})", self.data(), self.grad())
    }
}

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

// N.B. Required for array.mean
impl num_traits::FromPrimitive for Value {
    fn from_i64(n: i64) -> Option<Self> {
        Some(Value::from(n as f64))
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(Value::from(n as f64))
    }
}

/// Macro to implement binary operation traits on `Value`s
/// including `f64` expansion.
macro_rules! binary_op {
    [ $trait:ident, $op_name:ident, $op:tt ] => {
        impl $trait for &Value {
            type Output = Value;

            fn $op_name(self, rhs: Self) -> Self::Output {
                self.clone() $op rhs.clone()
            }
        }

        // N.B. For scalars values we could simplify the code, but
        // we would have to split on whether the operation is commutative or not.
        impl $trait<f64> for Value {
            type Output = Value;

            fn $op_name(self, rhs: f64) -> Self::Output {
                self $op <Value>::from(rhs)
            }
        }

        impl $trait<f64> for &Value {
            type Output = Value;

            fn $op_name(self, rhs: f64) -> Self::Output {
                self.clone() $op rhs
            }
        }

        impl $trait<Value> for f64 {
            type Output = Value;

            fn $op_name(self, rhs: Value) -> Self::Output {
                <Value>::from(self) $op rhs
            }
        }

        impl $trait<&Value> for f64 {
            type Output = Value;

            fn $op_name(self, rhs: &Value) -> Self::Output {
                self $op rhs.clone()
            }
        }
    };

    [ $trait:ident, $op_name:ident, $op:tt, $update_grad:expr ] => {
        impl $trait for Value {
            type Output = Value;

            fn $op_name(self, rhs: Self) -> Self::Output {
                let v1 = self.clone();
                let v2 = rhs.clone();

                let grad_fn = GradFn::new(stringify!($op_name), move |grad| {
                    let (dv1, dv2) = $update_grad(grad, v1.data(), v2.data());
                    v1.inner_mut().grad += dv1;
                    v2.inner_mut().grad += dv2;
                });

                let mut v = ValueData::new(self.data() $op rhs.data());
                v.prev.push(self.clone());
                v.prev.push(rhs.clone());
                v.grad_fn = Some(grad_fn);

                Value::new(v)
            }
        }

        binary_op![$trait, $op_name, $op];
    };
}

binary_op![Add, add, +, |grad, _a,_b| { (grad, grad) }];
binary_op![Sub, sub, -, |grad, _a,_b| { (grad, grad * -1.0) }];
binary_op![Mul, mul, *, |grad, a, b| { (grad * b, grad * a) }];
binary_op![Div, div, /, |grad, a, b| { (grad * 1.0 / b, grad * -1.0 * a / (b * b)) }];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_data() {
        let v = ValueData::new(5.0);
        assert_eq!(v.data, 5.0);
    }

    #[test]
    fn add_values() {
        let v1 = Value::from(5.0);
        let v2 = Value::from(1.0);
        let v = &v1 + &v2;
        assert_eq!(v.data(), 6.0);
        assert_eq!(v.inner().grad_fn.as_ref().map_or("", |g| &g.name), "add");

        {
            let val = v.inner();
            let prev: Vec<f64> = val.prev.iter().map(|x| x.data()).collect();
            assert_eq!(prev, vec![v1.data(), v2.data()]);
        }

        let mut v = v.inner_mut();
        v.grad = 1.0;
        v.backward();
        assert_eq!(v1.inner().grad, v.grad);
        assert_eq!(v2.grad(), v.grad);
    }

    #[test]
    fn multiply() {
        let v1 = Value::from(5.0);
        let v2 = Value::from(2.0);
        let v = &v1 * &v2;
        assert_eq!(v.data(), 10.0);
        assert_eq!(v.inner().grad_fn.as_ref().map_or("", |g| &g.name), "mul");
        let mut v = v.inner_mut();
        v.grad = 1.0;
        v.backward();
        assert_eq!(v1.grad(), 2.0);
        assert_eq!(v2.grad(), 5.0);
    }

    #[test]
    fn scalar_multiply() {
        let v1 = Value::from(5.0);
        let v = 2.0 * &v1;
        assert_eq!(v.data(), 10.0);
        assert_eq!(v.inner().grad_fn.as_ref().map_or("", |g| &g.name), "mul");
        assert_eq!(v.inner().prev.len(), 2);
        let mut v = v.inner_mut();
        v.grad = 1.0;
        v.backward();
        assert_eq!(v1.grad(), 2.0);
    }

    #[test]
    fn add_sub() {
        let v1 = Value::from(5.0);
        let v2 = &v1 - 10.0;
        let v3 = &v1 - &v1;
        assert_eq!(v2.data(), -5.0);
        assert_eq!(v3.data(), 0.0);
    }

    #[test]
    fn div() {
        let v1 = Value::from(5.0);
        let mut v2 = 2.0 / &v1;
        assert_eq!(v2.data(), 2.0 / 5.0);
        v2.backward();
        assert_eq!(v1.grad(), -2.0 / 25.0);
    }

    #[test]
    fn topological_order() {
        let v1 = Value::from(5.0);
        let v2 = Value::from(1.0);
        let v3 = &v1 + &v2; // 6
        let v4 = 2.0 * &v3; // 12
        let v5 = 3.0 * &v3; // 18
        let v6 = &v4 * &v5; // 216

        let order = Value::topological_sort(&v6);
        let datas: Vec<f64> = order.iter().map(|x| x.data()).collect();
        assert_eq!(datas.len(), 8);
        assert_eq!(datas, vec![216.0, 18.0, 3.0, 12.0, 6.0, 1.0, 5.0, 2.0]);
    }

    #[test]
    fn backward() {
        let v1 = Value::from(5.0);
        let v2 = Value::from(1.0);
        let v3 = &v1 + &v2; // 6
        let v4 = 2.0 * &v3; // 12
        let v5 = 3.0 * &v3; // 18
        let mut v6 = &v4 * &v5; // 216

        v6.backward();
        assert_eq!(v1.grad(), 12.0 * (v1.data() + v2.data()));
        assert_eq!(v2.grad(), 12.0 * (v1.data() + v2.data()));
    }

    #[test]
    fn relu() {
        let v1 = Value::from(5.0);
        let mut v2 = v1.relu();
        assert_eq!(v2.data(), v1.data());

        v2.backward();
        assert_eq!(v1.grad(), 1.0);
    }

    #[test]
    fn expr1() {
        let v1 = Value::from(5.0);
        let v2 = Value::from(1.0);
        let v3 = &v1 + &v2;
        let v4 = 2.0 * &v3;
        let mut v5 = v4.relu();
        v5.backward();
        assert_eq!(v1.grad(), 2.0);
        assert_eq!(v2.grad(), 2.0);
    }

    #[test]
    fn to_graphviz() {
        let v1 = Value::from(5.0);
        let mut v6 = &v1 * &v1;
        // let v2 = Value::from(1.0);
        // let v3 = &v1 + &v2; // 6
        // let v4 = 2.0 * &v3; // 12
        // let v5 = 3.0 * &v3; // 18
        // let mut v6 = &v4 * &v5; // 216
        v6.backward();

        println!("{}", v6.to_graphviz());
    }

    #[test]
    fn to_graphviz_2() {
        let v1 = Value::from(5.0);
        let v2 = Value::from(1.0);
        let v3 = &v1 + &v2;
        let v4 = 2.0 * &v3;
        let mut v5 = v4.relu();
        v5.backward();
        println!("{}", v5.to_graphviz());
    }
}
