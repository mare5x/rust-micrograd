use std::{ops::{Add, Mul}, rc::Rc, cell::{RefCell, Ref, RefMut}, sync::atomic::AtomicUsize, collections::HashSet};
use std::fmt;

// TODO don't do this
// N.B. This was only necessary for topological sorting...
static VAL_CNT: AtomicUsize = AtomicUsize::new(0);

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
        Self { name: String::from(name), grad_fn: Box::new(grad_fn) }
    }

    fn empty() -> GradFn {
        Self::new("None", |_| ())
    }

    fn call(&mut self, grad: f64) {
        (self.grad_fn)(grad)
    }
}

#[derive(Debug)]
pub struct Value {
    pub data: f64,
    pub grad: f64,
    prev: Vec<WrappedValue>,
    grad_fn: GradFn,
    id: usize,  // Unique id; for sorting purposes.
}

impl Value {
    fn new(data: f64) -> Value {
        let id = VAL_CNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Value { data, grad: 0.0, prev: vec![], grad_fn: GradFn::empty(), id }
    }

    fn backward(&mut self) {
        self.grad_fn.call(self.grad)
    }
}

#[derive(Debug, Clone)]
pub struct WrappedValue(Rc<RefCell<Value>>);

impl WrappedValue {
    fn new(value: Value) -> WrappedValue {
        WrappedValue(Rc::new(RefCell::new(value)))
    }

    fn from(data: f64) -> WrappedValue {
        WrappedValue::new(Value::new(data))
    }

    fn value(&self) -> Ref<Value> {
        let v = &*self.0;
        v.borrow()
    }

    fn value_mut(&self) -> RefMut<Value> {
        let v = &*self.0;
        v.borrow_mut()
    }

    fn data(&self) -> f64 {
        let v = &*self.0;
        v.borrow().data
    }

    fn topological_sort(root: &WrappedValue) -> Vec<WrappedValue> {
        fn build(root: &WrappedValue, visited: &mut HashSet<usize>, out: &mut Vec<WrappedValue>) {
            visited.insert(root.value().id);
            for v in root.value().prev.iter() {
                if !visited.contains(&v.value().id) {
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
    fn backward(&mut self) {
        // We must first topologically sort the nodes in the computation
        // graph so that a node doesn't start backpropagating its gradient
        // before its own gradient is calculated.
        let order = Self::topological_sort(self);
        
        // d(self) w.r.t. self = 1.0
        self.value_mut().grad = 1.0;
        for v in order.iter() {
            v.value_mut().backward();
        }
    }
}

impl Add for WrappedValue {
    type Output = WrappedValue;

    fn add(self, rhs: Self) -> Self::Output {
        let v1 = self.clone();
        let v2 = rhs.clone();
        
        let grad_fn = GradFn::new("add", move |grad| {
            v1.value_mut().grad += grad;
            v2.value_mut().grad += grad;
        });
        
        let mut v = Value::new(self.data() + rhs.data());
        v.prev.push(self.clone());
        v.prev.push(rhs.clone());
        v.grad_fn = grad_fn;
        
        WrappedValue::new(v)
    }
}

impl Add for &WrappedValue {
    type Output = WrappedValue;

    fn add(self, rhs: Self) -> Self::Output {
        self.clone() + rhs.clone()
    }
}

impl Mul for WrappedValue {
    type Output = WrappedValue;

    fn mul(self, rhs: Self) -> Self::Output {
        let v1 = self.clone();
        let v2 = rhs.clone();
        
        let grad_fn = GradFn::new("mul", move |grad| {
            v1.value_mut().grad += grad * v2.data();
            v2.value_mut().grad += grad * v1.data();
        });
        
        let mut v = Value::new(self.data() * rhs.data());
        v.prev.push(self.clone());
        v.prev.push(rhs.clone());
        v.grad_fn = grad_fn;
        
        WrappedValue::new(v)
    }
}

impl Mul for &WrappedValue {
    type Output = WrappedValue;

    fn mul(self, rhs: Self) -> Self::Output {
        self.clone() * rhs.clone()
    }
}

impl Mul<WrappedValue> for f64 {
    type Output = WrappedValue;

    fn mul(self, rhs: WrappedValue) -> Self::Output {
        // N.B. A simpler but more inefficient implementation would be:
        // ```
        // let lhs = WrappedValue::from(self);
        // &lhs * &rhs
        // ```

        let v2 = rhs.clone();        
        let grad_fn = GradFn::new("smul", move |grad| {
            v2.value_mut().grad += grad * self;
        });
        
        let mut v = Value::new(self * rhs.data());
        v.prev.push(rhs.clone());
        v.grad_fn = grad_fn;
        
        WrappedValue::new(v)
    }
}

impl Mul<&WrappedValue> for f64 {
    type Output = WrappedValue;

    fn mul(self, rhs: &WrappedValue) -> Self::Output {
        self * rhs.clone()
    }
}

impl Mul<f64> for WrappedValue { 
    type Output = WrappedValue;

    fn mul(self, rhs: f64) -> Self::Output {
        rhs * self
    }
}

impl Mul<f64> for &WrappedValue {
    type Output = WrappedValue;

    fn mul(self, rhs: f64) -> Self::Output {
        rhs * self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_value() {
        let v = Value::new(5.0);
        // println!("{v:?}");
        assert_eq!(v.data, 5.0);
    }

    #[test]
    fn add_values() {
        let v1 = WrappedValue::from(5.0);
        let v2 = WrappedValue::from(1.0);
        let v = &v1 + &v2;
        assert_eq!(v.data(), 6.0);
        assert_eq!(v.value().grad_fn.name, "add");

        {
            let val = v.value();
            let prev: Vec<f64> = val.prev.iter().map(|x| x.data()).collect();
            assert_eq!(prev, vec![v1.data(), v2.data()]);
        }

        let mut v = v.value_mut();
        v.grad = 1.0;
        v.backward();
        assert_eq!(v1.value().grad, v.grad);
        assert_eq!(v2.value().grad, v.grad);
    }

    #[test]
    fn multiply() {
        let v1 = WrappedValue::from(5.0);
        let v2 = WrappedValue::from(2.0);
        let v = &v1 * &v2;
        assert_eq!(v.data(), 10.0);
        assert_eq!(v.value().grad_fn.name, "mul");
        let mut v = v.value_mut();
        v.grad = 1.0;
        v.backward();
        assert_eq!(v1.value().grad, 2.0);
        assert_eq!(v2.value().grad, 5.0);
    }

    #[test]
    fn scalar_multiply() {
        let v1 = WrappedValue::from(5.0);
        let v = 2.0 * &v1;
        assert_eq!(v.data(), 10.0);
        assert_eq!(v.value().grad_fn.name, "smul");
        assert_eq!(v.value().prev.len(), 1);
        let mut v = v.value_mut();
        v.grad = 1.0;
        v.backward();
        assert_eq!(v1.value().grad, 2.0);
    }

    #[test]
    fn topological_order() {
        let v1 = WrappedValue::from(5.0);
        let v2 = WrappedValue::from(1.0);
        let v3 = &v1 + &v2;   // 6
        let v4 = 2.0 * &v3;   // 12
        let v5 = 3.0 * &v3;   // 18
        let v6 = &v4 * &v5;   // 216

        let order = WrappedValue::topological_sort(&v6);
        let datas: Vec<f64> = order.iter().map(|x| x.data()).collect();
        assert_eq!(datas.len(), 6);
        assert_eq!(datas, vec![216.0, 18.0, 12.0, 6.0, 1.0, 5.0]);
    }

    #[test]
    fn backward() {
        let v1 = WrappedValue::from(5.0);
        let v2 = WrappedValue::from(1.0);
        let v3 = &v1 + &v2;       // 6
        let v4 = 2.0 * &v3;       // 12
        let v5 = 3.0 * &v3;       // 18
        let mut v6 = &v4 * &v5;   // 216

        v6.backward();
        assert_eq!(v1.value().grad, 12.0 * (v1.data() + v2.data()));
        assert_eq!(v2.value().grad, 12.0 * (v1.data() + v2.data()));
    }
}