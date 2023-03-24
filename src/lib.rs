use std::{ops::Add, rc::Rc, cell::{RefCell, Ref}};

#[derive(Debug)]
pub struct Value {
    pub data: f64,
    pub grad: f64,
    prev: Vec<WrappedValue>,
}

impl Value {
    fn new(data: f64) -> Value {
        Value { data, grad: 0.0, prev: vec![] }
    }

    fn backward(&mut self) {

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

    fn data(&self) -> f64 {
        let v = &*self.0;
        v.borrow().data
    }
}

impl Add for WrappedValue {
    type Output = WrappedValue;

    fn add(self, rhs: Self) -> Self::Output {
        let v1 = self.clone();
        let v2 = rhs.clone();

        let mut v = Value::new(self.data() + rhs.data());
        v.prev.push(v1);
        v.prev.push(v2);

        WrappedValue::new(v)
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
        let v = v1.clone() + v2.clone();
        assert_eq!(v.data(), 6.0);

        let v = v.value();
        let prev: Vec<f64> = v.prev.iter().map(|x| x.data()).collect();
        assert_eq!(prev, vec![v1.data(), v2.data()]);
    }


}