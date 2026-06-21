//! The autograd engine.
//!
//! This module implements a small but complete reverse-mode automatic
//! differentiation system over 2-D tensors (matrices). Every [`Tensor`] is a
//! node in a dynamically built computation graph. Running an operation records
//! the operation and its inputs; calling [`Tensor::backward`] walks the graph
//! in reverse topological order and accumulates gradients into every node.
//!
//! The design is deliberately dependency-free and easy to read: the graph is
//! built out of `Rc<RefCell<..>>` nodes and the backward rule for each
//! operation is a plain `match` arm rather than a boxed closure, which keeps
//! the graph acyclic and leak-free.

use std::cell::{Ref, RefCell, RefMut};
use std::collections::HashSet;
use std::fmt;
use std::rc::Rc;

/// The shape of a 2-D tensor as `(rows, cols)`.
pub type Shape = (usize, usize);

/// The operation that produced a tensor. Used to drive the backward pass.
#[derive(Clone)]
pub enum Op {
    /// A leaf node (input data or a learnable parameter).
    Leaf,
    Add,
    Sub,
    /// Element-wise multiply (Hadamard product), with broadcasting.
    Mul,
    /// Multiply every element by a scalar constant.
    ScalarMul(f32),
    /// Add a scalar constant to every element.
    ScalarAdd(f32),
    MatMul,
    ReLU,
    Tanh,
    Sigmoid,
    Exp,
    /// Mean of all elements, producing a `(1, 1)` scalar.
    Mean,
    /// Sum of all elements, producing a `(1, 1)` scalar.
    Sum,
    /// Fused softmax + cross-entropy loss. Holds the target class index for
    /// each row. Produces a `(1, 1)` scalar (the mean loss over the batch).
    SoftmaxCrossEntropy(Rc<Vec<usize>>),
}

/// The data backing a single tensor / graph node.
pub struct TensorData {
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
    pub shape: Shape,
    pub op: Op,
    pub parents: Vec<Tensor>,
    pub requires_grad: bool,
}

/// A reference-counted handle to a node in the computation graph.
///
/// Cloning a `Tensor` is cheap: it clones the `Rc`, so both handles point at
/// the same underlying data and gradient buffers.
#[derive(Clone)]
pub struct Tensor(pub Rc<RefCell<TensorData>>);

// ---------------------------------------------------------------------------
// Broadcasting helpers
// ---------------------------------------------------------------------------

/// The broadcasted output shape of two operands. A dimension of size 1 in one
/// operand stretches to match the other.
fn broadcast_shape(a: Shape, b: Shape) -> Shape {
    let rows = a.0.max(b.0);
    let cols = a.1.max(b.1);
    debug_assert!(a.0 == rows || a.0 == 1, "rows not broadcastable");
    debug_assert!(b.0 == rows || b.0 == 1, "rows not broadcastable");
    debug_assert!(a.1 == cols || a.1 == 1, "cols not broadcastable");
    debug_assert!(b.1 == cols || b.1 == 1, "cols not broadcastable");
    (rows, cols)
}

/// Read element `(i, j)` of `data` interpreting size-1 dims as broadcast.
#[inline]
fn bget(data: &[f32], shape: Shape, i: usize, j: usize) -> f32 {
    let ii = if shape.0 == 1 { 0 } else { i };
    let jj = if shape.1 == 1 { 0 } else { j };
    data[ii * shape.1 + jj]
}

/// Accumulate `val` into element `(i, j)` of `grad`, folding broadcast dims.
#[inline]
fn bacc(grad: &mut [f32], shape: Shape, i: usize, j: usize, val: f32) {
    let ii = if shape.0 == 1 { 0 } else { i };
    let jj = if shape.1 == 1 { 0 } else { j };
    grad[ii * shape.1 + jj] += val;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

impl Tensor {
    /// Create a leaf tensor from raw data and a shape.
    pub fn new(data: Vec<f32>, shape: Shape, requires_grad: bool) -> Tensor {
        assert_eq!(
            data.len(),
            shape.0 * shape.1,
            "data length {} does not match shape {:?}",
            data.len(),
            shape
        );
        let n = data.len();
        Tensor(Rc::new(RefCell::new(TensorData {
            data,
            grad: vec![0.0; n],
            shape,
            op: Op::Leaf,
            parents: Vec::new(),
            requires_grad,
        })))
    }

    /// A tensor filled with zeros.
    pub fn zeros(shape: Shape, requires_grad: bool) -> Tensor {
        Tensor::new(vec![0.0; shape.0 * shape.1], shape, requires_grad)
    }

    /// A tensor filled with a constant.
    pub fn full(shape: Shape, value: f32, requires_grad: bool) -> Tensor {
        Tensor::new(vec![value; shape.0 * shape.1], shape, requires_grad)
    }

    /// Internal constructor for a node produced by an operation.
    fn from_op(data: Vec<f32>, shape: Shape, op: Op, parents: Vec<Tensor>) -> Tensor {
        let n = data.len();
        Tensor(Rc::new(RefCell::new(TensorData {
            data,
            grad: vec![0.0; n],
            shape,
            op,
            parents,
            requires_grad: true,
        })))
    }

    // -- Accessors ---------------------------------------------------------

    pub fn borrow(&self) -> Ref<'_, TensorData> {
        self.0.borrow()
    }

    pub fn borrow_mut(&self) -> RefMut<'_, TensorData> {
        self.0.borrow_mut()
    }

    pub fn shape(&self) -> Shape {
        self.0.borrow().shape
    }

    pub fn rows(&self) -> usize {
        self.0.borrow().shape.0
    }

    pub fn cols(&self) -> usize {
        self.0.borrow().shape.1
    }

    /// A copy of the underlying data buffer.
    pub fn data(&self) -> Vec<f32> {
        self.0.borrow().data.clone()
    }

    /// A copy of the gradient buffer.
    pub fn grad(&self) -> Vec<f32> {
        self.0.borrow().grad.clone()
    }

    /// For a `(1, 1)` scalar tensor, return its value.
    pub fn item(&self) -> f32 {
        let d = self.0.borrow();
        assert_eq!(d.data.len(), 1, "item() requires a scalar tensor");
        d.data[0]
    }

    /// Set the gradient buffer to all zeros.
    pub fn zero_grad(&self) {
        let mut d = self.0.borrow_mut();
        for g in d.grad.iter_mut() {
            *g = 0.0;
        }
    }

    /// Pointer identity, used to deduplicate nodes during graph traversal.
    fn id(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }

    // -- Element-wise & linear-algebra ops ---------------------------------

    /// Element-wise addition with broadcasting.
    pub fn add(&self, other: &Tensor) -> Tensor {
        let (a, b) = (self.0.borrow(), other.0.borrow());
        let out_shape = broadcast_shape(a.shape, b.shape);
        let mut data = vec![0.0; out_shape.0 * out_shape.1];
        for i in 0..out_shape.0 {
            for j in 0..out_shape.1 {
                data[i * out_shape.1 + j] =
                    bget(&a.data, a.shape, i, j) + bget(&b.data, b.shape, i, j);
            }
        }
        drop((a, b));
        Tensor::from_op(data, out_shape, Op::Add, vec![self.clone(), other.clone()])
    }

    /// Element-wise subtraction with broadcasting.
    pub fn sub(&self, other: &Tensor) -> Tensor {
        let (a, b) = (self.0.borrow(), other.0.borrow());
        let out_shape = broadcast_shape(a.shape, b.shape);
        let mut data = vec![0.0; out_shape.0 * out_shape.1];
        for i in 0..out_shape.0 {
            for j in 0..out_shape.1 {
                data[i * out_shape.1 + j] =
                    bget(&a.data, a.shape, i, j) - bget(&b.data, b.shape, i, j);
            }
        }
        drop((a, b));
        Tensor::from_op(data, out_shape, Op::Sub, vec![self.clone(), other.clone()])
    }

    /// Element-wise (Hadamard) multiplication with broadcasting.
    pub fn mul(&self, other: &Tensor) -> Tensor {
        let (a, b) = (self.0.borrow(), other.0.borrow());
        let out_shape = broadcast_shape(a.shape, b.shape);
        let mut data = vec![0.0; out_shape.0 * out_shape.1];
        for i in 0..out_shape.0 {
            for j in 0..out_shape.1 {
                data[i * out_shape.1 + j] =
                    bget(&a.data, a.shape, i, j) * bget(&b.data, b.shape, i, j);
            }
        }
        drop((a, b));
        Tensor::from_op(data, out_shape, Op::Mul, vec![self.clone(), other.clone()])
    }

    /// Multiply every element by a scalar.
    pub fn scalar_mul(&self, s: f32) -> Tensor {
        let a = self.0.borrow();
        let data: Vec<f32> = a.data.iter().map(|x| x * s).collect();
        let shape = a.shape;
        drop(a);
        Tensor::from_op(data, shape, Op::ScalarMul(s), vec![self.clone()])
    }

    /// Add a scalar to every element.
    pub fn scalar_add(&self, s: f32) -> Tensor {
        let a = self.0.borrow();
        let data: Vec<f32> = a.data.iter().map(|x| x + s).collect();
        let shape = a.shape;
        drop(a);
        Tensor::from_op(data, shape, Op::ScalarAdd(s), vec![self.clone()])
    }

    /// Negate every element.
    pub fn neg(&self) -> Tensor {
        self.scalar_mul(-1.0)
    }

    /// Matrix multiplication: `(m, k) x (k, n) -> (m, n)`.
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let (a, b) = (self.0.borrow(), other.0.borrow());
        let (m, k) = a.shape;
        let (k2, n) = b.shape;
        assert_eq!(
            k, k2,
            "matmul shape mismatch: {:?} x {:?}",
            a.shape, b.shape
        );
        let mut data = vec![0.0; m * n];
        for i in 0..m {
            for p in 0..k {
                let aip = a.data[i * k + p];
                if aip == 0.0 {
                    continue;
                }
                for j in 0..n {
                    data[i * n + j] += aip * b.data[p * n + j];
                }
            }
        }
        drop((a, b));
        Tensor::from_op(data, (m, n), Op::MatMul, vec![self.clone(), other.clone()])
    }

    // -- Activations -------------------------------------------------------

    pub fn relu(&self) -> Tensor {
        let a = self.0.borrow();
        let data: Vec<f32> = a.data.iter().map(|&x| x.max(0.0)).collect();
        let shape = a.shape;
        drop(a);
        Tensor::from_op(data, shape, Op::ReLU, vec![self.clone()])
    }

    pub fn tanh(&self) -> Tensor {
        let a = self.0.borrow();
        let data: Vec<f32> = a.data.iter().map(|&x| x.tanh()).collect();
        let shape = a.shape;
        drop(a);
        Tensor::from_op(data, shape, Op::Tanh, vec![self.clone()])
    }

    pub fn sigmoid(&self) -> Tensor {
        let a = self.0.borrow();
        let data: Vec<f32> = a.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        let shape = a.shape;
        drop(a);
        Tensor::from_op(data, shape, Op::Sigmoid, vec![self.clone()])
    }

    pub fn exp(&self) -> Tensor {
        let a = self.0.borrow();
        let data: Vec<f32> = a.data.iter().map(|&x| x.exp()).collect();
        let shape = a.shape;
        drop(a);
        Tensor::from_op(data, shape, Op::Exp, vec![self.clone()])
    }

    // -- Reductions --------------------------------------------------------

    /// Mean of all elements, as a `(1, 1)` scalar tensor.
    pub fn mean(&self) -> Tensor {
        let a = self.0.borrow();
        let sum: f32 = a.data.iter().sum();
        let mean = sum / a.data.len() as f32;
        drop(a);
        Tensor::from_op(vec![mean], (1, 1), Op::Mean, vec![self.clone()])
    }

    /// Sum of all elements, as a `(1, 1)` scalar tensor.
    pub fn sum(&self) -> Tensor {
        let a = self.0.borrow();
        let sum: f32 = a.data.iter().sum();
        drop(a);
        Tensor::from_op(vec![sum], (1, 1), Op::Sum, vec![self.clone()])
    }

    /// Mean squared error against a (non-differentiated) target tensor.
    pub fn mse_loss(&self, target: &Tensor) -> Tensor {
        let diff = self.sub(target);
        diff.mul(&diff).mean()
    }

    /// Fused, numerically stable softmax cross-entropy.
    ///
    /// `self` are the raw logits of shape `(batch, classes)`. `targets` holds
    /// the correct class index for each row. Returns the mean loss as a scalar.
    pub fn softmax_cross_entropy(&self, targets: &[usize]) -> Tensor {
        let a = self.0.borrow();
        let (m, n) = a.shape;
        assert_eq!(targets.len(), m, "one target per row required");
        let mut total = 0.0f32;
        for i in 0..m {
            let row = &a.data[i * n..(i + 1) * n];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0;
            for &v in row {
                sum_exp += (v - max).exp();
            }
            let log_sum_exp = max + sum_exp.ln();
            let t = targets[i];
            assert!(t < n, "target class out of range");
            total += log_sum_exp - row[t];
        }
        let loss = total / m as f32;
        drop(a);
        Tensor::from_op(
            vec![loss],
            (1, 1),
            Op::SoftmaxCrossEntropy(Rc::new(targets.to_vec())),
            vec![self.clone()],
        )
    }

    // -- Backward pass -----------------------------------------------------

    /// Run reverse-mode autodiff from this (scalar) tensor, populating the
    /// `grad` field of every tensor that contributed to it.
    pub fn backward(&self) {
        // Build a topological ordering of the graph via post-order DFS.
        let mut topo: Vec<Tensor> = Vec::new();
        let mut visited: HashSet<usize> = HashSet::new();
        build_topo(self, &mut visited, &mut topo);

        // Seed the output gradient with ones (d(self)/d(self) = 1).
        {
            let mut s = self.0.borrow_mut();
            for g in s.grad.iter_mut() {
                *g = 1.0;
            }
        }

        // Propagate gradients in reverse topological order.
        for node in topo.iter().rev() {
            node.backward_step();
        }
    }

    /// Push this node's gradient to its parents according to its op.
    fn backward_step(&self) {
        let node = self.0.borrow();
        let parents = node.parents.clone();
        match &node.op {
            Op::Leaf => {}

            Op::Add => {
                let (a, b) = (&parents[0], &parents[1]);
                let (as_, bs_) = (a.shape(), b.shape());
                let mut ag = a.borrow_mut();
                let mut bg = b.borrow_mut();
                for i in 0..node.shape.0 {
                    for j in 0..node.shape.1 {
                        let g = node.grad[i * node.shape.1 + j];
                        bacc(&mut ag.grad, as_, i, j, g);
                        bacc(&mut bg.grad, bs_, i, j, g);
                    }
                }
            }

            Op::Sub => {
                let (a, b) = (&parents[0], &parents[1]);
                let (as_, bs_) = (a.shape(), b.shape());
                let mut ag = a.borrow_mut();
                let mut bg = b.borrow_mut();
                for i in 0..node.shape.0 {
                    for j in 0..node.shape.1 {
                        let g = node.grad[i * node.shape.1 + j];
                        bacc(&mut ag.grad, as_, i, j, g);
                        bacc(&mut bg.grad, bs_, i, j, -g);
                    }
                }
            }

            Op::Mul => {
                let (a, b) = (&parents[0], &parents[1]);
                // Snapshot values first to stay sound even when a and b alias.
                let (adata, ashape) = {
                    let ad = a.borrow();
                    (ad.data.clone(), ad.shape)
                };
                let (bdata, bshape) = {
                    let bd = b.borrow();
                    (bd.data.clone(), bd.shape)
                };
                {
                    let mut ag = a.borrow_mut();
                    for i in 0..node.shape.0 {
                        for j in 0..node.shape.1 {
                            let g = node.grad[i * node.shape.1 + j];
                            bacc(&mut ag.grad, ashape, i, j, g * bget(&bdata, bshape, i, j));
                        }
                    }
                }
                {
                    let mut bg = b.borrow_mut();
                    for i in 0..node.shape.0 {
                        for j in 0..node.shape.1 {
                            let g = node.grad[i * node.shape.1 + j];
                            bacc(&mut bg.grad, bshape, i, j, g * bget(&adata, ashape, i, j));
                        }
                    }
                }
            }

            Op::ScalarMul(s) => {
                let a = &parents[0];
                let mut ag = a.borrow_mut();
                for (gi, &g) in ag.grad.iter_mut().zip(node.grad.iter()) {
                    *gi += g * s;
                }
            }

            Op::ScalarAdd(_) => {
                let a = &parents[0];
                let mut ag = a.borrow_mut();
                for (gi, &g) in ag.grad.iter_mut().zip(node.grad.iter()) {
                    *gi += g;
                }
            }

            Op::MatMul => {
                // C = A (m,k) x B (k,n). dA = dC x B^T, dB = A^T x dC.
                let (a, b) = (&parents[0], &parents[1]);
                let (adata, ashape) = {
                    let ad = a.borrow();
                    (ad.data.clone(), ad.shape)
                };
                let (bdata, bshape) = {
                    let bd = b.borrow();
                    (bd.data.clone(), bd.shape)
                };
                let (m, k) = ashape;
                let n = bshape.1;
                {
                    let mut ag = a.borrow_mut();
                    for i in 0..m {
                        for p in 0..k {
                            let mut acc = 0.0;
                            for j in 0..n {
                                acc += node.grad[i * n + j] * bdata[p * n + j];
                            }
                            ag.grad[i * k + p] += acc;
                        }
                    }
                }
                {
                    let mut bg = b.borrow_mut();
                    for p in 0..k {
                        for j in 0..n {
                            let mut acc = 0.0;
                            for i in 0..m {
                                acc += adata[i * k + p] * node.grad[i * n + j];
                            }
                            bg.grad[p * n + j] += acc;
                        }
                    }
                }
            }

            Op::ReLU => {
                let a = &parents[0];
                let mut ag = a.borrow_mut();
                for idx in 0..node.data.len() {
                    if node.data[idx] > 0.0 {
                        ag.grad[idx] += node.grad[idx];
                    }
                }
            }

            Op::Tanh => {
                let a = &parents[0];
                let mut ag = a.borrow_mut();
                for idx in 0..node.data.len() {
                    let y = node.data[idx];
                    ag.grad[idx] += node.grad[idx] * (1.0 - y * y);
                }
            }

            Op::Sigmoid => {
                let a = &parents[0];
                let mut ag = a.borrow_mut();
                for idx in 0..node.data.len() {
                    let y = node.data[idx];
                    ag.grad[idx] += node.grad[idx] * y * (1.0 - y);
                }
            }

            Op::Exp => {
                let a = &parents[0];
                let mut ag = a.borrow_mut();
                for idx in 0..node.data.len() {
                    ag.grad[idx] += node.grad[idx] * node.data[idx];
                }
            }

            Op::Sum => {
                let a = &parents[0];
                let g = node.grad[0];
                let mut ag = a.borrow_mut();
                for gi in ag.grad.iter_mut() {
                    *gi += g;
                }
            }

            Op::Mean => {
                let a = &parents[0];
                let mut ag = a.borrow_mut();
                let scale = node.grad[0] / ag.grad.len() as f32;
                for gi in ag.grad.iter_mut() {
                    *gi += scale;
                }
            }

            Op::SoftmaxCrossEntropy(targets) => {
                // d/dlogits = (softmax - onehot) / batch * upstream_grad.
                let a = &parents[0];
                let (adata, ashape) = {
                    let ad = a.borrow();
                    (ad.data.clone(), ad.shape)
                };
                let (m, n) = ashape;
                let upstream = node.grad[0];
                let mut ag = a.borrow_mut();
                for i in 0..m {
                    let row = &adata[i * n..(i + 1) * n];
                    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_exp = 0.0;
                    for &v in row {
                        sum_exp += (v - max).exp();
                    }
                    for j in 0..n {
                        let soft = (row[j] - max).exp() / sum_exp;
                        let onehot = if targets[i] == j { 1.0 } else { 0.0 };
                        ag.grad[i * n + j] += upstream * (soft - onehot) / m as f32;
                    }
                }
            }
        }
    }
}

/// Post-order DFS that appends each node after its parents.
fn build_topo(node: &Tensor, visited: &mut HashSet<usize>, topo: &mut Vec<Tensor>) {
    if !visited.insert(node.id()) {
        return;
    }
    let parents = node.0.borrow().parents.clone();
    for p in &parents {
        build_topo(p, visited, topo);
    }
    topo.push(node.clone());
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let d = self.0.borrow();
        write!(f, "Tensor(shape={:?}, data={:?})", d.shape, d.data)
    }
}
