#[derive(Clone, Copy)]
enum TapeOp {
    Constant,
    Parameter(usize),
    Add(usize, usize),
    Mul(usize, usize),
    Div(usize, usize),
    Tanh(usize),
    Exp(usize),
    Sqrt(usize),
    Sin(usize),
    Sigmoid(usize),
    Max(usize, usize),
    StopGradient(usize),
}

#[derive(Clone, Copy)]
struct TapeNode {
    value: f64,
    op: TapeOp,
}

struct AutodiffTape {
    nodes: RefCell<Vec<TapeNode>>,
    deferred: bool,
}

impl AutodiffTape {
    fn new() -> Self {
        Self {
            nodes: RefCell::new(Vec::new()),
            deferred: false,
        }
    }
    fn deferred() -> Self {
        Self {
            nodes: RefCell::new(Vec::new()),
            deferred: true,
        }
    }
    fn constant(&self, value: f64) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        nodes.push(TapeNode {
            value,
            op: TapeOp::Constant,
        });
        index
    }
    fn parameter(&self, parameter: usize, value: f64) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        nodes.push(TapeNode {
            value,
            op: TapeOp::Parameter(parameter),
        });
        index
    }
    fn add(&self, left: usize, right: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        let value = if self.deferred {
            0.0
        } else {
            nodes[left].value + nodes[right].value
        };
        nodes.push(TapeNode {
            value,
            op: TapeOp::Add(left, right),
        });
        index
    }
    fn mul(&self, left: usize, right: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        let value = if self.deferred {
            0.0
        } else {
            nodes[left].value * nodes[right].value
        };
        nodes.push(TapeNode {
            value,
            op: TapeOp::Mul(left, right),
        });
        index
    }
    fn div(&self, left: usize, right: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        let value = if self.deferred {
            0.0
        } else {
            nodes[left].value / nodes[right].value.max(1e-12)
        };
        nodes.push(TapeNode {
            value,
            op: TapeOp::Div(left, right),
        });
        index
    }
    fn tanh(&self, input: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        let value = if self.deferred {
            0.0
        } else {
            nodes[input].value.tanh()
        };
        nodes.push(TapeNode {
            value,
            op: TapeOp::Tanh(input),
        });
        index
    }
    fn exp(&self, input: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        let value = if self.deferred {
            0.0
        } else {
            nodes[input].value.exp()
        };
        nodes.push(TapeNode {
            value,
            op: TapeOp::Exp(input),
        });
        index
    }
    fn sqrt(&self, input: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        let value = if self.deferred {
            0.0
        } else {
            nodes[input].value.max(1e-12).sqrt()
        };
        nodes.push(TapeNode {
            value,
            op: TapeOp::Sqrt(input),
        });
        index
    }
    fn sin(&self, input: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        let value = if self.deferred {
            0.0
        } else {
            nodes[input].value.sin()
        };
        nodes.push(TapeNode {
            value,
            op: TapeOp::Sin(input),
        });
        index
    }
    fn sigmoid(&self, input: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        let value = if self.deferred {
            0.0
        } else {
            sigmoid(nodes[input].value)
        };
        nodes.push(TapeNode {
            value,
            op: TapeOp::Sigmoid(input),
        });
        index
    }
    fn max(&self, left: usize, right: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        let value = if self.deferred {
            0.0
        } else {
            nodes[left].value.max(nodes[right].value)
        };
        nodes.push(TapeNode {
            value,
            op: TapeOp::Max(left, right),
        });
        index
    }
    fn stop_gradient(&self, input: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        let value = if self.deferred {
            0.0
        } else {
            nodes[input].value
        };
        nodes.push(TapeNode {
            value,
            op: TapeOp::StopGradient(input),
        });
        index
    }
    fn accelerated_values(&self, selection: &BackendSelection) -> Result<Vec<f32>> {
        let (initial_values, opcodes, left, right, _) = self.accelerator_arrays();
        backend_scalar_graph_f32(selection, &initial_values, &opcodes, &left, &right)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }
    fn accelerator_arrays(&self) -> AcceleratorGraphArrays {
        let nodes = self.nodes.borrow();
        let mut initial_values = Vec::with_capacity(nodes.len());
        let mut opcodes = Vec::with_capacity(nodes.len());
        let mut left = Vec::with_capacity(nodes.len());
        let mut right = Vec::with_capacity(nodes.len());
        let mut parameter_ids = Vec::with_capacity(nodes.len());
        for node in nodes.iter() {
            initial_values.push(node.value as f32);
            let (opcode, lhs, rhs, parameter) = match node.op {
                TapeOp::Constant => (0, 0, 0, u32::MAX),
                TapeOp::Parameter(parameter) => (1, 0, 0, parameter as u32),
                TapeOp::Add(lhs, rhs) => (2, lhs, rhs, u32::MAX),
                TapeOp::Mul(lhs, rhs) => (3, lhs, rhs, u32::MAX),
                TapeOp::Div(lhs, rhs) => (4, lhs, rhs, u32::MAX),
                TapeOp::Tanh(input) => (5, input, 0, u32::MAX),
                TapeOp::Exp(input) => (6, input, 0, u32::MAX),
                TapeOp::Sqrt(input) => (7, input, 0, u32::MAX),
                TapeOp::Sin(input) => (8, input, 0, u32::MAX),
                TapeOp::Sigmoid(input) => (9, input, 0, u32::MAX),
                TapeOp::Max(lhs, rhs) => (10, lhs, rhs, u32::MAX),
                TapeOp::StopGradient(input) => (11, input, 0, u32::MAX),
            };
            opcodes.push(opcode);
            left.push(lhs as u32);
            right.push(rhs as u32);
            parameter_ids.push(parameter);
        }
        (initial_values, opcodes, left, right, parameter_ids)
    }
    #[allow(clippy::too_many_arguments)]
    fn accelerated_train_step(
        &self,
        selection: &BackendSelection,
        loss: usize,
        parameters: &mut [f64],
        first_moment: &mut [f64],
        second_moment: &mut [f64],
        step: u64,
        learning_rate: f64,
        weight_decay: f64,
    ) -> Result<f64> {
        let (initial_values, opcodes, left, right, parameter_ids) = self.accelerator_arrays();
        let mut parameters_f32 = parameters
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        let mut first_f32 = first_moment
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        let mut second_f32 = second_moment
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        let value = backend_scalar_graph_train_step_f32(
            selection,
            &initial_values,
            &opcodes,
            &left,
            &right,
            &parameter_ids,
            loss,
            &mut parameters_f32,
            &mut first_f32,
            &mut second_f32,
            step,
            learning_rate as f32,
            weight_decay as f32,
        )
        .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        for (output, value) in parameters.iter_mut().zip(parameters_f32) {
            *output = value as f64;
        }
        for (output, value) in first_moment.iter_mut().zip(first_f32) {
            *output = value as f64;
        }
        for (output, value) in second_moment.iter_mut().zip(second_f32) {
            *output = value as f64;
        }
        Ok(value as f64)
    }
    fn value(&self, value: usize) -> f64 {
        self.nodes.borrow()[value].value
    }
    fn backward(&self, loss: usize, parameter_count: usize) -> Vec<f64> {
        let nodes = self.nodes.borrow();
        let mut gradients = vec![0.0; nodes.len()];
        let mut parameter_gradients = vec![0.0; parameter_count];
        gradients[loss] = 1.0;
        for index in (0..nodes.len()).rev() {
            let gradient = gradients[index];
            match nodes[index].op {
                TapeOp::Constant => {}
                TapeOp::Parameter(parameter) => parameter_gradients[parameter] += gradient,
                TapeOp::Add(left, right) => {
                    gradients[left] += gradient;
                    gradients[right] += gradient;
                }
                TapeOp::Mul(left, right) => {
                    gradients[left] += gradient * nodes[right].value;
                    gradients[right] += gradient * nodes[left].value;
                }
                TapeOp::Div(left, right) => {
                    let denominator = nodes[right].value.max(1e-12);
                    gradients[left] += gradient / denominator;
                    gradients[right] -= gradient * nodes[left].value / denominator.powi(2);
                }
                TapeOp::Tanh(input) => {
                    gradients[input] += gradient * (1.0 - nodes[index].value.powi(2))
                }
                TapeOp::Exp(input) => gradients[input] += gradient * nodes[index].value,
                TapeOp::Sqrt(input) => {
                    gradients[input] += gradient / (2.0 * nodes[index].value.max(1e-12))
                }
                TapeOp::Sin(input) => gradients[input] += gradient * nodes[input].value.cos(),
                TapeOp::Sigmoid(input) => {
                    gradients[input] += gradient * nodes[index].value * (1.0 - nodes[index].value)
                }
                TapeOp::Max(left, right) => {
                    if nodes[left].value >= nodes[right].value {
                        gradients[left] += gradient;
                    } else {
                        gradients[right] += gradient;
                    }
                }
                TapeOp::StopGradient(_) => {}
            }
        }
        parameter_gradients
    }
}

