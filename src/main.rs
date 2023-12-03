use burn::{
    module::Module,
    nn,
    backend::Wgpu,
    tensor::Tensor, data::dataset::Dataset,
};

type Backend = Wgpu;

fn main() {
    let x = Tensor::
}

#[derive(Module, Debug, Clone)]
pub struct MLP {
    l1: nn::Linear<Backend>,
    l2: nn::Linear<Backend>,
    l3: nn::Linear<Backend>,
    gelu: nn::GELU,
}

impl MLP {
    pub fn forward<const D: usize>(&self, input: Tensor<Backend, D>) -> Tensor<Backend, D> {
        let x = self.l1.forward(input);
        let x = self.gelu.forward(x);
        let x = self.l2.forward(x);
        let x = self.gelu.forward(x);
        self.l3.forward(x)
    }
}

#[derive(Debug, Clone)]
pub struct XYDataset {
    x: Tensor<Backend, 1>,
    y: Tensor<Backend, 1>,
}

impl XYDataset {
    pub fn new(x: Vec<Tensor<Backend, 1>>, y: Vec<Tensor<Backend, 1>>) -> XYDataset {
        XYDataset { x, y }
    }
}

impl Dataset<(Tensor<Backend, 1>, Tensor<Backend, 1>)> for XYDataset {
    fn get(&self, index: usize) -> Option<(Tensor<Backend, 1>, Tensor<Backend, 1>)> {
        Some((self.x.select(0, index), self.y.select(0, index)))
    }

    fn len(&self) -> usize {
        self.x.shape().num_elements()
    }
}
