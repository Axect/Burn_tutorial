use burn::{
    module::Module,
    nn,
    backend::WgpuBackend,
    tensor::Tensor, data::dataset::Dataset,
};

type Backend = WgpuBackend;

fn main() {

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
