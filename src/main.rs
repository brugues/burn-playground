#![recursion_limit = "256"]

mod data;
mod model;
mod training;

#[cfg(feature = "cuda")]
fn main() {
    use burn::backend::cuda::{Cuda, CudaDevice};
    let device = CudaDevice::default();
    training::run::<Cuda>(device);
}

#[cfg(feature = "flex")]
fn main() {
    use burn::backend::flex::{Flex, FlexDevice};
    let device = FlexDevice::default();
    training::run::<Flex>(device);
}

#[cfg(not(any(feature = "cuda", feature = "flex")))]
fn main() {
    panic!("Please enable one of the following features: cuda, flex");
}
