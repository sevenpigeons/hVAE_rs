use crate::model::hVAE;
use burn_import::pytorch::PyTorchFileRecorder;
use burn::record::FullPrecisionSettings;
use burn::record::Recorder;
use burn::prelude::*;

#[cfg(feature = "wgpu")]
use burn_wgpu::{graphics::AutoGraphicsApi,init_setup_async,Wgpu, WgpuDevice};

#[cfg(feature = "wgpu")]
type Backend = Wgpu<f64,f64>;


#[cfg(all(feature = "ndarray", not(feature = "wgpu")))]
pub type Backend = burn::backend::ndarray::NdArray<f64>;

// somewhat stolen from burn mnist example
pub async fn build_load_model() -> hVAE<Backend> {

    #[cfg(feature = "wgpu")]
    init_setup_async::<AutoGraphicsApi>(&WgpuDevice::default(), Default::default());

    let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
        .load("./weights.pt".into(),&Default::default())
        .expect("idk");
    // Load weights from PyTorch file
    //let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
    //    .load("./conv2d.pt".into(), &device)
     //   .expect("Should decode state successfully");

    // Initialize model and load weights
    hVAE::init(&Default::default(),vec![784,514],3).load_record(record)
}
