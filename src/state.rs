use crate::model::hVAE;
use burn::module::Module;
use burn::record::BinBytesRecorder;
use burn::record::HalfPrecisionSettings;
use burn_import::pytorch::PyTorchFileRecorder;
use burn::record::FullPrecisionSettings;
use burn::record::Recorder;

#[cfg(feature = "wgpu")]
use burn::backend::wgpu::{Wgpu, WgpuDevice, graphics::AutoGraphicsApi, init_setup_async};
//use burn_wgpu::{graphics::AutoGraphicsApi,init_setup_async,Wgpu, WgpuDevice};

#[cfg(feature = "wgpu")]
pub type Backend = Wgpu<f32,i32>;


#[cfg(all(feature = "ndarray", not(feature = "wgpu")))]
pub type Backend = burn::backend::ndarray::NdArray<f64>;



//static PYTORCH_FILENAME: &str = "./weights.pt";

static STATE_ENCODED: &[u8] = include_bytes!("../model.bin");

// somewhat stolen from burn mnist example
pub async fn build_load_model() -> hVAE<Backend> {

    #[cfg(feature = "wgpu")]
    init_setup_async::<AutoGraphicsApi>(&WgpuDevice::default(), Default::default()).await;

    
 //   let record = match PyTorchFileRecorder::<FullPrecisionSettings>::default()
 //       .load(PYTORCH_FILENAME.into(),&Default::default()) {
 //       Ok(val) => val,
 //       Err(e) => panic!("{e}")
 //       };

    let model: hVAE<Backend> = hVAE::init(&Default::default(),vec![784,514],3);

    let model_record = BinBytesRecorder::<FullPrecisionSettings,&'static [u8]>::default()
        .load(STATE_ENCODED, &Default::default())
        .expect("iod");

    //
    //
    //
    // Load weights from PyTorch file
    //let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
    //    .load("./conv2d.pt".into(), &device)
     //   .expect("Should decode state successfully");

    // Initialize model and load weights
    model.load_record(model_record)
}
