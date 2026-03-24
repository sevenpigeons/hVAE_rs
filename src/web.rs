use burn::Tensor;
use alloc::string::String;
use burn::tensor::TensorData;
use js_sys::Array;

#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::*;

use crate::model::{self, hVAE};
use crate::state::{Backend, build_load_model};



#[cfg_attr(target_family = "wasm", wasm_bindgen(start))]
pub fn start() {
    console_error_panic_hook::set_once();
}


#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub struct Mnist {
    model: Option<hVAE<Backend>>
}


#[cfg_attr(target_family = "wasm", wasm_bindgen)]
impl Mnist {
    #[cfg_attr(target_family = "wasm", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        Self { model: None }
        
    }
    pub async fn inference(&mut self,input: &[f32]) -> Result<Array,String> {
        if self.model.is_none() {
            self.model = Some(build_load_model().await);
        }
        let model = self.model.as_ref().unwrap();
        let device = Default::default();
        let mut tensor_input = Tensor::<Backend,1>::from_floats(input, &device);
        //let mut tensor_input = Tensor::<Backend,1>::from_floats(model::mnist_data, &device);
        tensor_input = tensor_input/255;//normalized in the example but we dont do that

        let (reconstructions,mu,log_var,z) = model.forward(tensor_input);

        let array0 = Array::new();
        let array1 = Array::new();
        let array2 = Array::new();
let arr_vec = [array0,array1,array2];






        for i in 0..3 {
            let recon = &reconstructions[i];
            for value in recon.clone().into_data_async().await.unwrap().iter::<f32>() {
                //array0.push(&value.into());
                &(arr_vec[i]).push(&value.into());
            }

        }

//        let output0:TensorData = reconstructions[0].clone().into_data_async().await.unwrap();
 //       let output1:TensorData = reconstructions[1].clone().into_data_async().await.unwrap();
  //      let output2:TensorData = reconstructions[2].clone().into_data_async().await.unwrap();
        let res = Array::new();
        for arr in arr_vec{
        res.push(&arr.into());
        }
        Ok(res)
    }
    
}
