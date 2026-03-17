use burn::Tensor;
use js_sys::Array;

#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::*;

use crate::model::hVAE;
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
        tensor_input = tensor_input/255;//normalized in the example but we dont do that

        let (reconstructions,mu,log_var,z) = model.forward(tensor_input);


        let output = reconstructions[2].clone().into_data_async().await.unwrap();
        let array = Array::new();
        for value in output.iter::<f32>() {
            array.push(&value.into());
        }
        Ok(array)
    }
    
}
