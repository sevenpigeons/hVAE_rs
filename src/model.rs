use std::vec;

use burn::{prelude::*, tensor::activation::{relu, sigmoid}};

#[derive(Module,Debug)]
pub struct Encoder<B: Backend> {
    encoder_list: Vec<nn::Linear<B>>,
    mu_var_layer: nn::Linear<B>,
    features: usize
}

#[derive(Module,Debug)]
pub struct Decoder<B: Backend> {
    decoder_list: Vec<nn::Linear<B>>,
    mean_layer: nn::Linear<B>,
}


#[derive(Module,Debug)]
pub struct hVAE<B: Backend> {
    encoder: Encoder<B>,
    decoders: Vec<Decoder<B>>

}


pub const mnist_data: [f32;784] = [0.        , 0.        , 0.        , 0.        , 0., 0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.45490196, 0.7647059 , 0.27450982, 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.26666668, 0.9490196 , 0.9882353 ,
       0.972549  , 0.24313726, 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.05098039,
       0.87058824, 0.9882353 , 0.9882353 , 0.99215686, 0.49019608,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.19215687, 0.77254903, 0.9882353 , 0.9882353 ,
       0.9882353 , 0.99215686, 0.95686275, 0.2509804 , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.4       ,
       0.9882353 , 0.9882353 , 0.9882353 , 0.8745098 , 0.99215686,
       0.9882353 , 0.3764706 , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.18431373, 0.93333334, 0.9882353 , 0.9882353 ,
       0.6627451 , 0.12156863, 0.99215686, 0.9882353 , 0.73333335,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.06666667, 0.7137255 ,
       0.9882353 , 0.9882353 , 0.9882353 , 0.40784314, 0.        ,
       0.99215686, 0.9882353 , 0.84705883, 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.19215687, 0.9882353 , 0.9882353 , 0.9882353 ,
       0.9882353 , 0.43529412, 0.38039216, 0.99215686, 0.9882353 ,
       0.9019608 , 0.69411767, 0.50980395, 0.27058825, 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.19215687,
       0.9882353 , 0.9882353 , 0.9882353 , 0.9882353 , 0.9882353 ,
       0.9882353 , 0.99215686, 0.9882353 , 0.9882353 , 0.9882353 ,
       0.9882353 , 0.827451  , 0.08235294, 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.01176471, 0.49803922, 0.9882353 ,
       0.9882353 , 0.9882353 , 0.9882353 , 0.9882353 , 0.99215686,
       0.9882353 , 0.9882353 , 0.9882353 , 0.9882353 , 0.9882353 ,
       0.1882353 , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.16078432, 0.4745098 ,
       0.9019608 , 0.94509804, 1.        , 0.99215686, 0.9882353 ,
       0.627451  , 0.4745098 , 0.34117648, 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.99215686, 0.9882353 , 0.84705883, 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.99215686, 0.9882353 ,
       0.84705883, 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.99215686, 0.9882353 , 0.84705883, 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.99215686,
       0.9882353 , 0.84705883, 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.99215686, 0.9882353 , 0.84705883,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.16078432,
       0.99215686, 0.9882353 , 0.84705883, 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.4745098 , 0.99215686, 0.9882353 ,
       0.84705883, 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.04313726, 0.99215686, 0.9882353 , 0.9137255 , 0.15686275,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.29411766,
       0.7882353 , 0.84705883, 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,0., 0., 0., 0., 0.,0., 0., 0., 0.];



impl <B:Backend> hVAE<B> {
    pub fn init(device: &B::Device,steps:Vec<usize>,features:usize) -> Self {
        let encoder = Encoder::init(device, steps.clone(), features);
        let mut decoders = vec![];
        for i in 0..features {
            decoders.push(Decoder::init(device, steps.clone(), i+1));
        }
        Self { encoder, decoders }
    }

    pub fn reparameterize(self,mu: Tensor<B,1>, log_var:Tensor<B,1>) -> Tensor<B,1> {
        let std = (log_var*0.5).exp();
        let eps = std.random_like(burn::tensor::Distribution::Normal(0.0, 1.0));
        mu + (eps*std)
    }

    pub fn forward(&self,x:Tensor<B,1>) -> (Vec<Tensor<B,1>>,Tensor<B,1>,Tensor<B,1>,Tensor<B,1>) {
        let (mu, log_var) = &self.encoder.forward(x);
        //let size = &mu.slice(s![..,0]).dims();
        let mut reconstructions = Vec::with_capacity(3);
        let z = &self.clone().reparameterize(mu.clone(), log_var.clone()).clone();
        for (i,decoder) in self.decoders.iter().enumerate() {
            reconstructions.push(
                decoder.forward(z.clone().slice(0..i+1))
            );
        }

        (reconstructions,mu.clone(),log_var.clone(),z.clone())

    }
}



impl <B:Backend> Decoder<B> {
    pub fn init(device: &B::Device,steps:Vec<usize>,features:usize) -> Self {
        let mut decoder_list:Vec<nn::Linear<B>> = vec![];
        let mut new_steps = steps;
        new_steps.push(features);
        for i in 0..(new_steps.len() -2 ) {
            decoder_list.push(nn::LinearConfig::new(new_steps[new_steps.len()-(i+1)],new_steps[new_steps.len()-(i+2)]).init(device));
        }


        let mean_layer = nn::LinearConfig::new(new_steps[1], new_steps[0]).init(device);

        Self { decoder_list, mean_layer }
    }

    pub fn forward(&self,x:Tensor<B,1>) -> Tensor<B,1> {
        let mut x1 = x;
        for layer in &self.decoder_list {
            x1 = relu(layer.forward(x1));
        }
        x1 = sigmoid(self.mean_layer.forward(x1));
        x1
    }
    
}


impl<B:Backend> Encoder<B>  {
    pub fn init(device: &B::Device, steps:Vec<usize>,features:usize) -> Self {
        let mut encoder_list:Vec<nn::Linear<B>> = vec![];
        for i in 0..(steps.len()-1) {
            encoder_list.push(nn::LinearConfig::new(steps[i], steps[i+1]).init(device));
        }
        Self { encoder_list: encoder_list, mu_var_layer: nn::LinearConfig::new(steps[steps.len()-1], features*2).init(device), features: features}
    }

    pub fn forward(&self, x: Tensor<B,1>) -> (Tensor<B,1>,Tensor<B,1>) {
        let mut x1 = x;
        for layer in &self.encoder_list {
            x1 = relu(layer.forward(x1));
        };
        x1 = self.mu_var_layer.forward(x1);
        let x2  = x1.reshape([-1isize,2,self.features.try_into().unwrap()]);
        let mu = x2.clone().slice(s![..,(0..1),..]).squeeze();
        let var = x2.clone().slice(s![..,(1..2),..]).squeeze();
        (mu,var)


    }
}


