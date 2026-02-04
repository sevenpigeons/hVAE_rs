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


#[allow(nonstandard_style)]
#[derive(Module,Debug)]
pub struct hVAE<B:Backend> {
    encoder: Encoder<B>,
    decoders: Vec<Decoder<B>>

}


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
        let mut reconstructions = vec![];
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


