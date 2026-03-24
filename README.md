# hVAE - rust port

This is a rust port of the [hVAE](https://github.com/sevenpigeons/hVAE) codebase for visualization purposes purposes.

Some amounts of stuff from here has been modified code from the MNIST web example from the [burn](https://burn.dev) library,
in the [examples](https://github.com/tracel-ai/burn/tree/main/examples/mnist-inference-web) part of their code repository.



## Running

Same steps as with the burn's mnist example apply, 

1. Install rust toolchain
    At [this](https://rust-lang.org/tools/install/) link you will find installation steps for your architecture

2. Install wasm-pack
    1. Optional binstall

    `cargo install binstal`
    
    `cargo binstall wasm-pack`
    
    `binstall` isn't necessarily required, but as an installation method is searches github for precomiled binaries
    so you dont have to build all the tools directly on your machine locally, saving time

3. Run the build script with wither `ndarray` or `wgpu` backend

    `./build-for-web.sh {backend}`

4. Run the webserver with 

    ```python3 -m http.server -d .```
