
# Setup

- Build the project: `cargo build`
- Test the project: `cargo test`
- Run an example: `cargo run -r --bin linreg`

## Docker

- Build the image: `docker build -t rust-micrograd .`  
- Run the container: `docker run -it --rm rust-micrograd /bin/bash`
  - Mount (for development purposes): 
`docker run --mount type=bind,src="$(pwd)/src",target=/micrograd/src -it --rm rust-micrograd /bin/bash`
