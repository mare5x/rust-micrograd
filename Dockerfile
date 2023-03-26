FROM rust:1.67

# For better Docker image caching, we first create
# a dummy project and install the project's dependencies.
# See: https://stackoverflow.com/questions/42130132/can-cargo-download-and-build-dependencies-without-also-building-the-application

RUN cd / && cargo new micrograd
WORKDIR /micrograd

COPY Cargo.toml .
RUN cargo build
# Remove auto-generated files
RUN rm src/*.rs

# Copy our files and build again, using the cached dependencies.
# N.B. `.dockerignore` exists.
COPY . .
RUN cargo build

# Default command if no command given to `docker run`.
CMD ["cargo", "test"]
