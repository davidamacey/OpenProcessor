/// Environment-based configuration.

#[derive(Clone, Debug)]
pub struct Config {
    pub triton_url: String,
    pub opensearch_url: String,
    pub port: u16,
    pub grpc_pool_size: usize,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            triton_url: std::env::var("TRITON_URL")
                .unwrap_or_else(|_| "triton-server:8001".to_string()),
            opensearch_url: std::env::var("OPENSEARCH_URL")
                .unwrap_or_else(|_| "http://opensearch:9200".to_string()),
            port: std::env::var("PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8000),
            grpc_pool_size: std::env::var("GRPC_POOL_SIZE")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(4),
        }
    }
}
