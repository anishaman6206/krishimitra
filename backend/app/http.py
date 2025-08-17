import httpx
from typing import Optional

# Global HTTP client instance
client: Optional[httpx.AsyncClient] = None

async def init_http():
    """Initialize the global HTTP client with optimized settings."""
    global client

    try:
        import h2
        http2_available = True
    except ImportError:
        http2_available = False
        print("⚠️  HTTP/2 not available. Install with: pip install httpx[http2]")

    # Set reasonable timeouts for different API types:
    # - connect: 10s (establishing connection)
    # - read: 25s (reading response) 
    # - write: 10s (sending request)
    # - pool: 30s (getting connection from pool)
    timeout_config = httpx.Timeout(
        connect=10.0,
        read=25.0,
        write=10.0,
        pool=30.0
    )

    client = httpx.AsyncClient(
        timeout=timeout_config,
        http2=http2_available,
        limits=httpx.Limits(
            max_keepalive_connections=100, 
            max_connections=100,
            keepalive_expiry=30  # Keep connections alive for 30s
        ),
        headers={
            "Accept-Encoding": "gzip, deflate, br",
            "User-Agent": "KrishiMitra/1.0 (+https://krishimitra.example.com)"
        },
    )

async def close_http():
    """Close the global HTTP client."""
    global client
    if client:
        await client.aclose()
        client = None

def get_http_client() -> httpx.AsyncClient:
    """Get the global HTTP client instance."""
    if client is None:
        raise RuntimeError("HTTP client not initialized. Call init_http() first.")
    return client
