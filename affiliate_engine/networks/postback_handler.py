"""
Postback handler for affiliate network conversions.
"""

import logging
from datetime import UTC, datetime
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from ..analytics.postback_security import PostbackSecurity, SecurityConfig
from ..analytics.signature_replay import SignatureReplay
from .cleanup_tasks import start_cleanup_task

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize security components
security_config = SecurityConfig(
    secret_key="your-secret-key",  # TODO: Move to environment variables
    replay_ttl_seconds=600,  # 10 minutes
    max_signatures_per_ip=1000,
)

postback_security = PostbackSecurity(security_config)
replay_protection = SignatureReplay(
    redis_url="redis://localhost:6379/0",  # TODO: Move to environment variables
    ttl_seconds=600,  # 10 minutes
    max_signatures_per_ip=1000,
)

app = FastAPI(title="Affiliate Postback Handler")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    # Start cleanup task
    background_tasks = BackgroundTasks()
    start_cleanup_task(background_tasks, replay_protection)
    app.background_tasks = background_tasks


async def get_network_from_request(request: Request) -> str:
    """Extract network name from request path."""
    path = request.url.path
    if path.startswith("/postback/"):
        return path.split("/")[2]
    raise HTTPException(status_code=400, detail="Invalid postback URL")


async def parse_payload(request: Request) -> Dict[str, Any]:
    """Parse and validate postback payload."""
    try:
        payload = await request.json()
        return payload
    except Exception as e:
        logger.error(f"Error parsing payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid payload format")


async def extract_signature(request: Request) -> str:
    """Extract signature from request headers."""
    signature = request.headers.get("X-Signature")
    if not signature:
        raise HTTPException(status_code=400, detail="Missing signature header")
    return signature


async def log_replay_attempt(network: str, payload: Dict[str, Any], result: Any, ip: str) -> None:
    """Log replay attempt details."""
    logger.warning(
        f"Replay attempt detected:\n"
        f"Network: {network}\n"
        f"IP: {ip}\n"
        f"Error: {result.error_message}\n"
        f"Payload: {payload}"
    )


@app.post("/postback/{network}")
async def handle_postback(request: Request, network: str = Depends(get_network_from_request)):
    """Handle postback from affiliate network."""
    try:
        # Extract request data
        payload = await parse_payload(request)
        signature = await extract_signature(request)
        ip = request.client.host

        # First check for replay attempts
        replay_result = replay_protection.check_replay(
            network=network, payload=payload, signature=signature, ip=ip
        )

        if replay_result.is_replay:
            await log_replay_attempt(network, payload, replay_result, ip)
            return JSONResponse(
                status_code=200,
                content={"status": "duplicate", "message": replay_result.error_message},
            )

        # Then validate signature
        security_result = postback_security.validate_postback(
            network=network, payload=payload, signature=signature
        )

        if not security_result.is_valid:
            logger.warning(f"Invalid signature from {network}: {security_result.error_message}")
            return JSONResponse(
                status_code=400,
                content={"status": "invalid", "message": security_result.error_message},
            )

        # Process valid conversion
        # TODO: Add your conversion processing logic here
        # For example:
        # - Record conversion in database
        # - Update affiliate stats
        # - Trigger notifications

        return JSONResponse(
            status_code=200, content={"status": "success", "message": "Conversion recorded"}
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing postback: {e}")
        raise HTTPException(status_code=500, detail="Internal server error processing postback")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now(UTC).isoformat()}


@app.get("/stats")
async def get_stats():
    """Get replay protection statistics."""
    try:
        stats = replay_protection.get_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving statistics")
