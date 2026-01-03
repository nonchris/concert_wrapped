"""FastAPI application entry point."""

import io
import os
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.responses import Response
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBasic
from fastapi.security import HTTPBasicCredentials
from pydantic import BaseModel

from concert_data_thing.logger import LOGGING_PROVIDER
from concert_data_thing.main import analyze_concert_csv

logger = LOGGING_PROVIDER.new_logger("concert_data_thing.endpoints")

# Basic Auth configuration from environment variables
BASIC_AUTH_USERNAME = os.getenv("BASIC_AUTH_USERNAME", None)
BASIC_AUTH_PASSWORD = os.getenv("BASIC_AUTH_PASSWORD", None)
BASIC_AUTH_ENABLED = bool(BASIC_AUTH_USERNAME and BASIC_AUTH_PASSWORD)

# FastAPI security scheme
if BASIC_AUTH_ENABLED:
    security = HTTPBasic()
    logger.info(f"Basic auth is enabled, username: {BASIC_AUTH_USERNAME}, password: {'*' * len(BASIC_AUTH_PASSWORD)}")
else:
    security = lambda: True
    logger.warning("!!! Basic auth is disabled !!!")


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> HTTPBasicCredentials:
    """
    Verify basic auth credentials.

    Args:
        credentials: HTTP basic auth credentials from request.

    Returns:
        Credentials if valid.

    Raises:
        HTTPException: If authentication is enabled and credentials are invalid.
    """
    if not BASIC_AUTH_ENABLED:
        return credentials

    is_correct_username = credentials.username == BASIC_AUTH_USERNAME
    is_correct_password = credentials.password == BASIC_AUTH_PASSWORD

    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifespan events."""
    # Startup
    yield
    # Shutdown


class AnalyzeConcertRequest(BaseModel):
    """Request model for concert CSV analysis."""

    csv_str: str
    filter_year: int
    user_name: str
    city: str
    date: str = "Date"
    date_format: str = "%d.%m.%y"
    sep: str = ","
    artist: str = "Artist"
    venue: str = "Venue"
    city_column: str = "City"
    country: str = "Country"
    paid_price: str = "Bezahlt Preis"
    original_price: str = "Preis"
    merch_cost: str = "Merch Ausgaben"
    type: str = "Typ"
    headline_label: str = "auto"
    support_label: str = "auto"
    festival_label: str = "F"
    event_name: str = "Event Name"
    running_order_headline_last: bool = True


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="CSV Data Concert API",
        description="Like stats.fm but for concerts",
        version="0.0.1",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],  # NextJS dev server
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()

entry_point_form = (Path(__file__).parent / "forms" / "entry_form.html").read_text()


def get_svg_path(relative_path: str) -> Path:
    """Get absolute path for a relative SVG path."""
    # relative_path should be like "out/user_data_{uuid}/file.svg"
    base_path = Path(__file__).parent
    full_path = base_path / relative_path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"SVG file not found: {relative_path}")
    return full_path


@app.get("/api/v1/svg/{file_path:path}")
async def get_svg(file_path: str, credentials: HTTPBasicCredentials = Depends(verify_credentials)) -> Response:
    """
    Serve an SVG file.

    Args:
        file_path: Relative path to the SVG file (e.g., "out/user_data_{uuid}/file.svg").

    Returns:
        SVG file content.
    """
    svg_path = get_svg_path(file_path)
    return FileResponse(
        svg_path,
        media_type="image/svg+xml",
        headers={"Content-Disposition": f'inline; filename="{svg_path.name}"'},
    )


@app.get("/api/v1/download/{file_path:path}")
async def download_svg(file_path: str, credentials: HTTPBasicCredentials = Depends(verify_credentials)) -> Response:
    """
    Download an SVG file.

    Args:
        file_path: Relative path to the SVG file (e.g., "out/user_data_{uuid}/file.svg").

    Returns:
        SVG file as download.
    """
    svg_path = get_svg_path(file_path)
    return FileResponse(
        svg_path,
        media_type="image/svg+xml",
        headers={"Content-Disposition": f'attachment; filename="{svg_path.name}"'},
    )


@app.get("/api/v1/download-all/{request_id}")
async def download_all_svgs(
    request_id: str, credentials: HTTPBasicCredentials = Depends(verify_credentials)
) -> StreamingResponse:
    """
    Download all SVGs for a request as a zip file.

    Args:
        request_id: Request ID UUID.

    Returns:
        Zip file containing all SVGs.
    """
    user_data_folder = Path(__file__).parent / "out" / f"user_data_{request_id}"
    if not user_data_folder.exists():
        raise HTTPException(status_code=404, detail=f"Request data not found: {request_id}")

    # Create zip in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for svg_file in user_data_folder.rglob("*.svg"):
            zip_file.write(svg_file, svg_file.name)

    zip_buffer.seek(0)

    return StreamingResponse(
        io.BytesIO(zip_buffer.read()),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="concert_analysis_{request_id}.zip"'},
    )


@app.get("/", response_class=HTMLResponse)
async def index(credentials: HTTPBasicCredentials = Depends(verify_credentials)) -> HTMLResponse:
    """Serve the concert CSV analysis form."""
    return HTMLResponse(content=entry_point_form)


@app.get("/favicon.png")
async def favicon() -> FileResponse:
    """Serve the favicon."""
    favicon_path = Path(__file__).parent / "forms" / "favicon.png"
    return FileResponse(favicon_path)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint (no auth required)."""
    return {"status": "healthy"}


class AnalyzeConcertResponse(BaseModel):
    """Response model for concert analysis."""

    request_id: str
    user_svgs: list[str]  # List of user SVG paths
    artist_svgs: list[str]
    venue_svgs: list[str]
    city_svgs: list[str]


@app.post("/api/v1/analyze-concert")
async def analyze_concert_route(
    request: AnalyzeConcertRequest, credentials: HTTPBasicCredentials = Depends(verify_credentials)
) -> AnalyzeConcertResponse:
    """
    Analyze concert CSV data and return JSON with SVG paths.

    Args:
        request: Request body containing CSV string and analysis parameters.

    Returns:
        JSON response with request_id and SVG paths.
    """
    print(f"Received request: city={request.city}, user_name={request.user_name}, filter_year={request.filter_year}")

    request_id = uuid.uuid4()

    result = analyze_concert_csv(
        csv_str=request.csv_str,
        filter_year=request.filter_year,
        user_name=request.user_name,
        city=request.city,
        date=request.date,
        date_format=request.date_format,
        sep=request.sep,
        artist=request.artist,
        venue=request.venue,
        city_column=request.city_column,
        country=request.country,
        paid_price=request.paid_price,
        original_price=request.original_price,
        merch_cost=request.merch_cost,
        type=request.type,
        headline_label=request.headline_label,
        support_label=request.support_label,
        festival_label=request.festival_label,
        event_name=request.event_name,
        running_order_headline_last=request.running_order_headline_last,
        request_id=request_id,
    )

    # Convert Path objects to strings relative to the request_id folder
    def path_to_api_path(path: Path) -> str:
        """Convert file path to API-accessible path."""
        # Convert to string and normalize path separators
        path_str = str(path).replace("\\", "/")
        # If it's an absolute path, extract relative part from 'out' directory
        if "/out/" in path_str:
            idx = path_str.index("/out/")
            return path_str[idx + 1 :]  # Remove leading slash
        return path_str

    return AnalyzeConcertResponse(
        request_id=result["request_id"],
        user_svgs=[path_to_api_path(p) for p in result["user_svgs"]],
        artist_svgs=[path_to_api_path(p) for p in result["artist_svgs"]],
        venue_svgs=[path_to_api_path(p) for p in result["venue_svgs"]],
        city_svgs=[path_to_api_path(p) for p in result["city_svgs"]],
    )


def main():
    """Entry point for CLI."""
    import uvicorn

    uvicorn.run("concert_data_thing.endpoints:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "concert_data_thing.endpoints:app",
        reload=True,
        reload_dirs=[str(Path(__file__).parent)],
        host="0.0.0.0",
        port=8000,
    )
