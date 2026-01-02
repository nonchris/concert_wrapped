"""FastAPI application entry point."""
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from concert_data_thing.main import analyze_concert_csv


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
    artist: str = "Artist"
    venue: str = "Venue"
    city_column: str = "City"
    country: str = "Country"
    paid_price: str = "Bezahlt Preis"
    original_price: str = "Preis"
    merch_cost: str = "Merch Ausgaben"
    type: str = "Typ"
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

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the concert CSV analysis form."""
    return HTMLResponse(content=entry_point_form)


@app.post("/api/v1/analyze-concert", response_class=HTMLResponse)
async def analyze_concert_route(request: AnalyzeConcertRequest) -> HTMLResponse:
    """
    Analyze concert CSV data and return HTML statistics page.
    
    Args:
        request: Request body containing CSV string and analysis parameters.
        
    Returns:
        HTMLResponse containing the statistics page.
    """
    # Call the analysis function
    analyze_concert_csv(
        csv_str=request.csv_str,
        filter_year=request.filter_year,
        user_name=request.user_name,
        city=request.city,
        date=request.date,
        artist=request.artist,
        venue=request.venue,
        city_column=request.city_column,
        country=request.country,
        paid_price=request.paid_price,
        original_price=request.original_price,
        merch_cost=request.merch_cost,
        type=request.type,
        event_name=request.event_name,
        running_order_headline_last=request.running_order_headline_last,
    )
    
    # Read and return the HTML file
    html_path = Path(__file__).parent / "images" / "test.html"
    html_content = html_path.read_text(encoding="utf-8")
    
    return HTMLResponse(content=html_content)


def main():
    """Entry point for CLI."""
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "data_stuff.endpoints:app",
        reload=True,
        reload_dirs=[str(Path(__file__).parent)],
        host="0.0.0.0",
        port=8000,
    )
