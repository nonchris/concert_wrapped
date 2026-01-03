# Concert Data Thing

A FastAPI-based web service that analyzes concert attendance data from CSV files and generates SVG visualizations.

Think of it as "Wrapped but for concerts" - providing insights into your concert-going habits, favorite artists, venues, and cities.

## What It Does

The API accepts concert data in CSV format and generates comprehensive visualizations including:

- **User-level statistics**: Total shows attended, ticket costs, calendar heatmaps, and more
- **Artist analysis**: Top artists seen, venues and cities where you saw them, pricing information
- **Venue analysis**: Most visited venues with location and pricing data
- **City analysis**: Concert activity across different cities

All visualizations are generated as SVG files that can be viewed in browsers, downloaded individually, or exported as a ZIP archive.

The SVG approach allows easy modifications to the images by the enduser, since everything is editable.

## Technology: SVG Template Approach

The application uses a **placeholder-driven SVG template system** for generating visualizations. Instead of programmatically drawing every element, the system, it uses pre-designed SVG templates with placeholder markers (e.g., `<Tcx>` for total cost)


This approach provides:
- **Consistent design**: Templates ensure visual consistency across all generated graphics
- **Easy customization**: Modify SVG templates without changing Python code
- **Scalable output**: SVG format is resolution-independent and web-friendly
- **Fast generation**: Template replacement is more efficient than programmatic drawing
- **Editable after genration**: Everything can be changed after the image is generated and downloaded.

## Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `BASIC_AUTH_USERNAME` | Username for HTTP Basic Authentication | `None` (disabled) | No |
| `BASIC_AUTH_PASSWORD` | Password for HTTP Basic Authentication | `None` (disabled) | No |
| `ARTIFACTS_PATH` | Directory path where generated SVG files are stored | `"out"` | No |
| `LOG_DIR` | Directory path for log files | `None` (console only) | No |

**Note**: If `BASIC_AUTH_USERNAME` and `BASIC_AUTH_PASSWORD` are both set, all endpoints (except `/health`) will require HTTP Basic Authentication.
**If either is unset, authentication is disabled.**

## Running the thing

### Prerequisites

- Python 3.9 or higher
- pip

### Setup

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd concert_data_thing
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Set environment variables** (optional):
   ```bash
   export BASIC_AUTH_USERNAME="your_username"
   export BASIC_AUTH_PASSWORD="your_password"
   export ARTIFACTS_PATH="out"
   ```

5. **Run the API server**:
   ```bash
   start-concert-api
   ```

   Or directly with Python:
   ```bash
   python -m concert_data_thing.endpoints
   ```

   Or using uvicorn directly:
   ```bash
   uvicorn concert_data_thing.endpoints:app --host 0.0.0.0 --port 8000 --reload
   ```

6. **Access the API**:
   - Web interface: http://localhost:8000/
   - API documentation: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

The API will be available at `http://localhost:8000`.
Generated SVG files will be stored in the `out/` directory (or the path specified by `ARTIFACTS_PATH`).
