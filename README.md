# Puerto Vallarta seasonal satellite timelapse

A true-color Sentinel-2 timelapse of an exact 10 km × 10 km area centered on Puerto Vallarta, Mexico. It contains one low-cloud scene for each meteorological season from autumn 2023 through summer 2026.

![Puerto Vallarta seasonal Sentinel-2 timelapse](puerto-vallarta-seasonal-rgb-2023-2026.gif)

## Result

- **Frames:** 12 seasonal observations
- **Resolution:** 768 × 768 pixels
- **Area:** 10,000 m × 10,000 m, calculated in the local UTM projection
- **Center:** 20.6534° N, 105.2307° W
- **Source:** Tilebox `open_data.aws_earth.sentinel2` dataset, Sentinel-2 Level-2A imagery
- **Scene metadata:** [`puerto-vallarta-seasonal-rgb-2023-2026.json`](puerto-vallarta-seasonal-rgb-2023-2026.json)

## Method

The Tilebox workflow fans out one task per season. Each task:

1. Queries Sentinel-2 Level-2A scenes intersecting the area and season.
2. Excludes edge tiles with substantial no-data coverage.
3. Reads the Scene Classification Layer over the exact area.
4. Chooses the candidate with the lowest local cloud and cloud-shadow coverage.
5. Reads the true-color Cloud Optimized GeoTIFF window and renders a captioned frame.

After all frame tasks complete, a final task assembles the frames chronologically into the GIF and writes the JSON scene manifest.

## Run the workflow

Install dependencies and configure a Tilebox API key:

```bash
uv sync
export TILEBOX_API_KEY="YOUR_TILEBOX_API_KEY"
```

Build and publish the workflow release:

```bash
tilebox workflow build-release --json
tilebox workflow publish-release --json
```

The workflow task definitions and runner are in [`runner.py`](runner.py).
