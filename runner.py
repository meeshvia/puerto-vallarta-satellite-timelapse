import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer
from shapely import Polygon
from tilebox.datasets import Client as DatasetClient
from tilebox.datasets import field, iter_datapoints
from tilebox.datasets.assets import AssetCollection
from tilebox.storage.aio import Client as StorageClient
from tilebox.storage.geotiff import window_from_bounds
from tilebox.workflows import Client, ExecutionContext, Runner, Task
from tilebox.workflows.cache import LocalFileSystemCache


DATASET = "open_data.aws_earth.sentinel2"
COLLECTION = "L2A"
SEASONS = ("Winter", "Spring", "Summer", "Autumn")
TASK_PREFIX = "puerto-vallarta-seasonal-timelapse"


@dataclass(frozen=True)
class Period:
    label: str
    start: str
    end: str


def seasonal_periods(start: str, end: str) -> list[Period]:
    start_date = datetime.fromisoformat(start)
    end_date = datetime.fromisoformat(end)
    periods: list[Period] = []
    current = start_date
    while current < end_date:
        next_month = current.month + 3
        next_year = current.year
        if next_month > 12:
            next_month -= 12
            next_year += 1
        period_end = min(datetime(next_year, next_month, 1), end_date)
        season = SEASONS[(current.month % 12) // 3]
        year = current.year if season != "Winter" else period_end.year
        periods.append(
            Period(
                label=f"{season} {year}",
                start=current.date().isoformat(),
                end=period_end.date().isoformat(),
            )
        )
        current = period_end
    return periods


def square_aoi(
    longitude: float, latitude: float, size_meters: int
) -> tuple[Polygon, tuple[float, float, float, float]]:
    zone = int((longitude + 180) // 6) + 1
    epsg = 32600 + zone if latitude >= 0 else 32700 + zone
    to_utm = Transformer.from_crs(4326, epsg, always_xy=True)
    to_wgs84 = Transformer.from_crs(epsg, 4326, always_xy=True)
    center_x, center_y = to_utm.transform(longitude, latitude)
    half = size_meters / 2
    corners = [
        to_wgs84.transform(x, y)
        for x, y in (
            (center_x - half, center_y - half),
            (center_x + half, center_y - half),
            (center_x + half, center_y + half),
            (center_x - half, center_y + half),
        )
    ]
    polygon = Polygon(corners)
    return polygon, polygon.bounds


def scalar(datapoint, name: str):
    return datapoint[name].item()


async def select_and_render(
    period: Period,
    location_name: str,
    aoi: Polygon,
    bounds: tuple[float, float, float, float],
    storage: StorageClient,
) -> tuple[Image.Image, dict[str, object]]:
    scenes = DatasetClient().dataset(DATASET).query(
        collections=[COLLECTION],
        temporal_extent=(period.start, period.end),
        spatial_extent=aoi,
        filter=(field("cloud_cover") < 40)
        & (field("nodata_pixel_percentage") < 30),
    )
    candidates = sorted(
        iter_datapoints(scenes),
        key=lambda item: float(scalar(item, "cloud_cover")),
    )[:30]
    if not candidates:
        raise RuntimeError(f"No Sentinel-2 candidates found for {period.label}")

    best = None
    for datapoint in candidates:
        assets = AssetCollection.from_datapoint(datapoint)
        scl_geotiff = await storage.open_geotiff(assets["scl"])
        try:
            scl_window = window_from_bounds(
                scl_geotiff,
                bounds=bounds,
                crs="EPSG:4326",
                require_fully_contained=True,
            )
        except ValueError:
            continue
        scl = (await scl_geotiff.read(window=scl_window)).data[0]
        observed = scl != 0
        coverage = float(observed.mean() * 100)
        if coverage < 98:
            continue
        obscured = np.isin(scl, [3, 8, 9, 10])
        local_cloud = float(obscured[observed].mean() * 100)
        if best is None or local_cloud < best[0]:
            best = (local_cloud, coverage, datapoint, assets)
        if local_cloud <= 1:
            break

    if best is None:
        raise RuntimeError(f"No scene fully covers the AOI for {period.label}")

    local_cloud, coverage, datapoint, assets = best
    visual_geotiff = await storage.open_geotiff(assets["visual"])
    visual_window = window_from_bounds(
        visual_geotiff,
        bounds=bounds,
        crs="EPSG:4326",
        require_fully_contained=True,
    )
    pixels = (await visual_geotiff.read(window=visual_window)).data
    rgb = np.moveaxis(pixels[:3], 0, -1).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB").resize(
        (768, 768), Image.Resampling.LANCZOS
    )

    acquired = np.datetime_as_string(datapoint.time.values, unit="D")
    metadata_cloud = float(scalar(datapoint, "cloud_cover"))
    scene_id = str(scalar(datapoint, "stac_id"))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 706, 768, 768), fill=(0, 0, 0, 170))
    draw.text(
        (18, 714),
        f"{location_name} · {period.label}",
        fill="white",
        font=ImageFont.load_default(size=22),
    )
    draw.text(
        (18, 743),
        f"Sentinel-2 · {acquired} · AOI cloud {local_cloud:.1f}%",
        fill=(225, 235, 245),
        font=ImageFont.load_default(size=16),
    )
    return image, {
        "season": period.label,
        "acquired": acquired,
        "scene_id": scene_id,
        "aoi_coverage_percent": round(coverage, 3),
        "aoi_cloud_percent": round(local_cloud, 3),
        "scene_cloud_percent": round(metadata_cloud, 3),
    }


class RenderSeasonFrame(Task):
    index: int
    label: str
    period_start: str
    period_end: str
    location_name: str
    longitude: float
    latitude: float
    size_meters: int

    @staticmethod
    def identifier() -> tuple[str, str]:
        return f"{TASK_PREFIX}/RenderSeasonFrame", "v1.2"

    def execute(self, context: ExecutionContext) -> None:
        context.current_task.display = f"Render {self.label}"
        period = Period(self.label, self.period_start, self.period_end)
        aoi, bounds = square_aoi(self.longitude, self.latitude, self.size_meters)

        async def render() -> tuple[Image.Image, dict[str, object]]:
            with context.tracer.span(
                "select-seasonal-scene", attributes={"season": self.label}
            ):
                return await select_and_render(
                    period,
                    self.location_name,
                    aoi,
                    bounds,
                    StorageClient(),
                )

        frame, selected = asyncio.run(render())
        frame_bytes = BytesIO()
        frame.save(frame_bytes, format="PNG", optimize=True)
        cache_key = f"{self.index:02d}"
        context.job_cache.group("frames")[cache_key] = frame_bytes.getvalue()
        context.job_cache.group("metadata")[cache_key] = json.dumps(selected).encode()
        context.logger.info("Selected seasonal scene", **selected)
        context.progress("seasons").done(1)


class EncodeTimelapse(Task):
    location_name: str
    longitude: float
    latitude: float
    size_meters: int
    start: str
    end: str
    output_path: str

    @staticmethod
    def identifier() -> tuple[str, str]:
        return f"{TASK_PREFIX}/EncodeTimelapse", "v1.2"

    def execute(self, context: ExecutionContext) -> None:
        periods = seasonal_periods(self.start, self.end)
        context.current_task.display = f"Encode {len(periods)}-frame GIF"
        frame_cache = context.job_cache.group("frames")
        metadata_cache = context.job_cache.group("metadata")
        frames: list[Image.Image] = []
        manifest: list[dict[str, object]] = []
        for index, period in enumerate(periods):
            cache_key = f"{index:02d}"
            try:
                frames.append(
                    Image.open(BytesIO(frame_cache[cache_key])).convert("RGB")
                )
                selected = json.loads(metadata_cache[cache_key])
            except KeyError as error:
                raise RuntimeError(f"Missing rendered frame for {period.label}") from error
            if selected["season"] != period.label:
                raise RuntimeError(f"Unexpected frame order for {period.label}")
            manifest.append(selected)

        output = Path(self.output_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            duration=1200,
            loop=0,
            optimize=True,
        )
        aoi, _ = square_aoi(self.longitude, self.latitude, self.size_meters)
        output.with_suffix(".json").write_text(
            json.dumps(
                {
                    "dataset": DATASET,
                    "location": self.location_name,
                    "center": [self.longitude, self.latitude],
                    "size_meters": self.size_meters,
                    "aoi_geojson": aoi.__geo_interface__,
                    "scenes": manifest,
                },
                indent=2,
            )
        )
        context.logger.info("Timelapse complete", output=str(output), frames=len(frames))


class BuildSeasonalTimelapse(Task):
    location_name: str
    longitude: float
    latitude: float
    size_meters: int
    start: str
    end: str
    output_path: str

    @staticmethod
    def identifier() -> tuple[str, str]:
        return f"{TASK_PREFIX}/BuildSeasonalTimelapse", "v1.2"

    def execute(self, context: ExecutionContext) -> None:
        periods = seasonal_periods(self.start, self.end)
        context.current_task.display = f"Plan {len(periods)} seasonal frames"
        context.progress("seasons").add(len(periods))
        renders = context.submit_subtasks(
            [
                RenderSeasonFrame(
                    index=index,
                    label=period.label,
                    period_start=period.start,
                    period_end=period.end,
                    location_name=self.location_name,
                    longitude=self.longitude,
                    latitude=self.latitude,
                    size_meters=self.size_meters,
                )
                for index, period in enumerate(periods)
            ],
            max_retries=2,
        )
        context.submit_subtask(
            EncodeTimelapse(
                location_name=self.location_name,
                longitude=self.longitude,
                latitude=self.latitude,
                size_meters=self.size_meters,
                start=self.start,
                end=self.end,
                output_path=self.output_path,
            ),
            depends_on=renders,
        )


runner = Runner(
    tasks=[BuildSeasonalTimelapse, RenderSeasonFrame, EncodeTimelapse],
    cache=LocalFileSystemCache(
        Path.home() / ".cache/tilebox/puerto-vallarta-seasonal-timelapse"
    ),
)


if __name__ == "__main__":
    # direct runner mode, when started with `python runner.py`
    runner.connect_to(Client()).run_forever()
