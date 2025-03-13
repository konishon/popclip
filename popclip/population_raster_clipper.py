import urllib.request
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm
import logging
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PopulationRasterClipper:
    RASTER_URLS = {
        "2020": "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/0_Mosaicked/ppp_2020_1km_Aggregated.tif",
        "2019": "https://data.worldpop.org/GIS/Population/Global_2000_2020/2019/0_Mosaicked/ppp_2019_1km_Aggregated.tif",
        "2018": "https://data.worldpop.org/GIS/Population/Global_2000_2020/2018/0_Mosaicked/ppp_2018_1km_Aggregated.tif"
    }

    def __init__(self, data_folder="data"):
        self.data_folder = Path(data_folder)
        self.data_folder.mkdir(parents=True, exist_ok=True)

    def robust_download(self, url, local_path, retries=3):
        temp_path = Path(str(local_path) + ".part")
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url)
                resume_byte_pos = temp_path.stat().st_size if temp_path.exists() else 0

                if resume_byte_pos:
                    req.add_header('Range', f'bytes={resume_byte_pos}-')
                    mode = 'ab'
                    logger.info("Resuming download from byte position %s", resume_byte_pos)
                else:
                    mode = 'wb'

                with urllib.request.urlopen(req) as response:
                    total_size = int(response.getheader('Content-Length', 0)) + resume_byte_pos
                    with open(temp_path, mode) as f, tqdm(
                        total=total_size, initial=resume_byte_pos,
                        unit='B', unit_scale=True, desc=temp_path.name
                    ) as bar:
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                            bar.update(len(chunk))

                shutil.move(temp_path, local_path)
                logger.info("Download completed and saved to %s", local_path)
                return
            except Exception as e:
                logger.warning("Attempt %d failed: %s", attempt + 1, e)
                if attempt < retries - 1:
                    logger.info("Retrying...")
        raise Exception("All download attempts failed.")

    def clip_raster(self, year, geojson_path, output_folder):
        raster_url = self.RASTER_URLS.get(str(year))
        if not raster_url:
            raise ValueError(f"No raster URL found for the year: {year}")

        raster_filename = raster_url.split('/')[-1]
        local_raster_path = self.data_folder / raster_filename

        if not local_raster_path.exists():
            self.robust_download(raster_url, local_raster_path)

        geojson = gpd.read_file(geojson_path)

        if geojson.empty:
            raise ValueError("GeoJSON file provided is empty.")

        geojson = geojson[geojson.is_valid]
        if geojson.empty:
            raise ValueError("No valid geometries in GeoJSON.")

        with rasterio.open(local_raster_path) as src:
            if geojson.crs is None:
                raise ValueError("GeoJSON file has no CRS defined. Please define a valid CRS.")

            if geojson.crs != src.crs:
                logger.info("Reprojecting GeoJSON to raster CRS.")
                geojson = geojson.to_crs(src.crs)

            clipped_image, clipped_transform = mask(src, geojson.geometry, crop=True)

            if clipped_image.size == 0:
                raise ValueError("Clipping resulted in an empty raster.")

            clipped_meta = src.meta.copy()
            clipped_meta.update({
                "driver": "GTiff",
                "height": clipped_image.shape[1],
                "width": clipped_image.shape[2],
                "transform": clipped_transform
            })

        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        clipped_raster_path = output_path / f"clipped_population_{year}.tif"

        with rasterio.open(clipped_raster_path, "w", **clipped_meta) as dst:
            dst.write(clipped_image)

        logger.info("Clipped raster saved at %s", clipped_raster_path)
        return clipped_raster_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and clip population raster data.")
    parser.add_argument("year", choices=["2018", "2019", "2020"], help="Year of raster data")
    parser.add_argument("geojson", help="Path to GeoJSON file")
    parser.add_argument("output_folder", help="Output directory for clipped raster")

    args = parser.parse_args()

    clipper = PopulationRasterClipper()
    raster_path = clipper.clip_raster(args.year, args.geojson, args.output_folder)
    logger.info("Raster available at: %s", raster_path)