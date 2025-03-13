import yaml
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm
import shutil
import logging
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataClipper:
    def __init__(self, yaml_config_path, geojson_clip_path, output_folder):
        self.yaml_config_path = Path(yaml_config_path)
        self.geojson_clip_path = geojson_clip_path
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.data_folder = Path("data")
        self.data_folder.mkdir(parents=True, exist_ok=True)
        self.geojson = gpd.read_file(self.validate_path(geojson_clip_path))
        logging.info("Initialization complete.")

    def validate_path(self, path):
        p = Path(path)
        if not p.exists():
            logging.error("File not found: %s", path)
            raise FileNotFoundError(f"File not found: {path}")
        logging.info("Validated path: %s", path)
        return p

    def robust_download(self, url, local_path, retries=3):
        if local_path.exists():
            logging.info("File already exists, skipping download: %s", local_path)
            return

        temp_path = Path(str(local_path) + ".part")
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req) as response:
                    total_size = int(response.headers.get('content-length', 0))
                    with open(temp_path, "wb") as f, tqdm(
                        total=total_size, unit='B', unit_scale=True, desc=temp_path.name
                    ) as bar:
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                            bar.update(len(chunk))
                shutil.move(temp_path, local_path)
                logging.info("Download complete: %s", local_path)
                return
            except Exception as e:
                logging.warning("Attempt %d failed: %s", attempt + 1, e)
                if attempt == retries - 1:
                    raise

    def extract_zip(self, zip_path):
        extract_dir = zip_path.with_suffix('')
        if extract_dir.exists():
            logging.info("ZIP already extracted, skipping: %s", extract_dir)
            return extract_dir
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        logging.info("Extracted ZIP file to: %s", extract_dir)
        return extract_dir

    def clip_vector(self, input_path, output_path):
        if output_path.with_suffix('.geojson').exists():
            logging.info("Vector file already clipped, skipping: %s", output_path)
            return
        
        logging.info("Clipping vector data: %s", input_path)
        gdf = gpd.read_file(input_path)
        if gdf.crs != self.geojson.crs:
            logging.info("Reprojecting vector data to match GeoJSON CRS.")
            gdf = gdf.to_crs(self.geojson.crs)

        clipped = gdf.clip(self.geojson)
        if clipped.empty:
            logging.warning("Clipped vector data is empty: %s", input_path)
        else:
            clipped.to_file(output_path.with_suffix('.geojson'), driver="GeoJSON")
        logging.info("Vector data clipped and saved: %s", output_path)

    def clip_raster(self, input_path, output_path):
        if output_path.exists():
            logging.info("Raster file already clipped, skipping: %s", output_path)
            return
        
        logging.info("Clipping raster data: %s", input_path)
        with rasterio.open(input_path) as src:
            if self.geojson.crs != src.crs:
                logging.info("Reprojecting GeoJSON to raster CRS.")
                clip_geom = self.geojson.to_crs(src.crs)
            else:
                clip_geom = self.geojson

            out_image, out_transform = mask(src, clip_geom.geometry, crop=True)
            if out_image.size == 0:
                logging.warning("Clipped raster data is empty: %s", input_path)
                return

            clipped_meta = src.meta.copy()
            clipped_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

        with rasterio.open(output_path, "w", **clipped_meta) as dest:
            dest.write(out_image)
        logging.info("Raster clipped and saved at %s", output_path)

    def process(self):
        logging.info("Processing datasets from YAML configuration.")
        with open(self.validate_path(self.yaml_config_path), 'r') as f:
            datasets = yaml.safe_load(f)

        def process_dataset(data):
            name = data.get("name")
            source = data.get("source")
            path_or_url = data.get("path")
            description = data.get("description", "")

            logging.info("Processing: %s from %s. Description: %s", name, source, description)

            input_path = self.data_folder / Path(path_or_url).name
            if path_or_url.startswith("http") and not input_path.exists():
                self.robust_download(path_or_url, input_path)

            if input_path.suffix.lower() == ".zip":
                extracted_dir = self.extract_zip(input_path)
                tif_files = list(extracted_dir.glob("*.tif"))
                shapefiles = list(extracted_dir.glob("*.shp"))
                input_path = tif_files[0] if tif_files else shapefiles[0] if shapefiles else None
                if input_path is None:
                    logging.warning("No valid files found in extracted ZIP: %s", input_path)
                    return

            output_path = self.output_folder / f"{name}_clipped"
            output_path.mkdir(parents=True, exist_ok=True)

            try:
                if input_path.suffix.lower() in [".tif", ".tiff", ".img", ".vrt"]:
                    self.clip_raster(input_path, output_path / f"{name}_clipped.tif")
                elif input_path.suffix.lower() in [".shp", ".geojson", ".gpkg", ".kml"]:
                    self.clip_vector(input_path, output_path / f"{name}_clipped.geojson")
                else:
                    logging.warning("Unsupported file format: %s", input_path)
            except Exception as e:
                logging.error("Error processing %s: %s", name, e)

        with ThreadPoolExecutor() as executor:
            executor.map(process_dataset, datasets)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clip raster and vector data using GeoJSON boundary.")
    parser.add_argument("yaml_config", help="Path to YAML configuration file")
    parser.add_argument("geojson_clip", help="Path to GeoJSON clip file")
    parser.add_argument("output_folder", help="Output directory for clipped data")

    args = parser.parse_args()

    clipper = DataClipper(args.yaml_config, args.geojson_clip, args.output_folder)
    clipper.process()
