import os
import logging
import itertools
import shutil
import yaml
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def get_logger():
    """Singleton logger setup to prevent duplicate log handlers."""
    logger = logging.getLogger("DataClipper")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

class DataClipper:
    def __init__(self, yaml_config_path, geojson_clip_path, output_folder, download=True):
        self.logger = get_logger()
        self.logger.info("Initializing DataClipper.")
        
        self.yaml_config_path = os.path.abspath(yaml_config_path)
        self.geojson_clip_path = os.path.abspath(geojson_clip_path)
        self.output_folder = os.path.abspath(output_folder)
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.data_folder = os.path.abspath("data")
        os.makedirs(self.data_folder, exist_ok=True)
        
        self.geojson = self.load_geojson(self.geojson_clip_path)
        self.download = download
        self.logger.info("Initialization complete. Global download enabled: %s", self.download)

    def load_geojson(self, geojson_path):
        if not os.path.exists(geojson_path):
            self.logger.error("GeoJSON file not found: %s", geojson_path)
            raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")
        try:
            gdf = gpd.read_file(geojson_path)
            if gdf.empty:
                raise ValueError("GeoJSON file is empty.")
            if gdf.crs is None:
                self.logger.warning("GeoJSON file has no CRS defined, assuming EPSG:4326")
                gdf.set_crs("EPSG:4326", inplace=True)
            self.logger.info("GeoJSON successfully loaded.")
            return gdf
        except Exception as e:
            self.logger.error("Error loading GeoJSON: %s", e)
            raise

    def robust_download(self, url, local_path, retries=3):
        if os.path.exists(local_path):
            self.logger.info("File already exists, skipping download: %s", local_path)
            return

        temp_path = local_path + '.part'
        for attempt in range(retries):
            try:
                import urllib.request
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as response, open(temp_path, "wb") as f:
                    shutil.copyfileobj(response, f)
                os.replace(temp_path, local_path)
                self.logger.info("Download complete: %s", local_path)
                return
            except Exception as e:
                self.logger.warning("Attempt %d failed: %s", attempt + 1, e)
                if attempt == retries - 1:
                    raise

    def extract_zip(self, zip_path):
        import zipfile
        extract_dir = os.path.splitext(zip_path)[0]
        if os.path.exists(extract_dir) and os.listdir(extract_dir):
            self.logger.info("ZIP already extracted, skipping: %s", extract_dir)
            return extract_dir
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        self.logger.info("Extracted ZIP file to: %s", extract_dir)
        return extract_dir

    def clip_vector(self, input_path, output_path):
        if not output_path.endswith(".geojson"):
            output_path += ".geojson"
        if os.path.exists(output_path):
            self.logger.info("Vector file already clipped, skipping: %s", output_path)
            return
        
        self.logger.info("Clipping vector data: %s", input_path)
        try:
            gdf = gpd.read_file(input_path)
            if gdf.crs is None:
                self.logger.warning("Vector file has no CRS, assuming EPSG:4326")
                gdf.set_crs("EPSG:4326", inplace=True)
            if gdf.crs != self.geojson.crs:
                gdf = gdf.to_crs(self.geojson.crs)

            clipped = gpd.overlay(gdf, self.geojson, how="intersection")
            if clipped.empty:
                self.logger.warning("Clipped vector data is empty: %s", input_path)
            else:
                clipped.to_file(output_path, driver="GeoJSON")
                self.logger.info("Vector data clipped and saved: %s", output_path)
        except Exception as e:
            self.logger.error("Error processing vector file %s: %s", input_path, e)

    def clip_raster(self, input_path, output_path):
        if os.path.exists(output_path):
            self.logger.info("Raster file already clipped, skipping: %s", output_path)
            return
        
        self.logger.info("Clipping raster data: %s", input_path)
        try:
            with rasterio.open(input_path) as src:
                clip_geom = self.geojson.to_crs(src.crs) if self.geojson.crs != src.crs else self.geojson
                out_image, out_transform = mask(src, clip_geom.geometry, crop=True, nodata=src.nodata)
                if out_image.size == 0:
                    self.logger.warning("Clipped raster data is empty: %s", input_path)
                    return

                clipped_meta = src.meta.copy()
                clipped_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "nodata": src.nodata
                })

            with rasterio.open(output_path, "w", **clipped_meta) as dest:
                dest.write(out_image)
            self.logger.info("Raster clipped and saved at %s", output_path)
        except Exception as e:
            self.logger.error("Error clipping raster %s: %s", input_path, e)

    def process(self):
        self.logger.info("Processing datasets from YAML configuration.")
        with open(self.yaml_config_path, 'r') as f:
            datasets = yaml.safe_load(f)

        def process_dataset(data):
            name = data.get("name")
            if not name:
                self.logger.error("Dataset entry missing 'name' field. Skipping.")
                return

            path_or_url = data.get("path")
            if not path_or_url:
                self.logger.error("Dataset %s missing 'path' field. Skipping.", name)
                return

            dataset_download = data.get("download", self.download)
            self.logger.info("Processing dataset: %s (download=%s)", name, dataset_download)

            data_save_filename = data.get("data_save_filename")
            final_output = os.path.join(self.output_folder, data_save_filename if data_save_filename else f"{name}_clipped")

            os.makedirs(os.path.dirname(final_output), exist_ok=True)

            if not path_or_url.startswith("http"):
                input_path = os.path.abspath(path_or_url)
                if not os.path.exists(input_path):
                    self.logger.error("Local file not found: %s", input_path)
                    return
            else:
                input_filename = os.path.basename(path_or_url)
                input_path = os.path.join(self.data_folder, input_filename)

                if os.path.exists(final_output):
                    self.logger.info("Final output %s already exists; skipping dataset %s.", final_output, name)
                    return

                if not os.path.exists(input_path) and dataset_download:
                    self.robust_download(path_or_url, input_path)

            if os.path.splitext(input_path)[1].lower() == ".zip":
                extracted_dir = self.extract_zip(input_path)
                input_path = next((os.path.join(extracted_dir, f) for f in os.listdir(extracted_dir) if f.lower().endswith((".tif", ".shp"))), None)
                if not input_path:
                    self.logger.warning("No valid files found in extracted ZIP for %s", name)
                    return

            ext = os.path.splitext(input_path)[1].lower()
            if ext in [".tif", ".tiff"]:
                self.clip_raster(input_path, f"{final_output}.tif")
            elif ext in [".shp", ".geojson"]:
                self.clip_vector(input_path, f"{final_output}.geojson")

        with ThreadPoolExecutor() as executor:
            executor.map(process_dataset, datasets)
