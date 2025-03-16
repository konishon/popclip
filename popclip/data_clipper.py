import os
import logging
import shutil
import geopandas as gpd
import rasterio
from rasterio.mask import mask

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
    def __init__(self, input_path, geojson_clip_path, output_folder):
        self.logger = get_logger()
        self.logger.info("Initializing DataClipper.")

        self.input_path = os.path.abspath(input_path)
        self.geojson_clip_path = os.path.abspath(geojson_clip_path)
        self.output_folder = os.path.abspath(output_folder)
        os.makedirs(self.output_folder, exist_ok=True)

        self.geojson = self.load_geojson(self.geojson_clip_path)
        self.logger.info("Initialization complete.")

    def load_geojson(self, geojson_path):
        """Loads the GeoJSON clipping file."""
        if not os.path.exists(geojson_path):
            self.logger.error("GeoJSON file not found: %s", geojson_path)
            raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")
        try:
            gdf = gpd.read_file(geojson_path)
            if gdf.empty:
                raise ValueError("GeoJSON file is empty.")
            if gdf.crs is None:
                self.logger.warning("GeoJSON has no CRS, assuming EPSG:4326")
                gdf.set_crs("EPSG:4326", inplace=True)
            self.logger.info("GeoJSON successfully loaded.")
            return gdf
        except Exception as e:
            self.logger.error("Error loading GeoJSON: %s", e)
            raise

    def extract_zip(self, zip_path):
        """Extracts ZIP file if necessary and returns the first valid vector or raster file inside."""
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
        """Clips vector data using the GeoJSON and saves the result."""
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
        """Clips raster data using the GeoJSON and saves the result."""
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
        """Determines file type and processes accordingly."""
        if not os.path.exists(self.input_path):
            self.logger.error("Input file not found: %s", self.input_path)
            return
        
        file_name = os.path.basename(self.input_path)
        output_path = os.path.join(self.output_folder, file_name)

        if self.input_path.endswith(".zip"):
            extracted_dir = self.extract_zip(self.input_path)
            input_path = next(
                (os.path.join(extracted_dir, f) for f in os.listdir(extracted_dir) if f.lower().endswith((".tif", ".shp"))), 
                None
            )
            if not input_path:
                self.logger.warning("No valid files found in extracted ZIP: %s", self.input_path)
                return
        else:
            input_path = self.input_path

        ext = os.path.splitext(input_path)[1].lower()
        if ext in [".tif", ".tiff"]:
            self.clip_raster(input_path, f"{output_path}.tif")
        elif ext in [".shp", ".geojson"]:
            self.clip_vector(input_path, f"{output_path}.geojson")
        else:
            self.logger.warning("Unsupported file format: %s", input_path)
