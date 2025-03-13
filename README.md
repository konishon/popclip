## Usage:

Install dependencies and set up your project:

```bash
poetry install
```

Then use your script via Python or integrate into other applications:

```python
from popclip.population_raster_clipper import PopulationRasterClipper

clipper = PopulationRasterClipper(data_folder="./data")
result = clipper.clip_raster(
    year="2020",
    geojson_path="your_geojson.geojson",
    output_folder="./output"
)

print(f"Result available at {result}")
```