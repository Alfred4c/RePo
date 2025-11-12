import math
import os
import requests
import time


def latlon_to_tile(lat, lon, zoom):
    n = 2.0**zoom
    x_tile = int((lon + 180.0) / 360.0 * n)
    y_tile = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x_tile, y_tile


zoom = 18
lat_range = (30.56, 30.62)
lon_range = (103.92, 104.07)
tile_server = "" ## choose a server

x_min, y_max = latlon_to_tile(lat_range[0], lon_range[0], zoom)
x_max, y_min = latlon_to_tile(lat_range[1], lon_range[1], zoom)

save_dir = "../data/tiles/chengdu"
os.makedirs(save_dir, exist_ok=True)
failed_urls = []

for x in range(x_min, x_max + 1):
    for y in range(y_min, y_max + 1):
        file_path = f"{save_dir}/{x}_{y}.png"
        if os.path.exists(file_path):
            print(f"Skip {x}-{y}, already exists.")
            continue

        url = tile_server.format(x=x, y=y, z=zoom)
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(response.content)
                print(f"Successfully {x}-{y}")
            else:
                print(f"HTTP Error {response.status_code}: {url}")
                ## save failed images
                failed_urls.append(url)
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            failed_urls.append(url)

        time.sleep(1)

if failed_urls:
    with open("chengdu_failed_urls.txt", "w") as f:
        for url in failed_urls:
            f.write(url + "\n")
    print(
        f"\nFailed URLs saved to chengdu_failed_urls.txt ({len(failed_urls)} records)"
    )

print(f"x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}")


save_dir = "../data/tiles/chengdu"
os.makedirs(save_dir, exist_ok=True)

