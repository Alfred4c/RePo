import yaml
import math


def log_file_name(batch_size=None, lr=None, num_layer=None, dim=None, cls_token=None):
    import time

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}_bs{batch_size}_lr{lr}_layer{num_layer}_dim{dim}_cls{cls_token}.log"


def load_yaml(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_config(dataset, metric):
    train_config = load_yaml("./config/train.yaml")
    config = load_yaml(f"./config/{dataset}.yaml")
    config.update(train_config)
    config["metric"] = metric
    return config

# map point into grid
def latlon_to_tile(lat, lon, zoom):

    n = 2.0**zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return f"{x}_{y}"


def batch_trajectories_to_unique_tiles(batch_trajectories, zoom):

    def single_traj_to_tiles(trajectory):
        tiles = []
        prev_tile = None
        for lon, lat in trajectory:
            if lat == 0.0 and lon == 0.0:
                break
            tile = latlon_to_tile(lat, lon, zoom)
            if tile != prev_tile:
                tiles.append(tile)
                prev_tile = tile
        return tiles

    return [single_traj_to_tiles(traj) for traj in batch_trajectories]


def lonlat_to_mercator(lon, lat):
    R = 6378137

    x = math.radians(lon) * R
    y = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * R

    return x, y
