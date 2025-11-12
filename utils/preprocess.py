import os
import yaml
import argparse
import numpy as np
import pickle
from pyproj import Geod
from tool_funcs import batch_trajectories_to_unique_tiles
import torch
import torch.nn as nn
from torch_geometric.nn import Node2Vec
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


geod = Geod(ellps="WGS84")


def extend_traj(track):
    num_points = len(track)
    track_extended = np.zeros((num_points, 6))
    for i, (lon, lat) in enumerate(track):
        track_extended[i, 0:2] = [lon, lat]
        if i > 0:
            lon_prev, lat_prev = track[i - 1]
            az12, az21, dist_prev = geod.inv(lon_prev, lat_prev, lon, lat)
            track_extended[i, 2] = dist_prev
            track_extended[i, 3] = az12
        else:
            lon_next, lat_next = track[i + 1]
            az12_next, az21_next, dist_next = geod.inv(lon, lat, lon_next, lat_next)
            track_extended[i, 2] = dist_next
            track_extended[i, 3] = az12_next

        if i < num_points - 1:
            lon_next, lat_next = track[i + 1]
            az12_next, az21_next, dist_next = geod.inv(lon, lat, lon_next, lat_next)
            track_extended[i, 4] = dist_next
            track_extended[i, 5] = az12_next
        else:
            lon_prev, lat_prev = track[i - 1]
            az12, az21, dist_prev = geod.inv(lon_prev, lat_prev, lon, lat)
            track_extended[i, 4] = dist_prev
            track_extended[i, 5] = az12

    return track_extended.tolist()


def latlon_to_mercator(lon, lat):
    R = 6378137.0
    x = R * np.radians(lon)
    y = R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y


def normalize_mercator_coords(lon, lat, x_min, x_max, y_min, y_max):
    x, y = latlon_to_mercator(lon, lat)
    x_norm = (x - x_min) / (x_max - x_min)
    y_norm = (y - y_min) / (y_max - y_min)
    return x_norm, y_norm


def extend_traj_with_config(track, config):
    mbr = config["mbr"]
    x_min, y_min = latlon_to_mercator(mbr["min_lon"], mbr["min_lat"])
    x_max, y_max = latlon_to_mercator(mbr["max_lon"], mbr["max_lat"])

    num_points = len(track)
    track_extended = np.zeros((num_points, 8))

    for i, (lon, lat) in enumerate(track):

        x_norm, y_norm = normalize_mercator_coords(lon, lat, x_min, x_max, y_min, y_max)
        track_extended[i, 0:2] = [x_norm, y_norm]

        track_extended[i, 6:8] = [lon, lat]

        if i > 0:
            lon_prev, lat_prev = track[i - 1]
            az12, _, dist_prev = geod.inv(lon_prev, lat_prev, lon, lat)
        else:
            lon_next, lat_next = track[i + 1]
            az12, _, dist_prev = geod.inv(lon, lat, lon_next, lat_next)

        if i < num_points - 1:
            lon_next, lat_next = track[i + 1]
            az12_next, _, dist_next = geod.inv(lon, lat, lon_next, lat_next)
        else:
            lon_prev, lat_prev = track[i - 1]
            az12_next, _, dist_next = geod.inv(lon_prev, lat_prev, lon, lat)

        track_extended[i, 2] = dist_prev / 1000.0
        az12 = (az12 + 360) % 360
        track_extended[i, 3] = az12 / 360.0
        track_extended[i, 4] = dist_next / 1000.0
        track_extended[i, 5] = az12_next / 180.0

    return track_extended.tolist()


def build_edge_index_from_traj(traj_idx: list[list[int]]) -> torch.Tensor:
    edge_set = set()

    for traj in traj_idx:
        for i in range(len(traj) - 1):
            src, dst = traj[i], traj[i + 1]
            if src != dst:
                edge_set.add((src, dst))
                edge_set.add((dst, src))

    src_list, dst_list = zip(*edge_set) if edge_set else ([], [])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    return edge_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="porto")
    yaml_path = f"../config/{parser.parse_args().dataset}.yaml"

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    filepath = os.path.join("..", config["file_path"])
    print(config["mbr"])
    with open(filepath, "rb") as f:
        trajs = pickle.load(f)  # [[lon,lat],[lon,lat].......]
    ## 4.2 Point-wise Encoder: Feature Extraction
    enriched_trajs = [extend_traj_with_config(traj, config) for traj in trajs]

    outputpath = os.path.join("..", config["enriched_file_path"])
    with open(outputpath, "wb") as f:
        pickle.dump(enriched_trajs, f)
    ## 4.1 Region-wise Encoder: Visual Context Embedding
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.00779, 0.00772, 0.00736], std=[0.0364, 0.0288, 0.0271]
            ),
        ]
    )  ##mean/std in images

    image_folder = os.path.join("..", config["tiles_path"])

    image_paths = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.endswith((".jpg", ".png"))
    ]

    embeddings_dict = {}

    for img_path in tqdm(image_paths, desc="Extracting Features"):
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            embedding = resnet(image).view(-1).numpy()

        embeddings_dict[os.path.splitext(os.path.basename(img_path))[0]] = embedding

    emd_path = os.path.join("..", config["grid_emb_path"])
    with open(emd_path, "wb") as f:
        pickle.dump(embeddings_dict, f)

    with open("../config/train.yaml", encoding="utf-8") as f:
        train_config = yaml.safe_load(f)
    with open(f"../{config['enriched_file_path']}", "rb") as f:
        trajs = pickle.load(f)
    train_trajs = trajs[: int(train_config["train_ratio"] * len(trajs))]
    print(len(train_trajs))
    zoom = config["zoom"]
    traj_tiles = batch_trajectories_to_unique_tiles(train_trajs, zoom)
    image_folder = f"../{config['tiles_path']}"
    all_image_names = sorted(
        [
            os.path.splitext(f)[0]
            for f in os.listdir(image_folder)
            if os.path.isfile(os.path.join(image_folder, f))
        ]
    )

    imgname_to_id = {name: idx for idx, name in enumerate(all_image_names)}
    id_to_imgname = {idx: name for idx, name in enumerate(all_image_names)}
    traj_idx = [[imgname_to_id[tile] for tile in traj] for traj in traj_tiles]
    print(traj_idx[0])
    edge_index = build_edge_index_from_traj(traj_idx)
    num_nodes = len(imgname_to_id)
    ## 4.1 Region-wise Encoder: Structual Context Embedding
    model = Node2Vec(
        edge_index=edge_index,
        embedding_dim=train_config["dim"],
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=1,
        q=1,
        num_nodes=num_nodes,
        sparse=True,
    ).to(train_config["device"])
    epochs = 50
    patience = 5
    best_loss = float("inf")
    worse_count = 0

    loader = model.loader(batch_size=128, shuffle=True)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(
                pos_rw.to(train_config["device"]), neg_rw.to(train_config["device"])
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1:02d} | Loss: {total_loss:.4f}")

        if total_loss < best_loss:
            best_loss = total_loss
            worse_count = 0
        else:
            worse_count += 1
            if worse_count >= patience:
                print(f"Early stopp at epoch {epoch + 1:02d}")
                break

    model.eval()
    with torch.no_grad():
        embeddings = model().cpu()

    embedding_dict = {
        id_to_imgname[idx]: embeddings[idx].numpy() for idx in range(num_nodes)
    }

    with open(f"../{config['graph_emb_path']}", "wb") as f:
        pickle.dump(embedding_dict, f)

    print(f"Node2Vec embedding saved as {config['graph_emb_path']}")
