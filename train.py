import logging
import sys
from utils.tool_funcs import get_config, log_file_name
import argparse
from Trainer import TrajTrainer
import numpy as np
import torch
import random


# set random seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser()
    dataset_name = "porto"
    parser.add_argument("--dataset", type=str, default=dataset_name)
    parser.add_argument("--metric", type=str, default="dtw")
    dataset, metric = parser.parse_args().dataset, parser.parse_args().metric
    config = get_config(dataset, metric)
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
        handlers=[
            logging.FileHandler(
                "./exp/log/"
                + log_file_name(
                    config["batch_size"], config["num_layers"], config["dim"]
                ),
                mode="w",
            ),
            logging.StreamHandler(),
        ],
    )

    logging.info("python " + " ".join(sys.argv))
    logging.info("=================================")
    traj = TrajTrainer(config)
    traj.train()
    metrics = traj.test()
    print(metrics)
