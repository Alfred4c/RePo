import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class GPSDataset(Dataset):
    def __init__(self, trajectories, global_indices):
        self.trajectories = trajectories
        self.global_indices = global_indices

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = torch.tensor(self.trajectories[idx], dtype=torch.float32)
        global_idx = self.global_indices[idx]
        return traj, global_idx


def collate_fn(batch, similarity_matrix):
    trajectories, global_indices = zip(*batch)

    padded_trajectories = pad_sequence(
        trajectories, batch_first=True, padding_value=0.0
    )#padding
    trajectory_lengths = torch.tensor([len(t) for t in trajectories], dtype=torch.long)

    # similarity matrix for finding positive sample
    batch_similarity = similarity_matrix[np.ix_(global_indices, global_indices)]
    batch_similarity = torch.tensor(batch_similarity, dtype=torch.float32)

    return (
        padded_trajectories,
        trajectory_lengths,
        batch_similarity,
        torch.tensor(global_indices, dtype=torch.long),
    )


def load_datasets(config):
    import pickle
    import numpy as np
    from torch.utils.data import DataLoader

    with open(config["enriched_file_path"], "rb") as f:
        trajectories = pickle.load(f)

    with open(config["similarity_matrix_path"], "rb") as f:
        similarity_matrix = pickle.load(f)

    num_samples = len(trajectories)
    indices = list(range(num_samples))

    np.random.seed(config.get("split_seed", 42))
    np.random.shuffle(indices)

    train_end = int(config["train_ratio"] * num_samples)
    val_end = train_end + int(config["val_ratio"] * num_samples)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    def create_loader(indices_subset, shuffle):
        subset_trajectories = [trajectories[i] for i in indices_subset]
        dataset = GPSDataset(subset_trajectories, indices_subset)
        loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=shuffle,
            collate_fn=lambda batch: collate_fn(batch, similarity_matrix),
        )
        return loader

    train_loader = create_loader(train_indices, shuffle=True)#shuffle
    val_loader = create_loader(val_indices, shuffle=False)
    test_loader = create_loader(test_indices, shuffle=False)

    return train_loader, val_loader, test_loader