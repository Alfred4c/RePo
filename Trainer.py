import torch
import os
import time
import logging
from utils.DataLoader import load_datasets
from model.RePo import RePo
import json
import argparse
from utils.accuracy import evaluate_similarity
import torch.nn.functional as F
from timm.scheduler import create_scheduler


class TrajTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config["device"]
        self.train_loader, self.val_loader, self.test_loader = load_datasets(config)
        self.model = RePo(config).to(config["device"])
        self.optimizer = torch.optim.Adam(
            self.get_trainable_params(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 1e-5),
        )
        self.scheduler, _ = create_scheduler(
            argparse.Namespace(
                epochs=config["epoch"],
                lr=config["learning_rate"],
                sched="cosine",
                warmup_lr=1e-6,
                warmup_epochs=10,
                cooldown_epochs=5,
                min_lr=1e-6,
                decay_rate=0.1,
                decay_epochs=30,
            ),
            self.optimizer,
        )

        self.early_stop_counter = 0
        self.early_stop_patience = config.get("early_stop_patience", 10)

        self.best_hr1 = -float("inf")
        self.best_model_path = os.path.join(
            config["best_model"],
            f"best_hr1_k_{config['hard_negative_topk']}_bs_{config['batch_size']}_dim_{config['dim']}_layers_{config['num_layers']}_{config['metric']}.pth",
        )
        self.metric_history = {
            "epoch": [],
            "HR@1": [],
            "HR@5": [],
            "HR@10": [],
        }

    def get_trainable_params(self):
        return list(self.model.parameters())

    def get_current_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def get_state_dict(self):
        return {"model": self.model.state_dict()}

    def load_best_model(self):
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        logging.info(f"Loaded best model from {self.best_model_path}")

    def save_metric_history(self, save_path="metric_history.json"):

        with open(save_path, "w") as f:
            json.dump(self.metric_history, f, indent=4)

        logging.info(f"Metric history saved to {save_path}")

    def train(self):
        logging.info("Training started")
        total_start_time = time.time()
        for epoch in range(1, self.config["epoch"] + 1):
            epoch_start_time = time.time()
            epoch_loss = 0
            self.model.train()

            for batch_idx, (traj, traj_length, dis_matrix, global_indices) in enumerate(
                self.train_loader, start=1
            ):
                batch_start_time = time.time()

                traj = traj.to(self.device)
                traj_length = traj_length.to(self.device)
                dis_matrix = dis_matrix.to(self.device)
                global_indices = global_indices.to(self.device)

                embeddings = self.model(traj, traj_length, global_indices)  # [B, D]

                embeddings = F.normalize(embeddings, dim=1, eps=1e-6)
                sim_matrix = embeddings @ embeddings.T  # [B, B]

                dis_matrix_masked = (
                    dis_matrix
                    + torch.eye(dis_matrix.size(0), device=dis_matrix.device) * 1e6
                )
                nearest_idx = torch.argmin(dis_matrix_masked, dim=1)
                pos_mask = F.one_hot(
                    nearest_idx, num_classes=dis_matrix.size(0)
                ).float()

                tau = max(self.config.get("temperature", 0.1), 1e-3)
                logits = sim_matrix / tau

                neg_mask = ~(
                    pos_mask.bool()
                    | torch.eye(logits.size(0), device=logits.device).bool()
                )
                logits_neg = logits.masked_fill(~neg_mask, float("-inf"))

                top_k = self.config.get("hard_negative_topk", 10)
                topk_logits, _ = torch.topk(
                    logits_neg, k=min(top_k, logits.size(1) - 1), dim=1
                )

                logsumexp_denom = torch.logsumexp(topk_logits, dim=1)  # [B]
                pos_logits = (logits * pos_mask).sum(dim=1)  # [B]

                loss = -(pos_logits - logsumexp_denom).mean()  # (Eq. 19 in paper)

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()

                batch_duration = time.time() - batch_start_time

                logging.info(
                    f'Epoch [{epoch}/{self.config["epoch"]}] Batch [{batch_idx}/{len(self.train_loader)}] - Loss: {loss.item():.4f} - Batch Time: {batch_duration:.2f}s'
                )
            epoch_duration = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / len(self.train_loader)

            logging.info(
                f'Epoch [{epoch}/{self.config["epoch"]}] - Avg Loss: {avg_epoch_loss:.4f} - LR: {self.get_current_lr():.6f} - Epoch Time: {epoch_duration:.2f}s'
            )

            self.scheduler.step(epoch)

            if epoch % self.config.get("eval_interval", 1) == 0:
                metrics = self.evaluate(mode="val", epoch=epoch)
                hr1 = metrics.get("HR@1", 0)
                hr5 = metrics.get("HR@5", 0)
                hr10 = metrics.get("HR@10", 0)
                eval_loss = metrics.get("eval_loss", 0)

                self.metric_history["epoch"].append(epoch)
                self.metric_history["HR@1"].append(hr1)
                self.metric_history["HR@5"].append(hr5)
                self.metric_history["HR@10"].append(hr10)

                trend_str = (
                    f"[Epoch {epoch}] eval loss: {eval_loss:.4f}| HR@1: {hr1:.4f} | HR@5: {hr5:.4f} | HR@10: {hr10:.4f} | "
                    f"Best HR@1 so far: {self.best_hr1:.4f}"
                )
                logging.info(trend_str)

                if hr1 > self.best_hr1:
                    self.best_hr1 = hr1
                    self.early_stop_counter = 0
                    torch.save(self.get_state_dict(), self.best_model_path)
                    logging.info(f"Best model updated with HR@10: {self.best_hr1:.4f}")
                else:
                    self.early_stop_counter += 1
                    logging.info(
                        f"No improvement in HR@1. Early stop counter: {self.early_stop_counter}/{self.early_stop_patience}"
                    )
                    if self.early_stop_counter >= self.early_stop_patience:
                        logging.info(f"Early stopping triggered at epoch {epoch}.")
                        break

        total_duration = time.time() - total_start_time
        logging.info(f"Training completed in {total_duration / 60:.2f} minutes.")

        self.save_metric_history()
        self.test()

    def test(self):
        metrics = self.evaluate(mode="test")
        logging.info(f"config:{self.config}")
        logging.info("[Full Eval - Test] Best model performance on test set:")
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.4f}")
        return metrics

    def evaluate(self, mode="val", epoch=None):
        import torch
        import torch.nn.functional as F
        from tqdm import tqdm
        import numpy as np

        if mode == "test":
            self.load_best_model()

        self.model.eval()

        if mode == "val":
            dataloader = self.val_loader
        elif mode == "test":
            dataloader = self.test_loader
        else:
            raise ValueError("mode must be 'val' or 'test'")

        all_embeddings = []
        all_global_indices = []

        with torch.no_grad():
            for padded_trajectories, trajectory_lengths, _, global_indices in tqdm(
                dataloader, desc=f"Embedding {mode.capitalize()} Set"
            ):
                padded_trajectories = padded_trajectories.to(self.config["device"])
                trajectory_lengths = trajectory_lengths.to(self.config["device"])
                global_indices = global_indices.to(self.config["device"])
                embeddings = self.model(
                    padded_trajectories, trajectory_lengths, global_indices
                )  # [B, D]
                all_embeddings.append(embeddings)

            for _, global_idx in dataloader.dataset:
                all_global_indices.append(global_idx)

        all_embeddings = torch.cat(all_embeddings, dim=0)  # [N, D]
        all_embeddings = F.normalize(all_embeddings, dim=1)

        pred_similarity = all_embeddings @ all_embeddings.T  # [N, N]

        full_sim_matrix = np.load(
            self.config["similarity_matrix_path"], allow_pickle=True
        )
        global_indices = dataloader.dataset.global_indices
        sub_sim_matrix = full_sim_matrix[np.ix_(global_indices, global_indices)]
        sub_sim_matrix_tensor = torch.tensor(
            sub_sim_matrix, dtype=torch.float32, device=pred_similarity.device
        )

        # Compute InfoNCE loss (for evaluation only)
        dis_matrix_masked = (
            sub_sim_matrix_tensor
            + torch.eye(
                sub_sim_matrix_tensor.size(0), device=sub_sim_matrix_tensor.device
            )
            * 1e6
        )
        # positive sample based on distance matrix
        nearest_idx = torch.argmin(dis_matrix_masked, dim=1)
        pos_mask = F.one_hot(nearest_idx, num_classes=dis_matrix_masked.size(0)).float()

        tau = max(self.config.get("temperature", 0.2), 1e-3)
        logits = pred_similarity / tau

        neg_mask = ~(
            pos_mask.bool() | torch.eye(logits.size(0), device=logits.device).bool()
        )
        logits_neg = logits.masked_fill(~neg_mask, float("-inf"))

        top_k = self.config.get("hard_negative_topk")
        # negative sample in representation space
        topk_logits, _ = torch.topk(logits_neg, k=min(top_k, logits.size(1) - 1), dim=1)

        logsumexp_denom = torch.logsumexp(topk_logits, dim=1)  # [B]
        pos_logits = (logits * pos_mask).sum(dim=1)  # [B]
        eval_loss = -(pos_logits - logsumexp_denom).mean().item()

        # Compute metrics
        target_similarity = (
            -sub_sim_matrix_tensor
            - torch.eye(
                sub_sim_matrix_tensor.size(0), device=sub_sim_matrix_tensor.device
            )
            * 1e6
        )
        metrics = evaluate_similarity(pred_similarity, target_similarity)
        metrics["eval_loss"] = eval_loss

        if mode == "val":
            logging.info(f"[Validation] Metrics after epoch {epoch}:")
        else:
            logging.info("[Test] Best model performance on test set:")

        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.4f}")

        return metrics

    # for scalability experiment, based on pretrained model on 10,000 trajectories.
    # def embedding(self):
    #     self.load_best_model()
    #     self.model.eval()
    #     self.model.tile = pickle.load(open(self.config["large_tile_path"], "rb"))
    #     dataloader = load_full_datasets(self.config)
    #     all_embeddings = []
    #     for traj, traj_length, indices in dataloader:
    #         traj = traj.to(self.config["device"])
    #         traj_length = traj_length.to(self.config["device"])
    #         indices = indices.to(self.config["device"])
    #         with torch.no_grad():
    #             embeddings = self.model(traj, traj_length, indices)  # [batch_size, D]
    #         embeddings = F.normalize(embeddings, dim=1)
    #         embeddings_cpu = embeddings.cpu()
    #         all_embeddings.append(embeddings_cpu)
    #         del traj, traj_length, embeddings
    #         torch.cuda.empty_cache()
    #     all_embeddings = torch.cat(all_embeddings, dim=0)  # [N, D] (on CPU)
    #     filename = "porto_20w_embeddings.pkl"
    #     with open(filename, "wb") as f:
    #         pickle.dump(all_embeddings, f)
    #     print("saved large embedding file!")
