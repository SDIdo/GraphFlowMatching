"""
CSV logging of the flow-matching training loss.

The original training script only pushes losses to Weights & Biases, which makes
the loss curves unavailable offline.  ``CSVLossLogger`` mirrors the same numbers
into two plain CSV files under the run directory so the report can be rebuilt
without a wandb account:

    <run_dir>/train_log_steps.csv    step, epoch, loss, diffusion_term,
                                     reaction_term, lr, wall_time_s
    <run_dir>/train_log_epochs.csv   epoch, steps, avg_loss, avg_diffusion_term,
                                     avg_reaction_term, lr, fid, wall_time_s
    <run_dir>/train_log_fid.csv      step, epoch, fid, wall_time_s
"""

import csv
import os
import time


class CSVLossLogger:
    def __init__(self, run_dir, log_every_steps=10, resume=False):
        self.run_dir = run_dir
        self.log_every_steps = max(1, int(log_every_steps))
        os.makedirs(run_dir, exist_ok=True)
        self.t0 = time.time()

        self.step_path = os.path.join(run_dir, "train_log_steps.csv")
        self.epoch_path = os.path.join(run_dir, "train_log_epochs.csv")
        self.fid_path = os.path.join(run_dir, "train_log_fid.csv")

        self._init_file(self.step_path, ["step", "epoch", "loss",
                                         "diffusion_term", "reaction_term",
                                         "lr", "wall_time_s"], resume)
        self._init_file(self.epoch_path, ["epoch", "steps", "avg_loss",
                                          "avg_diffusion_term",
                                          "avg_reaction_term", "lr", "fid",
                                          "wall_time_s"], resume)
        self._init_file(self.fid_path, ["step", "epoch", "fid",
                                        "wall_time_s"], resume)

    @staticmethod
    def _init_file(path, header, resume):
        if resume and os.path.exists(path) and os.path.getsize(path) > 0:
            return
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    def _append(self, path, row):
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def log_step(self, step, epoch, loss, diffusion_term, reaction_term, lr,
                 force=False):
        if not force and step % self.log_every_steps != 0:
            return
        self._append(self.step_path,
                     [step, epoch, f"{loss:.8g}", f"{diffusion_term:.8g}",
                      f"{reaction_term:.8g}", f"{lr:.8g}",
                      f"{time.time() - self.t0:.2f}"])

    def log_epoch(self, epoch, steps, avg_loss, avg_diffusion_term,
                  avg_reaction_term, lr, fid=None):
        self._append(self.epoch_path,
                     [epoch, steps, f"{avg_loss:.8g}",
                      f"{avg_diffusion_term:.8g}", f"{avg_reaction_term:.8g}",
                      f"{lr:.8g}", "" if fid is None else f"{fid:.8g}",
                      f"{time.time() - self.t0:.2f}"])

    def log_fid(self, step, epoch, fid):
        self._append(self.fid_path,
                     [step, epoch, f"{fid:.8g}", f"{time.time() - self.t0:.2f}"])
