import torch
from openood.pipelines.train_pipeline import TrainPipeline
from psychophysical_trainer import PsychophysicalTrainer
from psycho_imglist_dataset import ImagenetCPsychoDataset


class TrainPsychophysicalPipeline(TrainPipeline):
    """
    Pipeline for training models on ImageNet + ImageNet-C with psychophysical weighting.
    Integrates ImagenetCPsychoDataset and PsychophysicalTrainer.
    """

    def __init__(self, config):
        super().__init__(config)

        print("\n[OpenOOD] Initializing Psychophysical Training Pipeline...")

        # --- Build Dataset ---
        print("→ Loading psychophysical ImageNet-C dataset...")
        self.train_loader = self._build_dataloader(split_name="train")
        self.val_loader = self._build_dataloader(split_name="val")

        # --- Build Network ---
        print("→ Building network architecture...")
        self.net = self._build_network()

        # --- Build Optimizer / Scheduler ---
        self.optimizer, self.scheduler = self._build_optimizer_scheduler(self.net)

        # --- Build Trainer ---
        print("→ Initializing PsychophysicalTrainer...")
        self.trainer = PsychophysicalTrainer(config)
        self.trainer.net = self.net
        self.trainer.optimizer = self.optimizer
        self.trainer.scheduler = self.scheduler
        self.trainer.train_loader = self.train_loader
        self.trainer.val_loader = self.val_loader

    def _build_dataloader(self, split_name):
        """Constructs a DataLoader for the given split using ImagenetCPsychoDataset."""
        dataset = ImagenetCPsychoDataset(self.config["dataset"][split_name], split_name=split_name)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config["dataset"][split_name]["batch_size"],
            shuffle=self.config["dataset"][split_name]["shuffle"],
            num_workers=int(self.config["dataset"]["num_workers"]),
            pin_memory=True,
        )

    def run(self):
        """
        Standard training loop compatible with OpenOOD logging.
        """
        print(f"\n[OpenOOD] Starting psychophysical training for {self.config['trainer']['num_epochs']} epochs...\n")

        for epoch in range(self.config["trainer"]["num_epochs"]):
            print(f"\n===== Epoch {epoch+1}/{self.config['trainer']['num_epochs']} =====")

            train_stats = self.trainer.train_epoch()
            val_stats = self.trainer.eval_epoch()

            # Step LR scheduler if defined
            if self.scheduler is not None:
                self.scheduler.step()

            # Logging
            self.logger.info(
                f"[Epoch {epoch+1}] "
                f"Train Loss: {train_stats['train_loss']:.4f} | Train Acc: {train_stats['train_acc']:.3f} | "
                f"Val Loss: {val_stats['val_loss']:.4f} | Val Acc: {val_stats['val_acc']:.3f}"
            )

            # Save best model
            if val_stats["val_acc"] > self.best_acc:
                self.best_acc = val_stats["val_acc"]
                self.save_checkpoint(tag="best")

        print("\nTraining completed.")
        print(f"Best validation accuracy: {self.best_acc:.3f}")
        self.save_checkpoint(tag="final")
