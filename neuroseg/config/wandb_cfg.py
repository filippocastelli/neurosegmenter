from neuroseg.config import TrainConfig
import wandb


class WandbConfigurator:

    def __init__(self,
                 train_config: TrainConfig):
        self.train_config = train_config
        if train_config.enable_wandb_tracking:
            self.train_config = train_config
            self.project_name = train_config.wandb_project
            self.entity = train_config.wandb_entity
            self.init_wandb()
        else:
            pass

    def init_wandb(self):
        wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=self.train_config.run_name,
            dir=self.train_config.wandb_path,
            config=self.get_wandb_config_dict()
        )

    def get_wandb_config_dict(self):
        return {
            "epochs": self.train_config.epochs,
            "batch_size": self.train_config.batch_size,
            "mode": self.train_config.training_mode,
            "loss": self.train_config.loss,
            "metrics": self.train_config.track_metrics,
            # "classes": self.train_config.class_values,
            "model": self.train_config.model,
            "crop_shape": self.train_config.window_size,
            "unet_depth": self.train_config.unet_depth,
            "base_filters": self.train_config.base_filters,
            "initial_learning_rate": self.train_config.optimizer_cfg["learning_rate"],
            "batch_normalization": self.train_config.batch_normalization,
            "transposed_convolution": self.train_config.transposed_convolution,
            "residual_preactivation": self.train_config.residual_preactivation,
            "dataset_path": str(self.train_config.dataset_path),
            "output_path": str(self.train_config.output_path),
            "descr_path": str(self.train_config.descriptor_path),
            "data_augmentation_transforms": self.train_config.da_transforms,
            "data_augmentation_transforms_cfg": self.train_config.da_transform_cfg,
            "data_augmentation_seed": self.train_config.da_seed,
            "tiling_mode": self.train_config.tiling_mode,
            "window_overlap": self.train_config.window_overlap,
            "notes": self.train_config.notes
        }

    def log_metrics(self, metrics_dict: dict):
        if self.train_config.enable_wandb_tracking:
            if self.train_config.n_output_classes > 1:
                class_values = list(metrics_dict.keys())
                for class_value in class_values:
                    class_dict = metrics_dict[class_value]

                    for key, item in class_dict.items():
                        class_key_str = key + "_" + str(class_value)
                        wandb.run.summary[class_key_str] = item
            else:
                for key, item in metrics_dict.items():
                    wandb.run.summary[key] = item
        else:
            pass
