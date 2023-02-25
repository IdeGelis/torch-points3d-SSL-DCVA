import warnings
warnings.filterwarnings("ignore")
import os
import copy
import torch
import hydra
import time
import logging

from tqdm.auto import tqdm
import wandb

# Import building function for model and dataset
from torch_points3d.datasets.dataset_factory import instantiate_dataset
from torch_points3d.models.model_factory import instantiate_model

# Import BaseModel / BaseDataset for type checking
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.base_dataset import BaseDataset

# Import from metrics
from torch_points3d.metrics.base_tracker import BaseTracker
from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint

# Utils import
from torch_points3d.utils.colors import COLORS
from torch_points3d.utils.wandb_utils import Wandb
from torch_points3d.visualization import Visualizer

# DCVA utils
import skimage.filters as skf
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import Data, Batch
from torch_points3d.metrics.DCVA_3D_tracker import simple_tracker, Urb3DCD_dcva_tracker

log = logging.getLogger(__name__)

CONV_CLASSES = {0: 0, 1: 1, 2: 1, 3: 1, 4: 0, 5: 1, 6: 1}


class DCVA:
    """
    TorchPoints3d Trainer handles the logic between
        - BaseModel,
        - Dataset and its Tracker
        - A custom ModelCheckpoint
        - A custom Visualizer
    It supports MC dropout - multiple voting_runs for val / test datasets
    """

    def __init__(self, cfg):
        self._cfg = cfg
        self._initialize_trainer()

    def _initialize_trainer(self):
        # Enable CUDNN BACKEND
        torch.backends.cudnn.enabled = self.enable_cudnn

        if not self.has_training:
            resume = False
            self._cfg.training = self._cfg
        else:
            resume = bool(self._cfg.training.checkpoint_dir)

        # Get device
        if self._cfg.training.cuda > -1 and torch.cuda.is_available():
            device = "cuda"
            torch.cuda.set_device(self._cfg.training.cuda)
        else:
            device = "cpu"
        self._device = torch.device(device)
        log.info("DEVICE : {}".format(self._device))

        # Profiling
        if self.profiling:
            # Set the num_workers as torch.utils.bottleneck doesn't work well with it
            self._cfg.training.num_workers = 0

        # Start Wandb if public
        if self.wandb_log:
            Wandb.launch(self._cfg, self._cfg.wandb.public and self.wandb_log)

        # Checkpoint
        self._checkpoint: ModelCheckpoint = ModelCheckpoint(
            self._cfg.training.checkpoint_dir,
            self._cfg.model_name,
            self._cfg.training.weight_name,
            run_config=self._cfg,
            resume=resume,
        )

        # Create model and datasets
        if not self._checkpoint.is_empty:
            self._dataset: BaseDataset = instantiate_dataset(self._checkpoint.data_config)
            self._model: BaseModel = self._checkpoint.create_model(
                self._dataset, weight_name=self._cfg.training.weight_name
            )
        else:
            self._dataset: BaseDataset = instantiate_dataset(self._cfg.data)
            self._model: BaseModel = instantiate_model(copy.deepcopy(self._cfg), self._dataset)
            self._model.instantiate_optimizers(self._cfg, "cuda" in device)
            self._model.set_pretrained_weights()
            if not self._checkpoint.validate(self._dataset.used_properties):
                log.warning(
                    "The model will not be able to be used from pretrained weights without the corresponding dataset. Current properties are {}".format(
                        self._dataset.used_properties
                    )
                )
        self._checkpoint.dataset_properties = self._dataset.used_properties

        log.info(self._model)

        self._model.log_optimizers()
        log.info("Model size = %i", sum(param.numel() for param in self._model.parameters() if param.requires_grad))

        # Set dataloaders
        self._dataset.create_dataloaders(
            self._model,
            self._cfg.training.batch_size,
            self._cfg.training.shuffle,
            self._cfg.training.num_workers,
            self.precompute_multi_scale,
        )
        log.info(self._dataset)

        # Verify attributes in dataset
        if self._dataset.train_data is not None:
            self._model.verify_data(self._dataset.train_dataset[0])
        elif self._dataset.val_data is not None:
            self._model.verify_data(self._dataset.val_data)
        elif self._dataset.test_data is not None:
            self._model.verify_data(self._dataset.test_data)
        else:
            print('No dataset so no verification of attributes')

        # Choose selection stage
        selection_stage = getattr(self._cfg, "selection_stage", "")
        self._checkpoint.selection_stage = self._dataset.resolve_saving_stage(selection_stage)
        self._tracker: BaseTracker = self._dataset.get_tracker(self.wandb_log, self.tensorboard_log,
                                                               self.tracker_options.full_pc,
                                                               self.tracker_options.full_res)

        if self.wandb_log:
            Wandb.launch(self._cfg, not self._cfg.wandb.public and self.wandb_log)

        # Run training / evaluation
        self._model = self._model.to(self._device)
        if self.has_visualization:
            self._visualizer = Visualizer(
                self._cfg.visualization, self._dataset.num_batches, self._dataset.batch_size, os.getcwd()
            )

        self.compute_thresholds_bool = self._cfg.compute_thresholds
        if self._cfg.perform_annex_task:
            print('Perform annex task')
            self.get_cloud_classification_annex_task()

    def main_DCVA(self):
        if self.compute_thresholds_bool:
            print('Threshold computation')
            self.compute_thresholds()
        else:
            self.thresholds = [0.46849772]#[4.7891507]#[5.2416525]
        self.labels = [0, 1]
        log.info("Thresholds that will be used to segment PCs:" + str(self.thresholds))
        self.get_DCVA_labels()

    def get_cloud_classification_annex_task(self):
        # self._dataset.test_data.num_classes
        tracker_0 = simple_tracker(self._dataset, nb_class=7, wandb_log=False, use_tensorboard=False, full_res=True,
                                   full_pc=True)
        tracker_1 = simple_tracker(self._dataset, nb_class=7, wandb_log=False, use_tensorboard=False, full_res=True,
                                   full_pc=True)
        tracker_0.reset("test")
        tracker_1.reset("test")
        if self._dataset.has_test_loaders:
            loaders = self._dataset.test_dataloaders
            self._model._model_pre_trained.eval()
            for loader in loaders:
                with Ctq(loader) as tq_loader:
                    for data in tq_loader:
                        with torch.no_grad():
                            data0, data1 = data.to_data()
                            data1.area = data0.area
                            data0.y = None
                            self._model._model_pre_trained.set_input(data0, device=self._device)
                            with torch.cuda.amp.autocast(enabled=self._model.is_mixed_precision()):
                                self._model._model_pre_trained.forward(epoch=0, compute_loss=False)
                            tracker_0.track(self._model._model_pre_trained, data=data, full_pc=True, idx_data=0)
                            self._model._model_pre_trained.set_input(data1, device=self._device)
                            with torch.cuda.amp.autocast(enabled=self._model.is_mixed_precision()):
                                self._model._model_pre_trained.forward(epoch=0, compute_loss=False)
                            tracker_1.track(self._model._model_pre_trained, data=data, full_pc=True, idx_data=1)
            tracker_0.finalise(save_pc=True, idx_data=0, saving_path=os.path.join(os.getcwd(), 'SegSemFromPretrained'))
            tracker_1.finalise(save_pc=True, idx_data=1, saving_path=os.path.join(os.getcwd(), 'SegSemFromPretrained'))

    def compute_thresholds(self):
        self._is_training = False
        self._model.dmg = None
        if self._dataset.has_test_loaders:
            loaders = self._dataset.test_dataloaders
            self._model.eval()
            for loader in loaders:
                with Ctq(loader) as tq_loader:
                    for data in tq_loader:
                        with torch.no_grad():
                            self._model.set_input(data, self._device)
                            with torch.cuda.amp.autocast(enabled=self._model.is_mixed_precision()):
                                self._model.get_dmgs(save_dmg=True)
        dmgs = self._model.dmgs.cpu().numpy()
        print("Thresholding...")
        self.thresholds = skf.threshold_multiotsu(dmgs, self._cfg.nb_thresholds + 1)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 3.5))
        ax.hist(dmgs, 100, [np.nanmin(dmgs), np.nanmax(dmgs)])
        ax.set_title('Threshold' + str(self.thresholds))
        for thresh in self.thresholds:
            ax.axvline(thresh, color='r')
        fig.savefig(os.path.join(os.getcwd(), "DMG_histo.png"))
        plt.close(fig)

    def get_DCVA_labels(self):
        self._is_training = False
        self._tracker.reset(stage="test", nb_class=max(self.labels) + 1)
        if self._dataset.has_test_loaders:
            loaders = self._dataset.test_dataloaders
            self._model.eval()
            for loader in loaders:
                with Ctq(loader) as tq_loader:
                    for data in tq_loader:
                        with torch.no_grad():
                            self._model.set_input(data, self._device)
                            with torch.cuda.amp.autocast(enabled=self._model.is_mixed_precision()):
                                self._model.get_dmgs(save_dmg=False, thresholds=self.thresholds, labels=self.labels)
                            self._tracker.track(self._model, full_pc=True, idx_data=1)
            self._tracker.finalise(save_pc=True, idx_data=1,
                                   saving_path=os.path.join(os.getcwd(), 'ResAftThresholding'),
                                   conv_classes=CONV_CLASSES, filter_pts=0.3,#-1
                                   thresholds=self.thresholds, labels=self.labels)
            self._tracker.plot_save_metric_dc(plot=True, saving_path=os.path.join(os.getcwd(), 'ResAftThresholding'))

    @property
    def early_break(self):
        return getattr(self._cfg.debugging, "early_break", False) and self._is_training

    @property
    def profiling(self):
        return getattr(self._cfg.debugging, "profiling", False)

    @property
    def num_batches(self):
        return getattr(self._cfg.debugging, "num_batches", 50)

    @property
    def enable_cudnn(self):
        return getattr(self._cfg.training, "enable_cudnn", True)

    @property
    def enable_dropout(self):
        return getattr(self._cfg, "enable_dropout", True)

    @property
    def has_visualization(self):
        return getattr(self._cfg, "visualization", False)

    @property
    def has_tensorboard(self):
        return getattr(self._cfg, "tensorboard", False)

    @property
    def has_training(self):
        return getattr(self._cfg, "training", None)

    @property
    def precompute_multi_scale(self):
        return self._model.conv_type == "PARTIAL_DENSE" and getattr(self._cfg.training, "precompute_multi_scale", False)

    @property
    def wandb_log(self):
        if getattr(self._cfg, "wandb", False):
            return getattr(self._cfg.wandb, "log", False)
        else:
            return False

    @property
    def tensorboard_log(self):
        if self.has_tensorboard:
            return getattr(self._cfg.tensorboard, "log", False)
        else:
            return False

    @property
    def tracker_options(self):
        return self._cfg.get("tracker_options", {})

    @property
    def eval_frequency(self):
        return self._cfg.get("eval_frequency", 1)
