import os.path as osp
import numpy as np
import torch
import random
import os
from plyfile import PlyData, PlyElement
from torch_geometric.data import Data, extract_zip, Dataset

from torch_points3d.core.data_transform import GridSampling3D, CylinderSampling, SphereSampling
from torch_points3d.datasets.change_detection.AHNPairCylinder import AHNCylinder
# from torch_points3d.datasets.change_detection.AHNPairCylinder_tranfeats import AHNCylinder
from torch_points3d.datasets.change_detection.base_siamese_dataset import BaseSiameseDataset
from torch_points3d.metrics.DCVA_3D_tracker import Urb3DCD_dcva_tracker
from torch_points3d.datasets.change_detection.pair import Pair, MultiScalePair
from torch_points3d.datasets.change_detection.Urb3DSimulPairCylinder import to_ply

IGNORE_LABEL = -1
# INV_OBJECT_LABEL = {i:"class " + str(i) for i in range(URB3DSIMUL_NUM_CLASSES)}
H3D_NUM_CLASSES = 7
OBJECT_COLOR = np.asarray(
    [
        [67, 1, 84],  # 'unchanged'
        [0, 183, 255],  # 'newlyBuilt'
        [0, 12, 235],  # 'deconstructed'
        [0, 217, 33],  # 'newVegetation'
        [255, 230, 0],  # 'vegetationGrowUp'
        [255, 140, 0],  # 'vegetationRemoved'
        [255, 0, 0],  # 'mobileObjects'
    ]
)
def read_from_ply(filename, nameInPly = "params", sf = "label_ch"):
    """read XYZ for each vertex."""
    assert os.path.isfile(filename)
    with open(filename, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata[nameInPly].count
        vertices = np.zeros(shape=[num_verts, 4], dtype=np.float32)
        vertices[:, 0] = plydata[nameInPly].data["x"]
        vertices[:, 1] = plydata[nameInPly].data["y"]
        vertices[:, 2] = plydata[nameInPly].data["z"]
        vertices[:, 3] = plydata[nameInPly].data[sf]
    return vertices

class AHNCylinder_DCVA(AHNCylinder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes_ch = self.num_classes
        self.num_classes = H3D_NUM_CLASSES


class AHNDataset_DCVA(BaseSiameseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        self.radius = float(self.dataset_opt.radius)
        self.sample_per_epoch = int(self.dataset_opt.sample_per_epoch)
        self.DA = self.dataset_opt.DA
        self.TTA = False
        self.preprocessed_dir = self.dataset_opt.preprocessed_dir

        # self.train_dataset = Urb3DCDPairCylinder_DCVA(
        #     filePaths=self.dataset_opt.dataTrainFile,
        #     split="train",
        #     radius=self.radius,
        #     sample_per_epoch=self.sample_per_epoch,
        #     DA=self.DA,
        #     pre_transform=self.pre_transform,
        #     preprocessed_dir=osp.join(self.preprocessed_dir, "Train"),
        #     reload_preproc=self.dataset_opt.load_preprocessed,
        #     reload_trees=self.dataset_opt.load_trees,
        #     nameInPly=self.dataset_opt.nameInPly,
        # )
        # self.train_dataset_kmeans = Urb3DCDPairCylinder_DCVA(
        #     filePaths=self.dataset_opt.dataTrainFile,
        #     split="train",
        #     radius=self.radius,
        #     sample_per_epoch=-1,
        #     DA=self.DA,
        #     pre_transform=self.pre_transform,
        #     preprocessed_dir=osp.join(self.preprocessed_dir, "Train"),
        #     reload_preproc=self.dataset_opt.load_preprocessed,
        #     reload_trees=self.dataset_opt.load_trees,
        #     nameInPly=self.dataset_opt.nameInPly,
        # )
        # DCVA only applied on the test set, so it is not required to load other sets
        self.test_dataset = AHNCylinder_DCVA(
            filePaths=self.dataset_opt.dataTestFile,
            split="test",
            radius=self.radius,
            sample_per_epoch=-1,
            pre_transform=self.pre_transform,
            transform=self.test_transform,
            preprocessed_dir=osp.join(self.preprocessed_dir, "Test"),
            reload_preproc=self.dataset_opt.load_preprocessed,
            reload_trees=self.dataset_opt.load_trees,
            # nameInPly=self.dataset_opt.nameInPly,
            # comp_norm = False,
        )
        self.num_classes_orig = self.test_data.num_classes_ch


    @property
    def train_data(self):
        if type(self.train_dataset) == list:
            return self.train_dataset[0]
        else:
            return self.train_dataset

    @property
    def val_data(self):
        if type(self.val_dataset) == list:
            return self.val_dataset[0]
        else:
            return self.val_dataset

    @property
    def test_data(self):
        if type(self.test_dataset) == list:
            return self.test_dataset[0]
        else:
            return self.test_dataset


    @staticmethod
    def to_ply(pos, label, file, color=OBJECT_COLOR, sf = None):
        """ Allows to save Urb3DCD predictions to disk using Urb3DCD color scheme
            Parameters
            ----------
            pos : torch.Tensor
                tensor that contains the positions of the points
            label : torch.Tensor
                predicted label
            file : string
                Save location
            """
        to_ply(pos, label, file, color=color, sf = sf)

    def set_nbclass(self, nb_class):
        self.train_data.num_classes = nb_class
        self.test_data.num_classes = nb_class

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool, full_pc=False, full_res=False):
        """Factory method for the tracker
            Arguments:
                wandb_log - Log using weight and biases
                tensorboard_log - Log using tensorboard
            Returns:
                [BaseTracker] -- tracker
            """
        return Urb3DCD_dcva_tracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log,
                                           full_pc=full_pc, full_res=full_res)
