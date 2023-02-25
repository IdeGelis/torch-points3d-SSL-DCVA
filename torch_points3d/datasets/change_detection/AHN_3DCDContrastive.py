import os.path as osp
import numpy as np
import torch
import random
import os
from plyfile import PlyData, PlyElement
from torch_geometric.data import Data, extract_zip, Dataset

from torch_points3d.core.data_transform import GridSampling3D, CylinderSampling, SphereSampling
from torch_points3d.datasets.change_detection.Urb3DSimulPairCylinder import to_ply
from torch_points3d.datasets.change_detection.AHNPairCylinder import AHNCylinder
from torch_points3d.datasets.change_detection.base_siamese_dataset import BaseSiameseDataset
from torch_points3d.metrics.contrastiveselfsup_tracker import selfsup_tracker
from torch_points3d.datasets.change_detection.pair import Pair, MultiScalePair
from torch_points3d.datasets.change_detection.Urb3DSimulPairCylinder import cloud_loader


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

class AHNCylinder_Contrastive(AHNCylinder):
    def __init__(self, num_cluster=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_cluster = num_cluster
        self._get_shuffle = False
        if self._sample_per_epoch <= 0:
            self.shuffle_idx = np.random.shuffle(np.arange(self.__len__()))

    def get(self, idx):
        if self._sample_per_epoch > 0:
            return self._get_random()
        else:
            pair_correct = False
            while not pair_correct and idx<self.grid_regular_centers.shape[0]:
                centre = self.grid_regular_centers[idx, :3]
                area_sel = self.grid_regular_centers[idx, 3].int()
                pair = self._load_save(area_sel)
                cylinder_sampler = CylinderSampling(self._radius, centre, align_origin=False)
                dataPC0 = Data(pos=pair.pos, idx=torch.arange(pair.pos.shape[0]).reshape(-1))
                setattr(dataPC0, CylinderSampling.KDTREE_KEY, pair.KDTREE_KEY_PC0)
                dataPC1 = Data(pos=pair.pos_target, y=pair.y, idx=torch.arange(pair.pos_target.shape[0]).reshape(-1))
                setattr(dataPC1, CylinderSampling.KDTREE_KEY, pair.KDTREE_KEY_PC1)
                dataPC0_cyl = cylinder_sampler(dataPC0)
                dataPC1_cyl = cylinder_sampler(dataPC1)
                try:
                    if self.manual_transform is not None:
                        dataPC0_cyl = self.manual_transform(dataPC0_cyl)
                        dataPC1_cyl = self.manual_transform(dataPC1_cyl)
                    pair_cylinders = Pair(pos=dataPC0_cyl.pos, pos_target=dataPC1_cyl.pos, y=dataPC1_cyl.y,
                                          idx=dataPC0_cyl.idx, idx_target=dataPC1_cyl.idx, area=area_sel)
                    pair_cylinders.normalise()
                    pair_correct = True
                    centre_shuffled = self.grid_regular_centers[self.shuffle_idx[idx], :3]
                    area_sel_shuffled = self.grid_regular_centers[self.shuffle_idx[idx], 3].int()
                    pair = self._load_save(area_sel_shuffled)
                    cylinder_sampler = CylinderSampling(self._radius, centre_shuffled, align_origin=False)
                    dataPC0_s = Data(pos=pair.pos, idx=torch.arange(pair.pos.shape[0]).reshape(-1))
                    setattr(dataPC0, CylinderSampling.KDTREE_KEY, pair.KDTREE_KEY_PC0)
                    dataPC1_s = Data(pos=pair.pos_target, y=pair.y, idx=torch.arange(pair.pos_target.shape[0]).reshape(-1))
                    setattr(dataPC1, CylinderSampling.KDTREE_KEY, pair.KDTREE_KEY_PC1)
                    dataPC0_cyl_s = cylinder_sampler(dataPC0_s)
                    dataPC1_cyl_s = cylinder_sampler(dataPC1_s)
                    if self.manual_transform is not None:
                        dataPC0_cyl_s = self.manual_transform(dataPC0_cyl_s)
                        dataPC1_cyl_s = self.manual_transform(dataPC1_cyl_s)
                    pair_cylinders_s = Pair(pos=dataPC0_cyl_s.pos, pos_target=dataPC1_cyl_s.pos, y=dataPC1_cyl_s.y,
                                          idx=dataPC0_cyl_s.idx, idx_target=dataPC1_cyl_s.idx, area=area_sel_shuffled)
                    pair_cylinders_s.normalise()
                except:
                    print('pair not correct')
                    idx += 1
            if self._get_shuffle:
                return (pair_cylinders, pair_cylinders_s)
            else:
                return pair_cylinders

    def _get_random(self):
        pair_correct = False
        while not pair_correct:
            centre_idx = int(random.random() * (self._centres_for_sampling.shape[0] - 1))
            centre_idx_s = int(random.random() * (self._centres_for_sampling.shape[0] - 1))
            centre = self._centres_for_sampling[centre_idx]
            #  choice of the corresponding PC if several PCs are loaded
            area_sel = centre[3].int()
            pair = self._load_save(area_sel)
            cylinder_sampler = CylinderSampling(self._radius, centre[:3], align_origin=False)
            dataPC0 = Data(pos=pair.pos)
            setattr(dataPC0, CylinderSampling.KDTREE_KEY, pair.KDTREE_KEY_PC0)
            dataPC1 = Data(pos=pair.pos_target, y=pair.y)
            setattr(dataPC1, CylinderSampling.KDTREE_KEY, pair.KDTREE_KEY_PC1)
            dataPC0_cyl = cylinder_sampler(dataPC0)
            dataPC1_cyl = cylinder_sampler(dataPC1)
            try:
                if self.manual_transform is not None:
                    dataPC0_cyl = self.manual_transform(dataPC0_cyl)
                    dataPC1_cyl = self.manual_transform(dataPC1_cyl)
                pair_cyl = Pair(pos=dataPC0_cyl.pos, pos_target=dataPC1_cyl.pos, y=dataPC1_cyl.y, area=area_sel)
                if self.DA:
                    pair_cyl.data_augment()
                pair_cyl.normalise()
                pair_correct = True
            except:
                pair_correct = False

        pair_correct = False
        while not pair_correct:
            centre = self._centres_for_sampling[centre_idx_s]
            #  choice of the corresponding PC if several PCs are loaded
            area_sel = centre[3].int()
            pair = self._load_save(area_sel)
            cylinder_sampler = CylinderSampling(self._radius, centre[:3], align_origin=False)
            dataPC0 = Data(pos=pair.pos)
            setattr(dataPC0, CylinderSampling.KDTREE_KEY, pair.KDTREE_KEY_PC0)
            dataPC1 = Data(pos=pair.pos_target, y=pair.y)
            setattr(dataPC1, CylinderSampling.KDTREE_KEY, pair.KDTREE_KEY_PC1)
            dataPC0_cyl = cylinder_sampler(dataPC0)
            dataPC1_cyl = cylinder_sampler(dataPC1)
            try:
                if self.manual_transform is not None:
                    dataPC0_cyl = self.manual_transform(dataPC0_cyl)
                    dataPC1_cyl = self.manual_transform(dataPC1_cyl)
                pair_cyl_s = Pair(pos=dataPC0_cyl.pos, pos_target=dataPC1_cyl.pos, y=dataPC1_cyl.y, area=area_sel)
                if self.DA:
                    pair_cyl_s.data_augment()
                pair_cyl_s.normalise()
                pair_correct=True
            except:
                pair_correct=False

        if self._get_shuffle:
            return (pair_cyl, pair_cyl_s)
        else:
            return pair_cyl


class AHNDataset_Contrastive(BaseSiameseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        self.radius = float(self.dataset_opt.radius)
        self.sample_per_epoch = int(self.dataset_opt.sample_per_epoch)
        self.DA = self.dataset_opt.DA
        self.TTA = False
        self.preprocessed_dir = self.dataset_opt.preprocessed_dir
        self.num_pseudo_label = self.dataset_opt.num_pseudo_label
        self.train_dataset = AHNCylinder_Contrastive(
            filePaths=self.dataset_opt.dataTrainFile,
            split="train",
            radius=self.radius,
            sample_per_epoch=self.sample_per_epoch,
            DA=self.DA,
            num_cluster=self.dataset_opt.num_pseudo_label,
            pre_transform=self.pre_transform,
            preprocessed_dir=osp.join(self.preprocessed_dir, "Train"),
            reload_preproc=self.dataset_opt.load_preprocessed,
            reload_trees=self.dataset_opt.load_trees,
        )

        self.test_dataset = AHNCylinder_Contrastive(
            filePaths=self.dataset_opt.dataTestFile,
            split="test",
            radius=self.radius,
            num_cluster=self.dataset_opt.num_pseudo_label,
            sample_per_epoch=-1,
            pre_transform=self.pre_transform,
            transform=self.test_transform,
            preprocessed_dir=osp.join(self.preprocessed_dir, "Test"),
            reload_preproc=self.dataset_opt.load_preprocessed,
            reload_trees=self.dataset_opt.load_trees,
        )

    def create_dataloaders(
            self,
            model,
            batch_size: int,
            shuffle: bool,
            num_workers: int,
            precompute_multi_scale: bool,):
        """ Creates the data loaders. Must be called in order to complete the setup of the Dataset
        """
        super().create_dataloaders(model, batch_size, shuffle, num_workers, precompute_multi_scale)

        if self.train_dataset:
            self._train_loader_shuf = self._dataloader(
                self.train_dataset,
                self.train_pre_batch_collate_transform,
                model.conv_type,
                precompute_multi_scale,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                sampler=self.train_sampler,
            )

        if self.test_dataset:
            self._test_loaders_shuf = [
                self._dataloader(
                    dataset,
                    self.test_pre_batch_collate_transform,
                    model.conv_type,
                    precompute_multi_scale,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    sampler=self.test_sampler,
                )
                for dataset in self.test_dataset
            ]

        if self.val_dataset:
            self._val_loader_shuf = self._dataloader(
                self.val_dataset,
                self.val_pre_batch_collate_transform,
                model.conv_type,
                precompute_multi_scale,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                sampler=self.val_sampler,
            )

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

    @property
    def has_train_loader_shuf(self):
        return hasattr(self, "_train_loader_shuf")

    @property
    def has_val_loader_shuf(self):
        return hasattr(self, "_val_loader_shuf")

    @property
    def has_test_loaders_shuf(self):
        return hasattr(self, "_test_loaders_shuf")

    @property
    def train_dataloader_shuf(self):
        return self._train_loader_shuf

    @property
    def val_dataloader_shuf(self):
        return self._val_loader_shuf

    @property
    def test_dataloaders_shuf(self):
        if self.has_test_loaders_shuf:
            return self._test_loaders_shuf
        else:
            return []


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
        return selfsup_tracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
