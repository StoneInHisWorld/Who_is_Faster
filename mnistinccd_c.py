import os
from typing import Iterable, Any, Sized, List

import numpy as np
import pandas as pd
import skimage.transform
import torch
from PIL.Image import Image
from tqdm import tqdm

import data_related.data_related as dr
import utils.func.img_tools as itools
import utils.func.pytools as pytools
import utils.func.tensor_tools as ttools
from data_related.SelfDefinedDataset import SelfDefinedDataSet
from data_related.data_related import normalize
from utils.func.img_tools import read_img

# 16 holes
aperture_size = 16
border_size = 8
gap_size = 1 * aperture_size
n_row = 4
n_col = 4
xv = [border_size + i * (aperture_size + gap_size) for i in range(n_row)]
yv = [border_size + i * (aperture_size + gap_size) for i in range(n_col)]
yv, xv = np.meshgrid(xv, yv)
hole_pos = [(x, y) for x, y in zip(xv.reshape([-1]), yv.reshape([-1]))]
hole_size = [aperture_size for _ in range(len(hole_pos))]
"""previous version"""


class MNISTinCCD_C(SelfDefinedDataSet):

    def __init__(self, **kwargs):
        """
        MNIST-CCD采集图像源数据集类，负责进行源数据集加载。
        本数据集通常数据量较大，通常为10GB数量级，因此默认采用懒加载模式。
        本数据集的标签集为每张MNIST图片对应数字分类。
        """
        # 判断图片指定形状
        super().__init__(**kwargs)

    def _check_path(self, root: str, which: str) -> None:
        """
        检查数据集路径是否正确，否则直接中断程序。
        本数据集要求目录结构为（请在代码中查看）：
        └─root((TRAIN))
           ├─ TRAIN
           │   ├─ EXP_DATA
           │   │   └─which
           │   └─train_labels.csv
           └─ TEST
              ├─ EXP_DATA
              │   └─which
              └─test_labels.csv
        :param root: 数据集源目录。
        :param which: 数据集批次名
        :return: None
        """
        assert os.path.exists(root), f'路径{root}不存在！'
        path_iter = os.walk(root)
        _, folders, __ = next(path_iter)

        def __check_data_dir(root, labels_file_name):
            path_iter = os.walk(root)
            _, folders, files = next(path_iter)
            assert 'EXP_DATA' in folders, f'路径\"{root}\"目录下无\"\\EXP_DATA\\\"文件夹！'
            feature_root = os.path.join(root, 'EXP_DATA')
            path_iter = os.walk(feature_root)
            for _, exp_folders, _ in path_iter:
                assert which in exp_folders, f'路径\"{feature_root}\"目录下尚未找到需要读取的\"\\{which}\\\"文件夹！'
                fd_cand = os.path.join(feature_root, which)
                break
            assert labels_file_name in files, f'路径\"{root}\"目录下尚未找到需要读取的{labels_file_name}文件！'
            ld_cand = os.path.join(root, labels_file_name)
            return fd_cand, ld_cand

        # 检查训练数据目录
        assert 'TRAIN' in folders, f'路径\"{root}\"目录下无\"\\TRAIN\\\"训练数据文件夹！'
        self._train_fd, self._train_ld = __check_data_dir(os.path.join(root, 'TRAIN'), 'train_labels.csv')
        assert 'TEST' in folders, f'路径\"{root}\"目录下无\"\\TEST\\\"测试数据文件夹！'
        self._test_fd, self._test_ld = __check_data_dir(os.path.join(root, 'TEST'), 'test_labels.csv')

    @staticmethod
    def _get_fea_index(features, root) -> None:
        """
        读取根目录下的特征集索引
        :return: None
        """
        path_iter = os.walk(root)
        _, __, file_names = next(path_iter)
        file_names = sorted(
            file_names, key=lambda name: int(name.split(".")[0])
        )  # 给文件名排序！
        with tqdm(file_names, desc='整理特征集索引...', unit='张', position=0,
                  leave=True) as pbar:
            for fn in pbar:
                features.append(os.path.join(root, fn))
            pbar.set_description('整理完毕，正在进行存储...')

    @staticmethod
    def _get_lb_index(labels, root) -> None:
        """
        读取根目录下的标签集索引。
        因为标签集本身比索引还要简单，因此直接使用此函数读取标签集。
        :return: None
        """
        file = pd.read_csv(root)
        labels += file.values[:, 1].tolist()

    @staticmethod
    def read_fea_fn(index: Iterable or Sized, n_worker: int = 1) -> Iterable:
        """
        加载特征集数据批所用方法
        :param n_worker: 使用的处理机数目，若>1，则开启多线程处理
        :param index: 加载特征集数据批所用索引
        :return: 读取到的特征集数据批
        """
        data_slice = []
        preprocess = [
            (itools.crop_img, (), {'required_shape': (950, 950), 'loc': 'c'}),
            (itools.resize_img, (), {'required_shape': MNISTinCCD_C.f_required_shape}),
            (Image.rotate, (180,), {}),
            # (itools.binarize_img, (127, ), {})
        ]
        if int(n_worker) > 1:
            with tqdm(total=len(index), unit='张', position=0, desc=f"读取特征集图片中……", mininterval=1,
                      leave=True) as pbar:
                def task(indexes):
                    ret = []
                    for i in indexes:
                        ret.append(
                            read_img(i, MNISTinCCD_C.fea_mode, False, *preprocess)
                        )
                        pbar.update(1)
                    return ret

                data_slice = pytools.iterable_multi_process(index, task, True, n_worker)
        else:
            with tqdm(index, unit='张', position=0, desc=f"读取特征集图片中……", mininterval=1, leave=True) as pbar:
                for index in pbar:
                    data_slice.append(
                        read_img(index, MNISTinCCD_C.fea_mode, False, *preprocess)
                    )
        return data_slice

    @staticmethod
    def read_lb_fn(index: Iterable, n_worker: int = 1) -> Iterable:
        """
        加载标签集数据批所用方法。因为本数据集并不存储标签集索引，因此本函数没有实际作用，返回的为index值
        :param n_worker: 使用的处理机数目，若>1，则开启多线程处理
        :param index: 加载标签集数据批所用索引
        :return: 读取到的标签集数据批
        """
        return index

    @staticmethod
    def get_criterion_a():
        return dr.single_argmax_accuracy
        # return data_related.data_related.single_argmax_accuracy(Y_HAT, Y, size_average)

    @staticmethod
    def unwrap_fn(inputs, predictions, labels, acc_s, loss_es, comments) -> Any:
        print('正在拼装结果……')
        inp_s = ttools.tensor_to_img(inputs, MNISTinCCD_C.fea_mode)
        pre_s = torch.argmax(predictions, dim=1)
        lb_s = torch.argmax(labels, dim=1)
        # 制作输入、输出、标签对照图
        ret = itools.concat_imgs(
            *[
                [(inp, 'input')]
                for inp in inp_s
            ],
            comments=[
                f'pred = {pre.item()}\n'
                f'label = {lb.item()}\n'
                f'loss = {loss.item(): .3f}'
                for pre, lb, loss in zip(pre_s, lb_s, loss_es)
            ],
            text_size=15, border_size=5,
            required_shape=(750, 1500)
        )
        return ret

    @staticmethod
    def save_fn(result: Iterable[Image], root: str) -> None:
        pytools.check_path(root)
        print('正在保存结果……')
        for i, res in enumerate(result):
            res.save(os.path.join(root, f'{i}.jpg'))

    def default_preprocesses(self):
        self.feaIndex_preprocesses = [
            lambda d: np.array(d),
        ]
        self.lbIndex_preprocesses = [
            lambda d: np.array(d),
        ]
        self.fea_preprocesses = [
            lambda fea: itools.add_mask(
                fea,
                itools.get_mask(
                    hole_pos, hole_size,
                    MNISTinCCD_C.fea_channel, MNISTinCCD_C.f_required_shape
                )
            ),
            lambda fea: np.array(fea),
            lambda d: torch.from_numpy(d),
            lambda d: d.type(torch.float32),
            lambda d: normalize(d),
        ]
        self.lb_preprocesses = [
            lambda fea: np.array(fea),
            lambda d: torch.from_numpy(d),
            lambda d: d.type(torch.float32),
            lambda d: normalize(d),
        ]

    def AdaWZYNet_preprocesses(self):
        self.feaIndex_preprocesses = [
            lambda d: np.array(d),
        ]
        self.lbIndex_preprocesses = [
            lambda d: np.array(d),
        ]
        self.fea_preprocesses = [
            lambda fea: np.array(fea),
            lambda fea: itools.mean_LI_of_holes(fea, hole_pos, hole_size),
            lambda fea: itools.extract_and_cat_holes(
                fea, hole_pos, hole_size, n_row, n_col
            ),
            lambda d: torch.from_numpy(d),
            lambda d: d.type(torch.float32),
            lambda d: normalize(d),
        ]
        self.lb_preprocesses = [
            lambda d: pd.get_dummies(d),
            lambda fea: np.array(fea),
            lambda d: torch.from_numpy(d),
            lambda d: d.type(torch.float32),
        ]

    def GoogLeNet_preprocesses(self):
        self.feaIndex_preprocesses = [
            lambda d: np.array(d),
        ]
        self.lbIndex_preprocesses = [
            lambda d: np.array(d),
        ]
        self.fea_preprocesses = [
            lambda fea: np.array(fea),
            lambda fea: itools.mean_LI_of_holes(fea, hole_pos, hole_size),
            lambda fea: itools.extract_and_cat_holes(
                fea, hole_pos, hole_size, n_row, n_col
            ),
            lambda d: skimage.transform.resize(
                d, [len(d), self.fea_channel, *self.f_required_shape]
            ),
            lambda d: torch.from_numpy(d),
            lambda d: d.type(torch.float32),
            lambda d: normalize(d),
        ]
        self.lb_preprocesses = [
            lambda d: pd.get_dummies(d),
            lambda fea: np.array(fea),
            lambda d: torch.from_numpy(d),
            lambda d: d.type(torch.float32),
        ]

    @property
    def train_lb_set(self):
        return set(torch.unique(super()._train_l))

    @property
    def test_lb_set(self):
        return set(torch.unique(super()._test_l))
