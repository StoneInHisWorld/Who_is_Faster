import os
from typing import Tuple, Iterable, Any

import numpy as np
import pandas as pd
import torch
from PIL.Image import Image
from tqdm import tqdm

import utils.func.img_tools as itools
import utils.func.pytools as pytools
import utils.func.tensor_tools as ttools
from data_related.data_related import data_slicer, normalize
from data_related.datasets import DataSet, LazyDataSet
from networks.layers.ssim import SSIM
from networks.nets.adap2p import AdaP2P
from utils.func.img_tools import read_img
from utils.thread import Thread

# 16 holes
aperture_size = 16
border_size = 8
gap_size = 1 * aperture_size
n_row = 8
n_col = 8
# xv = [border_size, border_size + aperture_size + gap_size,
#       border_size + 2 * (aperture_size + gap_size),
#       border_size + 3 * (aperture_size + gap_size)]
# yv = [border_size, border_size + aperture_size + gap_size,
#       border_size + 2 * (aperture_size + gap_size),
#       border_size + 3 * (aperture_size + gap_size)]
xv = [border_size + i * (aperture_size + gap_size) for i in range(n_row)]
yv = [border_size + i * (aperture_size + gap_size) for i in range(n_col)]
yv, xv = np.meshgrid(xv, yv)
hole_pos = [(x, y) for x, y in zip(xv.reshape([-1]), yv.reshape([-1]))]
hole_size = [aperture_size for _ in range(len(hole_pos))]
"""previous version"""


# hole_pos = [(25, 25), (25, 75), (25, 125), (25, 175),
#             (75, 25), (75, 75), (75, 125), (75, 175),
#             (125, 25), (125, 75), (125, 125), (125, 175),
#             (175, 25), (175, 75), (175, 125), (175, 175)]
# hole_size = [25 for _ in range(len(hole_pos))]


# # 9 holes
# hole_pos = [(50, 50), (50, 112), (50, 175),
#             (112, 50), (112, 112), (112, 175),
#             (175, 50), (175, 112), (175, 175)]
# hole_size = [25 for _ in range(len(hole_pos))]

# # 4 holes
# hole_pos = [(100, 100), (150, 150),
#             (100, 150), (150, 100)]
# hole_size = [25 for _ in range(len(hole_pos))]


class MNISTinCCD_C:
    fea_channel = 1
    lb_channel = 1
    fea_mode = 'L'  # 读取二值图，读取图片速度将会大幅下降
    lb_mode = '1'
    f_required_shape = (256, 256)
    l_required_shape = (256, 256)

    def __init__(self,
                 where: str, which: str, module: type,
                 data_portion=1., shuffle=True,
                 f_lazy: bool = True, l_lazy: bool = False, lazy: bool = True,
                 f_req_sha: Tuple[int, int] = (256, 256),
                 l_req_sha: Tuple[int, int] = (256, 256),
                 required_shape: Tuple[int, int] = None):
        """
        MNIST-CCD采集图像源数据集类，负责进行源数据集加载。
        本数据集通常数据量较大，通常为10GB数量级，因此默认采用懒加载模式。
        本数据集的标签集为每张MNIST图片对应数字分类。
        :param where: 数据集所处路径
        :param which: 实验所用数据集的文件名，用于区分同一实验，不同采样批次。
        :param module: 实验涉及数据集类型。数据集会根据实验所用模型来自动指定数据预处理程序。
        :param data_portion: 选取的数据集比例
        :param shuffle: 是否打乱数据
        :param f_lazy: 特征集懒加载参数。指定后，特征集将变为懒加载模式。
        :param l_lazy: 标签集懒加载参数。指定后，标签集将变为懒加载模式。
        :param lazy: 懒加载参数。指定后，数据集会进行懒加载，即每次通过索引取数据时，才从存储中取出数据。lazy的优先级比f/l_lazy高。
        :param f_req_sha: 需要的特征集图片形状。指定后，会将读取到的特征集图片放缩成该形状。
        :param l_req_sha: 需要的标签集图片形状。指定后，会将读取到的标签集图片放缩成该形状。
        :param required_shape: 需要的图片形状。指定后，会将读取到的图片放缩成该形状。required_shape优先级比f/l_req_sha低
        """
        # 判断图片指定形状
        MNISTinCCD_C.f_required_shape = required_shape
        MNISTinCCD_C.l_required_shape = required_shape
        MNISTinCCD_C.f_required_shape = f_req_sha
        MNISTinCCD_C.l_required_shape = l_req_sha
        # 判断数据集懒加载程度
        self.__f_lazy = f_lazy
        self.__l_lazy = l_lazy
        self.__f_lazy = lazy
        self.__l_lazy = lazy
        # 进行训练数据路经检查
        self.__train_fd = None
        self.__train_ld = None
        self.__test_fd = None
        self.__test_ld = None
        self.__check_path(where, which)
        # 获取特征集、标签集及其索引集的预处理程序
        self.__set_preprocess(module)
        # 进行训练索引获取
        self.__train_f, self.__train_l = [], []
        self.__get_fea_index(self.__train_f, self.__train_fd)
        self.__get_lb_index(self.__train_l, self.__train_ld)
        # 按照数据比例切分数据集索引
        self.__train_f, self.__train_l = data_slicer(data_portion, shuffle, self.__train_f, self.__train_l)
        # 按照懒加载程度加载数据集
        self.__train_f = MNISTinCCD_C.read_fea_fn(self.__train_f, 16) \
            if not self.__f_lazy else self.__train_f
        self.__train_l = MNISTinCCD_C.read_lb_fn(self.__train_l, 16) \
            if not self.__l_lazy else self.__train_l
        # 进行测试索引获取
        self.__test_f, self.__test_l = [], []
        self.__get_fea_index(self.__test_f, self.__test_fd)
        self.__get_lb_index(self.__test_l, self.__test_ld)
        self.__test_f, self.__test_l = data_slicer(data_portion, shuffle, self.__test_f, self.__test_l)
        self.__test_f = MNISTinCCD_C.read_fea_fn(self.__test_f, 16) \
            if not self.__f_lazy else self.__test_f
        self.__test_l = MNISTinCCD_C.read_lb_fn(self.__test_l, 16) \
            if not self.__l_lazy else self.__test_l
        assert len(self.__train_f) == len(self.__train_l), '特征集和标签集长度须一致'
        assert len(self.__test_f) == len(self.__test_l), '特征集和标签集长度须一致'
        del self.__train_fd, self.__train_ld, self.__test_fd, self.__test_ld

    def __check_path(self, root: str, which: str) -> None:
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
            ld_cand = os.path.join(root, 'MNIST')
            return fd_cand, ld_cand

        # 检查训练数据目录
        assert 'TRAIN' in folders, f'路径\"{root}\"目录下无\"\\TRAIN\\\"训练数据文件夹！'
        self.__train_fd, self.__train_ld = __check_data_dir(os.path.join(root, 'TRAIN'), 'train_labels.csv')
        assert 'TEST' in folders, f'路径\"{root}\"目录下无\"\\TEST\\\"测试数据文件夹！'
        self.__test_fd, self.__test_ld = __check_data_dir(os.path.join(root, 'TEST'), 'test_labels.csv')

    def read_fn(self,
                fea_index_or_d: Iterable,
                lb_index_or_d: Iterable) -> Tuple[Iterable, Iterable]:
        """
        懒加载读取数据批所用方法。
        签名必须为：
        read_fn(fea_index: Iterable[path], lb_index: Iterable[path]) -> Tuple[features: Iterable, labels: Iterable]
        :param lb_index_or_d: 懒加载读取标签数据批所用索引
        :param fea_index_or_d: 懒加载读取特征数据批所用索引
        :return: 特征数据批，标签数据批
        """
        read_fea_thread = Thread(MNISTinCCD_C.read_fea_fn, fea_index_or_d)
        read_lb_thread = Thread(MNISTinCCD_C.read_lb_fn, lb_index_or_d)
        if self.__f_lazy:
            read_fea_thread.start()
        if self.__l_lazy:
            read_lb_thread.start()
        if read_fea_thread.is_alive():
            read_fea_thread.join()
        if read_lb_thread.is_alive():
            read_lb_thread.join()
        if self.__f_lazy:
            fea_index_or_d = read_fea_thread.get_result()
        if self.__l_lazy:
            lb_index_or_d = read_lb_thread.get_result()
        return fea_index_or_d, lb_index_or_d

    @staticmethod
    def __get_fea_index(features, root) -> None:
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
    def __get_lb_index(labels, root) -> None:
        """
        读取根目录下的标签集索引
        :return: None
        """
        labels = pd.read_csv(root)
        labels = labels.keys()
        # path_iter = os.walk(root)
        # _, __, file_names = next(path_iter)
        # file_names = sorted(
        #     file_names, key=lambda name: int(name.split(".")[0])
        # )  # 给文件名排序！
        # with tqdm(file_names, desc='整理标签集图片索引...', unit='张', position=0,
        #           leave=True) as pbar:
        #     for fn in pbar:
        #         labels.append(os.path.join(root, fn))
        #     pbar.set_description('整理完毕，正在进行存储...')

    @staticmethod
    def read_fea_fn(index: Iterable, n_worker: int = 1) -> Iterable:
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
            def task(indexes):
                ret = []
                for i in indexes:
                    ret.append(
                        read_img(i, MNISTinCCD_C.fea_mode, False, *preprocess)
                    )
                return ret

            data_slice = pytools.iterable_multi_process(index, task, False, n_worker, '读取特征集图片中……')
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
        加载标签集数据批所用方法
        :param n_worker: 使用的处理机数目，若>1，则开启多线程处理
        :param index: 加载标签集数据批所用索引
        :return: 读取到的标签集数据批
        """
        data_slice = []
        preprocess = [
            (itools.resize_img, (), {'required_shape': MNISTinCCD_C.l_required_shape}),
            (itools.binarize_img, (127,), {})
        ]
        if int(n_worker) > 1:
            def task(indexes):
                ret = []
                for i in indexes:
                    ret.append(
                        read_img(i, MNISTinCCD_C.lb_mode, False, *preprocess)
                    )
                return ret

            data_slice = pytools.iterable_multi_process(index, task, False, n_worker, '读取标签集集图片中……')
        else:
            with tqdm(index, unit='张', position=0, desc=f"读取标签集图片中...", mininterval=1, leave=True) as pbar:
                for index in pbar:
                    data_slice.append(
                        read_img(index, MNISTinCCD_C.lb_mode, False, *preprocess)
                    )
        return data_slice

    @staticmethod
    def accuracy(Y_HAT: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        # computer = SSIM(size_average=False)
        computer = SSIM(MNISTinCCD_C.lb_mode)
        y_hat = ttools.tensor_to_img(Y_HAT, MNISTinCCD_C.lb_mode)
        y = ttools.tensor_to_img(Y, MNISTinCCD_C.lb_mode)
        y_hat = ttools.img_to_tensor(y_hat, Y_HAT.dtype, Y_HAT.device)
        y = ttools.img_to_tensor(y, Y.dtype, Y.device)
        ssim = computer(y_hat, y)
        return torch.sum(ssim)

    @staticmethod
    def unwrap_fn(inputs: torch.Tensor,
                  predictions: torch.Tensor,
                  labels: torch.Tensor,
                  acc_s: torch.Tensor,
                  loss_es: torch.Tensor,
                  ) -> Any:
        print('正在拼装结果……')
        ret = []
        inp_s = ttools.tensor_to_img(inputs, MNISTinCCD_C.fea_mode)
        pre_s = ttools.tensor_to_img(predictions, MNISTinCCD_C.lb_mode)
        lb_s = ttools.tensor_to_img(labels, MNISTinCCD_C.lb_mode)
        # 制作输入、输出、标签对照图
        ret = itools.concat_imgs(
            *[
                [(inp, 'input'), (pre, 'prediction'), (lb, 'labels')]
                for inp, pre, lb in zip(inp_s, pre_s, lb_s)
            ],
            comment=[
                f'acc = {acc * 100: .3f}%, loss = {ls: .5f}'
                for acc, ls in zip(acc_s, loss_es)
            ]
        )
        return ret

    @staticmethod
    def save_fn(result: Iterable[Image], root: str) -> None:
        pytools.check_path(root)
        print('正在保存结果……')
        for i, res in enumerate(result):
            res.save(os.path.join(root, f'{i}.jpg'))

    def to_dataset(self) -> Tuple[LazyDataSet, LazyDataSet] or Tuple[DataSet, DataSet]:
        """
        根据自身模式，转换为合适的数据集，并对数据集进行预处理函数注册和执行。
        对于懒加载数据集，需要提供read_fn()，签名须为：
            read_fn(fea_index: Iterable[path], lb_index: Iterable[path]) -> Tuple[features: Iterable, labels: Iterable]
            数据加载器会自动提供数据读取路径index
        :return: (训练数据集、测试数据集)，两者均为pytorch框架下数据集
        """
        if self.__f_lazy or self.__l_lazy:
            ds = LazyDataSet(
                self.__train_f, self.__train_l, read_fn=self.read_fn
            )
            ds.register_preprocess(
                feaIndex_calls=self.feaIndex_preprocesses, lbIndex_calls=self.lbIndex_preprocesses
            )
            test_ds = LazyDataSet(
                self.__test_f, self.__test_l, read_fn=self.read_fn
            )
            test_ds.register_preprocess(
                feaIndex_calls=self.feaIndex_preprocesses, lbIndex_calls=self.lbIndex_preprocesses
            )
        else:
            ds = DataSet(self.__train_f, self.__train_l)
            test_ds = DataSet(self.__test_f, self.__test_l)
        ds.register_preprocess(features_calls=self.fea_preprocesses, labels_calls=self.lb_preprocesses)
        ds.preprocess()
        test_ds.register_preprocess(features_calls=self.fea_preprocesses, labels_calls=self.lb_preprocesses)
        test_ds.preprocess()
        return ds, test_ds

    def __set_preprocess(self, module: type):
        """
        根据处理模型的类型，自动指定预处理程序
        :param module: 用于处理本数据集的模型类型
        :return: None
        """
        if module == AdaP2P:
            self.feaIndex_preprocesses = [
                lambda d: np.array(d),
            ]
            self.lbIndex_preprocesses = [
                lambda d: np.array(d),
            ]
            # self.fea_preprocesses = [
            #     lambda fea: np.array(fea),
            #     lambda fea: itools.extract_and_cat_holes(
            #         fea, hole_pos, hole_size, len(xv), len(yv)),
            #     lambda d: torch.from_numpy(d),
            #     lambda d: d.type(torch.float32),
            #     lambda d: normalize(d),
            # ]
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
                lambda fea: np.array(fea),
                lambda d: torch.from_numpy(d),
                lambda d: d.type(torch.float32),
                lambda d: normalize(d),
            ]
        else:
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

    def __len__(self):
        return len(self.__train_f)
