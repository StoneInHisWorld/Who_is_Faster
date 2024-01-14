import utils.func.torch_tools as tools
from data_related import data_related as dr
from mnistinccd_c import MNISTinCCD_C as DataSet
from networks.nets.adawzynet import AdaWZYNet as Net
from utils.hypa_control import ControlPanel

net_name = Net.__name__.lower()
# 调参面板
cp = ControlPanel(
    DataSet,  # 处理的数据集类
    f'../../config/hp_control/{net_name}_hp.json',
    f'../../config/settings.json',  # 运行配置json文件路径
    f'../../log/{net_name}_log.csv',  # 结果存储文件路径
    f'../../log/trained_net/{net_name}/',  # 训练成果网络存储路径
    f'../../log/imgs/{net_name}/'  # 历史趋势图存储路径
)

print('正在整理数据……')
data = DataSet(
    where='../../data/', which='2023-11-12-17.55', module=Net,
    data_portion=cp['data_portion'], lazy=cp['lazy']
)
acc_func = DataSet.accuracy

print('数据预处理中……')
train_ds, test_ds = data.to_dataset()
train_sampler, valid_sampler = dr.split_data(train_ds, 0.8, 0, 0.2)
dataset_name = DataSet.__class__.__name__
del data

# 多组参数训练流水线
for trainer in cp:
    with trainer as hps:
        # 读取训练超参数
        max_load, base, num_epochs, batch_size, ls_fn, lr, optim_str, w_decay, init_meth, comment = hps
        device = cp.device
        for ds in [train_ds, test_ds]:
            ds.to(device)
        # 获取数据迭代器并注册数据预处理函数
        train_iter, valid_iter = [
            dr.to_loader(train_ds, batch_size, sampler=sampler, max_load=max_load)
            for sampler in [train_sampler, valid_sampler]
        ]
        test_iter = dr.to_loader(test_ds, batch_size)

        print(f'正在构造{net_name}……')
        # 构建网络
        net = Net(
            DataSet.fea_channel,
            train_ds.feature_shape[1], 2, [train_ds.label_shape[0]],
            base,
            device=device, init_meth=init_meth
        )
        trainer.register_net(net)

        print(f'本次训练位于设备{device}上')
        # 进行训练准备
        optimizer = tools.get_optimizer(net, optim_str, lr, w_decay)
        ls_fn = tools.get_loss(ls_fn)
        history = net.train_(
            train_iter, valid_iter=valid_iter, optimizer=optimizer, num_epochs=num_epochs,
            ls_fn=ls_fn, acc_fn=acc_func
        )
        # history = net.train__(
        #     train_iter, optimizer, num_epochs, ls_fn, acc_func, valid_iter
        # )

        print('测试中……')
        test_acc, test_ls = net.test_(test_iter, acc_func, ls_fn)
        cp.register_result(history, test_acc, test_ls, ls_fn=ls_fn, acc_fn=acc_func.__name__)
        del ls_fn, optimizer, history, net
