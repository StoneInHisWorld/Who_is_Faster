import torch

import utils.func.log_tools as ltools
from data_related import data_related as dr
from mnistinccd_c import MNISTinCCD_C as DataSet
from networks.nets.googlenet import GoogLeNet as Net

net_name = Net.__name__.lower()
max_load = 5000
read_queue = [33, 34]
device = 'cpu'
log_root = '../../log/'
log_path = log_root + f'{net_name}_log.csv'

print('正在整理数据……')
# 请注意data_portion太低，可能会导致独热编码的label.shape变化！
data = DataSet(
    where='../../data/', which='2023-11-12-17.55', module=Net,
    data_portion=0.5, lazy=False
)
criterion_a = DataSet.get_criterion_a()

print('数据预处理中……')
_, test_ds = data.to_dataset()
test_ds.to(device)
data_iter = dr.to_loader(test_ds, batch_size=32, shuffle=False, max_load=max_load)
del _

for exp_no in read_queue:
    hp = ltools.get_logData(log_path, exp_no)
    print(
        f'---------------------------实验{exp_no}号的结果'
        f'---------------------------'
    )
    # 构建网络
    try:
        net = Net(
            DataSet.fea_channel, test_ds.label_shape,
            version="2", device=device,
            init_meth='state', init_args=(log_root + f'trained_net/{net_name}/{exp_no}.ptsd', )
        )
    except FileNotFoundError:
        net = torch.load(log_root + f'trained_net/{net_name}/{exp_no}.ptm')

    results = net.predict_(
        data_iter, criterion_a, DataSet.unwrap_fn,
        ls_fn_args=(hp['ls_fn'], {'reduction': 'none'})
    )
    DataSet.save_fn(results, f'../../data/RESULT/{net_name}/{exp_no}/')
    print(
        f'----------------------已保存实验{exp_no}号的结果'
        f'----------------------'
    )
