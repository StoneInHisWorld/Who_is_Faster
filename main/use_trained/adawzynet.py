import utils.func.log_tools as ltools
import utils.func.torch_tools as ttools
from data_related import data_related as dr
from mnistinccd_c import MNISTinCCD_C as DataSet
from networks.nets.adawzynet import AdaWZYNet as Net

net_name = Net.__name__.lower()
max_load = 5000
read_queue = range(24, 28)
device = 'cpu'
log_path = f'../../log/{net_name}_log.csv'

print('正在整理数据……')
data = DataSet(
    where='../../data/', which='2023-11-12-17.55', module=Net,
    data_portion=1, lazy=False
)
acc_func = DataSet.accuracy

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
    net = Net(
        DataSet.fea_channel,
        test_ds.feature_shape[1], 2, [test_ds.label_shape[0]],
        base_channels=int(hp['base']), device=device
    )
    net.load_state_dict_(f'../../log/trained_net/{net_name}/{exp_no}.ptsd')

    results = net.predict_(
        data_iter,
        DataSet.accuracy,
        ttools.get_loss(hp['ls_fn']),
        DataSet.unwrap_fn
    )
    DataSet.save_fn(results, f'../../data/RESULT/{net_name}/{exp_no}/')
    print(
        f'----------------------已保存实验{exp_no}号的结果'
        f'----------------------'
    )
