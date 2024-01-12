import utils.func.log_tools as ltools
import utils.func.torch_tools as ttools
from data_related import data_related as dr
from mnistinccd_c import MNISTinCCD_C as DataSet
from networks.nets.pix2pix import Pix2Pix as Net

net_name = Net.__name__.lower()
max_load = 5000
read_queue = range(1663, 1743)
device = 'cpu'
log_path = f'../../log/{net_name}_log.csv'

print('正在整理数据……')
data = DataSet(
    '../../data/', '2023-11-12-17.55', lazy=False, data_portion=0.01,
    f_req_sha=Net.required_shape, l_req_sha=Net.required_shape
)
acc_func = DataSet.accuracy

print('数据预处理中……')
_, test_ds = data.to_dataset()
test_ds.to(device)
data_iter = dr.to_loader(test_ds, batch_size=32, shuffle=False, max_load=max_load)

for exp_no in read_queue:
    hp = ltools.get_logData(log_path, exp_no)
    print(
        f'---------------------------实验{exp_no}号的结果'
        f'---------------------------'
    )
    # 构建网络
    net = Net(
        DataSet.fea_channel,
        DataSet.lb_channel,
        int(hp['base']),
        device=device
    )
    net.load_state_dict_(f'../../log/trained_net/{net_name}/{exp_no}.ptsd')

    result = net.predict_(
        data_iter,
        ttools.get_loss(hp['ls_fn']),
        DataSet.accuracy,
        DataSet.unwrap_fn
    )
    DataSet.save_fn(result, f'../../data/RESULT/{net_name}/{exp_no}/')
    print(
        f'----------------------已保存实验{exp_no}号的结果'
        f'----------------------'
    )
