# Who_is_Faster
研究Pytorch神经网络训练过程中各种有关GPU内存的参数设置对训练速度、显存和内存占用的影响

## 实验设计
1. 不使用pinmemory&nonblocking，数据取出后再放入显存中
2. 不使用pinmemory&nonblocking直接将所有数据放在显存中
3. 使用bkgenerator，数据取出后再放入显存
4. 使用p&bkgenerator，数据取出后再放入显存
5. 使用p&n&bkgenerator，数据取出后再放入显存
6. 使用p&n&n_worker=4&prefetch=2，数据取出后再放入显存
7. 使用p&n&n_worker=4&prefetch=8，数据取出后再放入显存
8. 使用p&n&n_worker=8&prefetch=2，数据取出后再放入显存
9. 使用p&n&n_worker=8&prefetch=8，数据取出后再放入显存
10. 使用pin_memory&nonblocking，数据取出后再放入显存
11. 使用pin_memory，数据取出后再放入显存
每种参数组合运行10次，迭代100世代，数据量采用10%、40%、70%
### 说明
- 100%数据集中含有训练集50000张图片、测试集10000张图片，图片经预处理后大小为64x64。训练集中的80%用于训练，20%用于验证。
- 本实验使用的模型为GoogLeNet版本1，数据集为私有数据集。如要运行本程序，请将程序中使用的数据集替换为您使用的数据集。
- 起初打算每个组合运行10次，但是由于运行时间过长，改为每个组合运行3次。因此大部分数据来源于运行3次的平均，另一部分来源于运行10次的平均
- p代表pin_memory，是torch.utils.data中的DataLoader类初始化时设置的参数pin_memory
- n代表non_blocking，是进行数据迁移时设置的参数，需要搭配pin_memory参数。
- bkgenerator代表使用prefetch_generator包中的BackgroundGenerator作为数据迭代器，程序中的BackgroundGenerator集成了数据设备迁移操作。
- n_worker代表torch.utils.data.DataLoader获取数据时使用的处理机个数。此参数和BackgroundGenerator互斥，同时使用会发生死锁问题。
- prefetch代表torch.utils.data中的DataLoader类初始化时设置的参数prefetch_factor。