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
- 100%数据集中含有训练集50000张图片，测试集10000张图片。每张图片经预处理后大小为64x64。训练集中的80%用于训练，20%用于验证。
- 本实验使用的模型为GoogLeNet版本1，数据集为私有数据集。如要运行本程序，请将程序中使用的数据集替换为您使用的数据集。
- p对应torch.utils.data模块中的DataLoader类初始化时设置的参数pin_memory，默认值为`False`
  >- 该操作需要GPU支持CUDA。  
  >- `pin_memory == True`时，会将数据移入锁页内存。对于分页内存中的数据，GPU无法读取，需要先拷贝到锁页内存中，再传输给GPU内存；对于锁页内存中的数据GPU可以通过DMA直接传输，从而加速数据传输。  
  >- 若数据集中不存在`torch.Tensor`对象，则程序不会将数据集放入锁页内存。
  >- 建议在内存充足的情况下使用，否则会造成严重的性能问题。
  >>参考：  
  > https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
  > https://www.jianshu.com/p/e92e72c0ba51
  
- n代表non_blocking，是进行数据迁移时使用的`to()` `cuda()`方法设置的参数，当其值为`True`时开启异步GPU拷贝功能。默认值为`False`。
  >- 仅在`pin_memory == True`时才能使用。
  >- GPU可以通过异步拷贝功能开启异步传输模式，通过重叠其他操作时间和数据迁移时间来减少总体训练时间。编程范例如下：
  >```python
  > import torch
  >
  > x = torch.range(0, 1)
  > # 将 CPU Tensor 对象移动到 GPU 上，并启用异步传输模式
  > x_gpu = x.cuda(non_blocking=True)
  >
  > # 执行其它与x无关操作
  >
  > # 等待数据异步传输完成
  > torch.cuda.synchronize()
  >```
  >>参考：  
  > https://blog.csdn.net/hxxjxw/article/details/123585191
  > https://blog.csdn.net/bj_zhb/article/details/131019244
- n_worker对应torch.utils.data中的DataLoader类初始化时设置的参数num_workers，代表DataLoader数据加载时使用的子进程个数。
  - num_workers默认值为`0`，此时数据会被加载到主进程。
- prefetch对应torch.utils.data中的DataLoader类初始化时设置的参数prefetch_factor。
  >- prefetch_factor的数值表示每个处理机提前加载的数据批量数，num_batch == prefetch_factor * num_workers
  >- `num_worker == 0`时，本参量默认值为`None`；num_workers > 0时，本参量默认值为2
- bkgenerator代表使用`prefetch_generator`包中的`BackgroundGenerator`作为数据迭代器，并集成了数据设备迁移操作。
  >- 此参数和`num_workers`互斥，同时使用会发生死锁问题。
  >>参考：https://pypi.org/project/prefetch-generator/


## 参考
1. https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader  
2. https://www.jianshu.com/p/e92e72c0ba51
3. https://blog.csdn.net/hxxjxw/article/details/123585191
4. https://blog.csdn.net/bj_zhb/article/details/131019244
5. https://pypi.org/project/prefetch-generator/