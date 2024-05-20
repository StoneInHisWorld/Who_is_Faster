# Who_is_Faster
研究pytorch神经网络训练过程中各种有关GPU内存的参数设置对训练速度、显存和内存占用的影响

---
## 引入
深度学习是一种基于某种人工神经网络进行自动建模的技术，是机器学习下的一个分支，因其强大的建模能力使得其能够应用于各个学科当中。
技术的普适性推动技术的广泛应用，技术的广泛应用反哺于技术本身。随着深度学习的用户增加，加入深度学习技术社区的人也越来越多，开发深度学习开源框架的人也越来越多。
开源框架是指一份提供API接口的运行依赖，是集成某些编写好的功能，基于某种技术目标实现的代码编写范例。在学习深度学习时，最快的办法并非学习其数学理论，从零开始实现理论的每个细节，
而是借助深度学习框架实现简单的项目，从代码调试中逐步理解深度学习的理论体系。近年来深度学习框架迅速发展，知名的框架有PyTorch、Tensorflow、Theano、Caffe。
其中PyTorch具有自动求微分以及自动构造计算图机制，使其易用性得以提高；此外PyTorch还具有支持python的即时运行特性，方便进行代码调试；同时，PyTorch的性能不输其他深度学习框架。
这些特性使得PyTorch其成为一个易于学习，对新手友好的深度学习框架，被企业采用的同时也在学术界领域发展壮大。然而PyTorch框架中具有大量可调整参数设置，其中的部分参数设置关乎训练速度以及显存占用，影响生产研究的效率。
研究并阐明这些参数的技术理念以及用途，对于执行相关参数调整工作的用户十分重要。

---
### 参数说明
1. torch.utils.data模块中，DataLoader类初始化时设置的参数pin_memory，简写为”p“，默认值为`False`
  >- 该操作需要GPU支持CUDA。  
  >- `pin_memory == True`时，会将数据移入锁页内存。对于分页内存中的数据，GPU无法读取，需要先拷贝到锁页内存中，再传输给GPU内存；对于锁页内存中的数据GPU可以通过DMA直接传输，从而加速数据传输。  
  >- 若数据集中不存在`torch.Tensor`对象，则程序不会将数据集放入锁页内存。
  >- 建议在内存充足的情况下使用，否则会造成严重的性能问题。
  > ---
  > 参考：  
  > https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
  > https://www.jianshu.com/p/e92e72c0ba51
  
2. Pytorch进行数据迁移时使用的`to()` `cuda()`方法可以设置参数”non_blocking“，简写为”n“。当其值为`True`时开启异步GPU拷贝功能。默认值为`False`。
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
  > ...
  >
  > # 等待数据异步传输完成
  > torch.cuda.synchronize()
  >```
  > ---
  > 参考：  
  > https://blog.csdn.net/hxxjxw/article/details/123585191
  > https://blog.csdn.net/bj_zhb/article/details/131019244
3. torch.utils.data模块中，DataLoader类初始化时设置的参数num_workers代表了DataLoader数据加载时使用的子进程个数，简写为”nw“。
  > num_workers默认值为`0`，此时数据会被加载到主进程。
4. torch.utils.data模块中，DataLoader类初始化时设置的参数prefetch_factor，简写为”pf“。
  >- prefetch_factor的数值表示每个处理机提前加载的数据批量数，num_batch == prefetch_factor * num_workers
  >- `num_worker == 0`时，本参量默认值为`None`；num_workers > 0时，本参量默认值为2
  > ---  
  > 参考：  
  > https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
5. 简称”bkgg“代表使用`prefetch_generator`包中的`BackgroundGenerator`作为数据迭代器，并集成了数据设备迁移操作。
  >- 此参数和`num_workers`互斥，同时使用会发生死锁问题。
  > --- 
  > 参考：  
  > https://pypi.org/project/prefetch-generator/
6. 神经网络的训练可以选择使用CPU，也可以使用GPU进行计算，而GPU上的CUDA工具可以加速神经网络的训练，因此推荐有经济实力的用户使用GPU进行训练。
不过pytorch框架下规定参与某次运算的全部Tensor对象必须位于同一设备上，因此神经网络的训练中可以选择将数据集全部放入显存中，或者取出数据批后再放入显存中。

---
## 实验设计
针对上述参数，本次研究采用了如下设计：

| 组合编号 | p     | n     | nw | pf   | bkgg  | 加载数据后再放入显存 |
|------|:------|-------|----|------|-------|------------|
| 1    | True  | False | 0  | None | False | True       |
| 2    | True  | True  | 0  | None | False | True       |
| 3    | True  | True  | 8  | 8    | False | True       |
| 4    | True  | True  | 8  | 2    | False | True       |
| 5    | True  | True  | 4  | 8    | False | True       |
| 6    | True  | True  | 4  | 2    | False | True       |
| 7    | True  | True  | 0  | None | True  | True       |
| 8    | True  | False | 0  | None | True  | True       |
| 9    | False | False | 0  | None | True  | True       |
| 10   | False | False | 0  | None | False | True       |
| 11   | False | False | 0  | None | False | False      |
*<center>表1：参数组合表，其中“p”、“n”、“nw”、“pf”列中单元格为所代表参数的取值，“bkgg”、“加载数据后再放入显存”列单元格中的布尔值代表是否启用该功能</center>*

在训练过程中，每种参数组合运行3次，迭代100世代，数据量采用10%、40%、70%。为了避免内存监控对训练时间产生影响， 本次研究还进行了内存监控的对照组实验。
本次实验全程都会进行显存监控，以评估各个参数组合对显存占用的影响。显存占用的测试主要是利用pytorch框架中自带的显存占用检测函数，示例代码如下：
```python
import torch

# 开启显存占用
torch.cuda.memory._record_memory_history(True)

# 训练过程
...

# 计算显存占用
max_GPUmemory_allocated = torch.cuda.max_memory_allocated()
max_GPUmemory_reserved = torch.cuda.max_memory_reserved()
```

---
## 实验结果
本研究分析了每条实验数据，对每种参数组合所进行的训练时间、显存占用进行平均值计算，得到下面的图表。
### 训练时间分析
![Training Duration Overview.png](Training%20Duration%20Overview.png)
*<center>图1：训练时间分析总览图。y轴标签为采用的参数集合说明，x轴为数据集的切片大小</center>*

对每个数据集切片分别绘制条形图，如下：
![Training Duration Figure(data_portion=0.1).png](Training%20Duration%20Figure%28data_portion%3D0.1%29.png)
*<center>图2：训练时间分析图，数据切片大小为10%。y轴标签为采用的参数集合说明，x轴为数据集的切片大小</center>*

![Training Duration Figure(data_portion=0.4).png](Training%20Duration%20Figure%28data_portion%3D0.4%29.png)
*<center>图3：训练时间分析图，数据切片大小为40%。y轴标签为采用的参数集合说明，x轴为数据集的切片大小</center>*

![Training Duration Figure(data_portion=0.7).png](Training%20Duration%20Figure%28data_portion%3D0.7%29.png)
*<center>图4：训练时间分析图，数据切片大小为70%。y轴标签为采用的参数集合说明，x轴为数据集的切片大小</center>*

从以上三张图可以看出：
1. 神经网络的训练时间随着数据集切片的增大而增大。
2. 将整个数据集放入显存中，训练速度是最快的。
3. 使用pin_memory的确能够加速训练时间，不过仅限于两种情况： 
   1. 使用`BackgroundGenerator`进行训练。
   2. 使用较小的数据集进行训练。
4. 使用`non_blocking`的确能够加速训练时间，不过加速仅限于较大的数据集。对于较小的数据集，`non_blocking`甚至会拖累训练程序。 
5. 使用`num_worker`会明显拖慢速度。降低`num_worker`的使用量后，训练速度会增加。
6. 针对较小的数据集，较大的`prefectch_factor`对较多的`num_worker`比较有效果，较小的`prefetch_factor`对于较少的`num_worker`有效。
7. `BackgroundGenerator`确实能够加速训练时间。

### 显存占用分析
![GPU Memory Usage Overview.png](GPU%20Memory%20Usage%20Overview.png)
*<center>图5：显存占用总览图，包含各种数据集切片大小。
左侧y轴（主坐标轴）标签为最大分配显存大小，绘制以条形图，单位为MiB；右侧y轴的标签为最大保留显存大小，绘制以折线图，单位为MiB。
x轴标签为参数组合名称，每个参数组合包括大小不同数据集切片的训练数据，从左自右分别为10%、40%、70%。</center>*

由于将整个数据集放入显存中的显存占用远远大于按照数据批放入显存的，因此为了能更加清晰地观察其他参数组合对于显存占用的影响，将前者去除之后再绘制一幅总览图：
![GPU Memory Usage Overview(Except Putting the Whole Dataset in GPU Memory).png](GPU%20Memory%20Usage%20Overview%28Except%20Putting%20the%20Whole%20Dataset%20in%20GPU%20Memory%29.png)
*<center>图6：去除“将整个数据集放入显存”的数据后的显存占用总览图，包含各种数据集切片大小。
左侧y轴（主坐标轴）标签为最大分配显存大小，绘制以条形图，单位为MiB；右侧y轴的标签为最大保留显存大小，绘制以折线图，单位为MiB。
x轴标签为参数组合名称，每个参数组合包括大小不同数据集切片的训练数据，从左自右分别为10%、40%、70%。</center>*

对每个数据集切片分别绘制条形图，如下：
![GPU Memory Usage Figure(Except Putting the Whole Dataset in GPU Memory)data_portion=0.1.png](GPU%20Memory%20Usage%20Figure%28Except%20Putting%20the%20Whole%20Dataset%20in%20GPU%20Memory%29data_portion%3D0.1.png)
*<center>图7：去除“将整个数据集放入显存”的数据后的显存占用分析图，数据集切片大小为10%。
条形图中，蓝色标签为最大分配显存大小，橙色标签为最大预留显存大小，单位均为MiB；x轴标签为各种参数组合名称。</center>*

![GPU Memory Usage Figure(Except Putting the Whole Dataset in GPU Memory)data_portion=0.4.png](GPU%20Memory%20Usage%20Figure%28Except%20Putting%20the%20Whole%20Dataset%20in%20GPU%20Memory%29data_portion%3D0.4.png)
*<center>图8：去除“将整个数据集放入显存”的数据后的显存占用分析图，数据集切片大小为40%。
条形图中，蓝色标签为最大分配显存大小，橙色标签为最大预留显存大小，单位均为MiB；x轴标签为各种参数组合名称。</center>*

![GPU Memory Usage Figure(Except Putting the Whole Dataset in GPU Memory)data_portion=0.7.png](GPU%20Memory%20Usage%20Figure%28Except%20Putting%20the%20Whole%20Dataset%20in%20GPU%20Memory%29data_portion%3D0.7.png)
*<center>图9：去除“将整个数据集放入显存”的数据后的显存占用分析图，数据集切片大小为70%。
条形图中，蓝色标签为最大分配显存大小，橙色标签为最大预留显存大小，单位均为MiB；x轴标签为各种参数组合名称。</center>*

从图中可以看出：
1. 直接将数据集放入显存中，显存占用会随着数据集切片的大小变化；数据取出后再放入显存，数据集切片大小对于显存占用影响非常小。  
2. 数据集切片越大，显存占用就越小。
3. `pin_memory`以及`non_blocking`对显存的占用几乎没有影响。
4. 使用`num_workers`以及`prefetch_factor`会加重显存负担，但实际影响非常小。
5. 使用`BackgroundGenerator`会加重显存负担，但实际影响非常小。
6. `BackgroundGenerator`对显存占用的影响与`num_workers`以及`prefetch_factor`不相上下。  

### 内存占用总览图
将在日后进行更新。

详细的实验数据请参阅 [我的仓库](https://github.com/StoneInHisWorld/Who_is_Faster.git)

---
## 结论
观察上述图表，可以得出如下结论：
1. 如果显存足够，可以将全部数据集放入显存中加速训练。
2. `BackgroundGenerator`是加速神经网络训练的好工具，尽管它会轻微增加显存负担。
3. 使用`BackgroundGenerator`进行数据加载，搭配以`pin_memory`以提高训练速度；不启用`BackgroundGenerator`但数据集较小时，`pin_memory`也能够加速训练；
其他情况下不建议使用`pin_memory`。`pin_memory`虽然有可能加速训练，但是加速幅度实在有限。
4. 使用较大的数据集时，可以使用`non_blocking & pin_memory`的组合提高训练速度，否则将会拖累训练程序。
5. `num_workers`以及`prefetch_factor`的使用需要非常小心谨慎，否则就会拖慢训练速度。
6. 对于配置吃紧的用户，建议神经网络的训练时不要将数据集全部放入显存，而是进行数据加载后再放入显存中，由此避免因显存无法分配而导致的训练程序崩溃。此外，担心使用多线程加速会加重显存负担是没有必要的。

值得说明的是，实验结果数据可能会受到处理机状态的影响，且每条实验数据的采集时间间隔不定，由此得出的实验结论可能会和理论有偏差，研究结论仅对本次研究的实验结果负责。

---
## 讨论
`BackgroundGenerator`使用了多线程机制来预先加载python生成器的内容，将其应用到神经网络训练中的数据加载部分就能加速训练，可以从实验数据中看到出色的加速效果。  
根据pytorch官方论坛中官方人员的回复，一般建议`pin_memory`以及`non_blocking`同时开启，除非是用户的内存十分吃紧。
对于`pin_memory`机制，该机制会将数据加载至锁页内存，使得启用`BackgroundGenerator`的训练过程得以加速；对于`non_blocking`机制，其设计目的是方便用户在数据加载过程中进行一些与加载数据无关的操作，
理论上来说，两机制同时启用确实会对训练速度的提高有所帮助，但从实验数据来看，开启这两个机制似乎并没有达到理想效果，甚至还可能拖累训练程序。
`non_blocking`机制没有加速效果甚至会拖累训练时间，一部分可能是实验程序并不需要在数据加载过程中进行其他操作，因此无法体现其功能；另一部分是开启新的CUDA流可能增加处理机负担。
对于`pin_memory`机制失效，可能是因为使用的数据集相对较小，无法体现其优势。
通过对使用`num_workers`和`prefetch_factor`组合的训练日志数据进行分析，可以推测不恰当的处理机分配策略会加大多线程开销，从而极大地影响训练速度，由此需要进行进一步的研究来确定最佳的`num_workers` `prefetch_factor`指定策略。

---
## 参考
[1] Jake Choi, Heon Young Yeom, Yoonhee Kim. Improving Oversubscribed GPU Memory Performance in the PyTorch Framework[J]. Cluster Computing, 2023, 26：2835~2850.  
[2] Pytorch Contributors. PyTorch documentation[EB/OL]. https://pytorch.org/docs/stable/index.html ，2023-01-01/2024-05-20.  
[3] 星空下的胖子. CUDA 之 Pinned Memory[EB/OL]. https://www.jianshu.com/p/e92e72c0ba51 ，2021-01-28/2024-05-20.  
[4] hxxjxw. Pytorch的cuda non_blocking (pin_memory)[EB/OL]. https://blog.csdn.net/hxxjxw/article/details/123585191 ，2022-03-19/2024-05-20.  
[5] bj_zhb. Pytorch中x.cuda(non_blocking=True)参数解释[EB/OL]. https://blog.csdn.net/bj_zhb/article/details/131019244 ，2023-06-03/2024-05-20.  
[6] Python Software Foundation. prefetch-generator 1.0.3[DB/OL]. https://pypi.org/project/prefetch-generator/ ，2024-01-01/2024-05-20.  
[7] NVIDIA Corporation. CUDA Toolkit[EB/OL]. https://developer.nvidia.com/cuda-toolkit ，2024-01-01/2024-05-20.  