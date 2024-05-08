# Who_is_Faster
研究Pytorch神经网络训练过程中各种有关GPU内存的参数设置对训练速度、显存和内存占用的影响

需要做多组实验：
1. 不使用pinmemory&nonblocking直接将所有数据放在显存中
2. 使用pin_memory，数据取出后再放入显存
3. 使用pin_memory&non_blocking，数据取出后放入显存
4. 使用p&n&prefetchfactor，数据取出后再放入显存
5. 使用p&n&pre，以及backgroundgenerator，数据取出后再放入显存
6. 使用n_worker>1，以及以上所有参数 
每个变量10组实验，100epoch，数据量10%、20%……递增