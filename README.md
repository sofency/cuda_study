指令延迟指计算指令从调度到指令完成所需的时钟周期
如果在每个时钟周期都有就绪的线程束可以被执行，此时GPU处于满负荷状态
指令延迟被GPU满负荷计算状态所掩盖的现象称为延迟隐藏
延迟隐藏对GPU编程开发很重要，GPU设计目标是处理大量但是轻量级的线程函数

如何计算满足延迟隐藏所需要的线程束数量
  延迟时间 * GPU吞吐量


GPU 内存频率查询
nvidia-smi -a -q -d CLOCK | fgrep -A 3 "Max Clocks" |fgrep "Memory"
计算GPU每个时钟周期吞吐量: 内存带宽/内存频率
并行性需求 = 内存操作延迟 * 每时钟周期吞吐量
再根据应用程序实际的内存操作，计算出对应所需的线程束的量

流处理器占有率 = SM活动线程束 / SM最大线程束
查询GPU属性获取SM最大线程数
SM最大线程数 / 32 就是SM最大线程束

查看寄存器和共享内存使用情况
进行编译
nvcc -ptxas-options=-v code.cu -o code

查看帮助文档 nvprof --query-metrics | grep occupancy

线程束占有率分析
nvprof --metrics achieved_occupancy ./code

内核数据读取效率
nvprof --metrics gld_throughut ./code

程序对设备内存带宽利用率
nvprof --metrics gld_efficiency ./code




