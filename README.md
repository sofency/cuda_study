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


### 统一内存空间
统一内存空间在CUDA6.0中华首次引入的编程模型，只能在Kepler及其以上的架构中使用，
统一内存空间定义了一个托管内存空间，在这个托管内存空间中，所有处理器感知的是一个一致的内存映像
在统一内存空间中，底层系统管理数据的定位和访问
统一内存空间中，GPU,CPU间的数据传递不需要显式使用cudaMemcpy
统一内存空间提供数据单一指针模型

内存获取

静态声明 __managed__
动态获取cudaMallocManaged
释放 cudaFree()


### 页锁定内存
主机上分配的内存是分页内存，在操作系统的调度下，数据所存放的实际物理地址空间是可变的
从主机拷贝数据时，GPU驱动首先将数据拷贝到主机中的临时页锁定内存
分配过多的分页内存会降低系统整体性能
分配页锁定内存 cudaMallocHost 释放页锁定内存： cudaFreeHost

### 零拷贝内存
通常情况下，GPU和CPU无法直接访问彼此内存空间中定义的变量
零拷贝内存对GPU和CPU都可见
零拷贝内存作用
  设备内存不够时可以使用零拷贝内存
  避免主机和设备间数据拷贝
使用零拷贝内存时要同步主机和设备对内存的访问

### 动态共享分配内存

extern声明动态共享内存变量
调用内核时指定动态共享内存大小
kernel<<<grid, block, isize * sizeof(int)>>>(...)


### 共享内存屏障
#### 弱序内存
GPU线程将数据写入内存的顺序和其对应代码出现的顺序不一定完全一致 (就是代码优化)
写入内存中的数据对其他线程可见的顺序和数据写入的顺序不一定完全一致
#### 线程屏障
线程屏障保证线程块中所有线程必须都执行到某个特定点才能执行
建立线程屏障 __syncthreads
保证线程此前对全局内存，共享内存所做的任何改动对线程块中的所有线程可见
如果要在条件分支中使用线程屏障，那么此条件分支对于线程块中所有线程束具有一致的效果

线程屏障不会等待已经结束的线程

### 线程块栅栏概念

内存栅栏保证在此之前写入内存的数据被所有线程可见
线程块栅栏作用域统一线程块，保证对共享内存，全局内存数据同步
__threadfence_block

### 线程网格栅栏
网格栅栏作用范围为相同线程网格
网格栅栏保证写入全局内存中的数据对网格中所有线程可见

### shuffle指令
shuffle指令使同一线程束中的线程可以直接交换数据
shuffle指令的数据交换不依赖于共享内存或全局内存，延迟极低
shuffle指令分为两类: 用于整型数据，用于浮点型数据

lane代表线程束中的一个线程
lane的索引范围为0～31



### 显示和隐式同步

隐式同步: 会阻止主机线程往下执行，直到这些接口返回
  主机页锁定内存分配
  设备内存分配
  设备内存初始化
  在设备上拷贝数据
  修改一级缓存和共享内存配置

显示同步: 必须自己使用同步函数
  设备同步: cudaDeviceSynchronize()
  流同步: cudaStreamSynchronize()
  事件同步: cudaEventSynchronize()
  不同流同步: cudaStreamWaitEvent()

### 原子操作
cuda 只为全局内存和共享内存提供了原子操作，算术元算函数，为运算函数，交换函数
atomicAdd / Sub / Exch / Min / Max / Inc / Dec / CAS / And / Or / Xor

### cuda-gdb 调试
nvcc -g -G file.cu -o file
cuda-gdb ./file
help cuda 查看命令


b 22 断点设置在22行
b initialData 符号断点
b file.cu:22 断点 代码行断点

内核入口断点
set cuda bread_on_launch [type]
type 可以是 none / application (由应用程序发起的) / system / all 


r  运行到断点处
cuda kernel block thread 软件坐标
cuda kernel grid thread 
cuda device sm wrap lane 硬件坐标
cuda thread 3 转换到3线程
cuda thread 可以查看当前是哪个线程

中断 CTRL + C 中断死循环和死锁
单步执行 n
查看变量id  p id 


条件断点
bread martixadd.cu:22 if threadIdx.x == 5

or

break 22 先设置断点
cond 1 threadIdx.x == 5  // cond 1 是断点的序号

B是一维数组
print B[0]@3 b数组前3元素


nvcc -g -G -arch=sm_35 -rdc=true file.cu -o file

device 1  查询第一个设备信息 类比 sm wrap lane grid 
kernel 1 查询第一个kernel
查询所有断点 breakpoint all 

info cuda contexts 
info cuda blocks

查看寄存器中的值
$R<regnum>  <regnum>是寄存器的编号 $R10
寄存器内容查看 info registers [$R<regnum>]

p $i 查看当前变量存储位置 可能会显示寄存器地址
然后再查看寄存器 中的值
p $R0可以查看

info registers system 预测和状态寄存器

### 事件通知
上下文事件 cuda上下文在创建、入栈、出栈
内核事件

set cuda context_events on 开启上下文事件


cuda-memcheck工具可用于检测CUDA程序中各种内存访问错误问题

// 进行编译 检测内存错误
nvcc -g -G -lineinfo -Xcompiler -rdynamic debuf-segfault.cu -o debug-segfault
cuda-memcheck ./debug-segfault

自动错误检测
nvcc -g -G detect.cu -o detect
cuda-gdb ./detect

set cuda api_failures ignore 
r 运行程序