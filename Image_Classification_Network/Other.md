# k-则交叉验证(k-fold)
对于小数据集，将其分成k块数据，其中一块作为validation-dataset
剩下的作为training-dataset，循环k次取平均去优化超参数

# 过拟合和欠拟合（over fitting and underfitting）
- 过拟合：缺乏泛化性（大量的记住了训练数据，包括拟合了噪音），泛化误差增大
- 欠拟合：精度太小  
**类似泰勒展开的阶数**
## 权重衰退(weight decay)
- 通过限制参数值的选择范围来解决过拟合的问题  
  1. 硬性限制直接限制
  2. 加入`panalty`进行柔性限制,加入$\frac{\lambda}{2}||W||^2$作为惩罚项  

# 丢弃法（dropout)
在层间加入噪音，对每个元素进行扰动
$$ 
x'_i=  
\begin{cases}
0   & \text{with probablity p}\\
\frac{x_i}{1-p} & \text{otherise}
\end{cases}
$$
由此可以得到，E[$x'_i$]=x

# 数值稳定性

## 常见问题
- overflow
- underflow

# 批量归一化BN（将数据的期望和方差固定住）
- 最初为了减少内部协变量转移
- 后来指出可能就是通过在小批量里加入噪音，即加入随机偏移和随机缩放
- 可以加速收敛，但一般不改变模型精度  
``` python
nn.BatchNorm2d()
```
- 相对xavier在初始化是对权重做稳定操作，BN是在训练过程中做这件事

# FPGA(可编程阵列)
# Asic
- Systolic Array:针对深度学习计算的硬件