该数据为计算Ricker子波平移的Wasserstein度量（即本次final project 1中线性规划问题最优解）所需要的的cost matrix C和边缘分布\mu=fx, \nu=fy；
具体数据保存在.mat文件中。

我们考察的Ricker子波表达式为：R(t) = A(1-2\pi^2f_0^2t^2)e^{-\pi^2f_0^2t^2},  此处振幅A = 1；频率f_0 = 2；
此算例中我们将R(t)定义在区间[-3,3]中，并将[-3,3]等分地取了500个点，即横坐标x = linspace(-3,3,500)；
平移距离为s = 2，得到平移后的Ricker子波横坐标 y = x+s;
cost matrix C表达式则为C_{ij} = (x_i-y_j)^2，两个边缘分布为\mu = \nu为1*500的行向量，第i项取值为 R(x_i)^2/(sum_{j=1}^{500}R(x_j)^2)；
计算出来的Wasserstein度量的精确值应为 W_2^2(R(t),R(t+s)) = s^2；

即根据此cost matrix C和边缘分布fx fy得到的最优输运问题的最优值为s^2 = 4；

Remark: 如果发现用自己写的first-order method解这个问题所耗的时间较长，精度不高（指难以达到1e-7这种级别的精度）的话，不用担心，这是正常现象。