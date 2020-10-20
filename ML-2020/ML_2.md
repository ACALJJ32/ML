# 1.神经网络与BP神经网络  
* 神经网络模型：
![Alt text](https://images2015.cnblogs.com/blog/853467/201606/853467-20160630140644406-409859737.png)  
&ensp;&ensp;&ensp;&ensp;这是一个典型的三层神经网络模型，有输入层、隐含层和输出层，通常在隐含层引入一个非线性变换函数，例如sigmoid函数；通过该函数可以将模型变成一个非线性模型。    
&ensp;&ensp;&ensp;&ensp;假设我们有一组数据$\{X{1},X_{2},\dots,X_{n}\}$，每一个$X_{i}$代表一个$N$维的数据，将数据输入神经网络模型后，得到一个结果$y_{j}$;$y_{j}$可以是一个数(0或1)，也可以是一组向量。通常，这样得到结果往往是很难达到满意的结果，所以我们自然想到通过定义一个损失函数，通过损失函数来判断如何更新我们神经网络中各个链接的权重。  
* BP(Backpropagation)  
&ensp;&ensp;&ensp;&ensp;介绍神经网络模型之前需要再回顾共轭梯度法(Gradient Descent)。  
&ensp;&ensp;&ensp;&ensp;给定一组参数：$\theta = \{w_{1},w_{2},\dots,b_{1},b_{2},\dots,\}$，通过不断迭代更新参数：
$$\theta^{0}\rightarrow\theta^{1}\rightarrow\theta^{2}\dots\dots$$  
$$\nabla{L(\theta)}=\begin{bmatrix}
{\frac{\partial{L(\theta)}}{\partial{w_{1}}}}\\
{\frac{\partial{L(\theta)}}{\partial{w_{2}}}}\\
{\vdots}\\
{\frac{\partial{L(\theta)}}{\partial{b_{1}}}}\\
{\frac{\partial{L(\theta)}}{\partial{b_{2}}}}\\
{\vdots}\\
\end{bmatrix}$$  
&ensp;&ensp;&ensp;&ensp;计算好每个参数的偏导数后，可以更新神经网络中的参数：  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;$\theta^{1} =\theta^{0} - {\eta} {\nabla{L(\theta^{0})}}$
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;$\theta^{2} =\theta^{1} - {\eta} {\nabla{L(\theta^{1})}}$  

# 2. CNN  
$$x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$$  
$\sum_{i=0}^N\int_{a}^{b}g(t,i)\text{d}t$  
$\sum_{i=0}^N\int_{a}^{b}g(t,i)\text{d}t$