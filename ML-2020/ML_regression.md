## 回归(Regression)  
本质是利用已有的数据，对未来的数据进行预测

1. 定义损失函数  
        
        
    $$L(w,b) = \sum_{i=1}^{n}(\hat{y}_{i}-(b+wx_{i}))^2$$

3. 利用共轭梯度法更新参数w,b   
$$\frac{\partial L}{\partial w}=\sum2(\hat{y}_{i}-(b+wx_{i}))(-x_{i})$$  
$$\frac{\partial L}{\partial b}=\sum2(\hat{y}_{i}-(b+wx_{i}))$$
4. 正则化  
$$y=b+\sum_{i=1}^{n}{w_{i}x_{i}}$$  
$$L=\sum_{i}^{n}(\hat{y}_{i}-(b+\sum{w_{i}x_{i}}))^2+\lambda\sum(w_{i})^2$$
5. 补充  
* 若数据存在很大偏差：  
. 如果模型对训练数据都无法进行很好的拟合：模型欠拟合(Underfitting)  
. 如果模型对训练数据拟合的很好，但是在测试集上表现得很差：模型过拟合(Overfitting)  
* 对以上问题的解决方法：  
. 针对输入数据增加更多的特征  
. 设计更加复杂的模型  
* 局部加权线性回归(LWLR)  
. 针对线性回归中遇到的欠拟合问题，因为求的是具有最小均方误差的无偏估计，所以如果模型不能得到较好的预测结果，允许引入一些偏差，从而让降低预测的均方误差。  
$$\hat{w}=(X^{T}WX)^{-1}X^{T}Wy$$
其中$X$是一个系数阵，$W$是一个对角阵，对系数赋予不同的权重。 
## 共轭梯度法(Gradient Descent)补充  
也是通过计算损失函数的梯度，更新参数  
* Adagrad  

$$w_{1}=w_{0}-\frac{\eta_{0}}{\sigma_{0}}g_{0}\ \ \ \ \sigma_{0}=\sqrt{(g_{0})^2}$$  
$$w_{2}=w_{1}-\frac{\eta_{1}}{\sigma_{1}}g_{1}\ \ \ \ \sigma_{0}=\sqrt{\frac{1}{2}(g_{0}^2+g_{1}^{2})}$$  
$${\ldots}$$
$$w_{t+1}=w_{t}-\frac{\eta_{t}}{\sigma_{t}}g_{t}\ \ \ \ \sigma_{0}=\sqrt{\frac{1}{t+1}\sum_{i=0}^{t}{(g_{i})^2}}$$  

  
## 分类(Classification)  
* 可以结合不同的概率模型运算  
* 定义损失函数(最大似然估计)
  $$L(\mu,\Sigma)=\int_{u,\Sigma}(x_{1})\int_{u,\Sigma}(x_{2})\ldots\int_{u,\Sigma}(x_{n})$$  
  $$\mu^{*},\Sigma^{*}=arg\ \underbrace{max}_{\mu,\Sigma}L(\mu,\Sigma)$$  
  $$\mu^{*}=\frac{1}{n}\sum_{i=1}^{n}x_{i}$$  
  $$\Sigma^{*}=\frac{1}{n}\sum_{i=1}^{n}(x_{i}-\mu^*)(x_{i}-\Sigma)^{T}$$
$$P(c_{1}|x)=\frac{P(x|c_{1})P(c_{1})}{P(x|c_{1})P(c_{1})+P(x|c_{1})P(c_{2})}$$  
其中，针对$P(c_{1}|x)$的计算，可以采用不同的概率模型，比如朴素贝叶斯。  
* 逻辑回归(Logistic Regression)  
  $Step\ 1.$ 定义分类的目标函数 
  $$f_{w,b}(x)=\sigma(\sum_{i}^{n}w_{i}x_{i}+b)$$  
  $\sigma$函数输出0或1。  
  $Step\ 2.$ 对数据集$(x_{n},y_{n})$进行训练，定义损失函数：  
  $$L(f)=\sum_{n}C(f(x_{n},\hat{y}_{n}))$$  
  定义交叉熵(Cross entropy):  
  $$C(f(x_{n},\hat{y}_{n})=-[\hat{y}_{n}lnf(x_{n})+(1-\hat{y}_{n})ln(1-f(x_{n}))]$$  
  $Step\ 3.$寻找到最好的函数  
  令$z=\sum_{i}w_{i}x_{i}$, 对$lnL(w,b)$求偏导，最后可以得到:  
  $$\frac{\partial{ln(1-\sigma(z))}}{\partial{z}}=\sigma{z}$$  
  $$\frac{lnL(w,b)}{\partial{w_{i}}}=\sum_{n}(\hat{y}_{n}-f_{w,b}(x_{n}))x_{i}^{n}$$  
  所以可以得到：  
  $$w_{i}\leftarrow{w_{i}-\eta\sum_{n}[-(\hat{y}_{n}-f_{w,b}(x))x_{i}^{n}]}$$  
* 关于逻辑回归损失函数补充  
  如果和回归模型一样，不采用交叉熵，会出现以下问题：  
  假设损失函数模型为：  
  $$L(f)=\frac{1}{2}\sum_{n}(f_{w,b}(x_{n})-\hat{y}_{n})^{2}$$  
  对$L(f)进行求导$:  
  $$\frac{\partial(f_{w,b}(x)-\hat{y})^2}{\partial{w_{i}}}=2(f_{w,b}(x)-\hat{y})f_{w,b}(x)(1-f_{w,b}(x))x_{i}$$  
  假设$\hat{y}_{n}=1$，若$f_{w,b}(x_{n})=1$，$\frac{\partial{L}}{\partial{w_{i}}}=0$，模型的预测结果符合真实结果；若$f_{w,b}(x_{n})=0$，同样可以得到$\frac{\partial{L}}{\partial{w_{i}}}=0$。  
  综上，不可以用最小二乘法作为该二分类问题的损失函数。
