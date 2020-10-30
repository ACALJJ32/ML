# Pytorch  Note

* 引用
  ```python
    import torch
    import torch.nn as nn  
    import torch.nn.functional as F
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    torch.manual_seed(446)
    np.random.seed(446)
  ```
* 张量和Numpy
```python
    # we create tensors in a similar way to numpy nd arrays
    x_numpy = np.array([0.1, 0.2, 0.3])
    x_torch = torch.tensor([0.1, 0.2, 0.3])
    print('x_numpy, x_torch')
    print(x_numpy, x_torch)
    print()

    # to and from numpy, pytorch
    print('to and from numpy and pytorch')
    print(torch.from_numpy(x_numpy), x_torch.numpy())
    print()

    # we can do basic operations like +-*/
    y_numpy = np.array([3,4,5.])
    y_torch = torch.tensor([3,4,5.])
    print("x+y")
    print(x_numpy + y_numpy, x_torch + y_torch)
    print()

    # many functions that are in numpy are also in pytorch
    print("norm")
    print(np.linalg.norm(x_numpy), torch.norm(x_torch))
    print()

    # to apply an operation along a dimension,
    # we use the dim keyword argument instead of axis
    print("mean along the 0th dimension")
    x_numpy = np.array([[1,2],[3,4.]])
    x_torch = torch.tensor([[1,2],[3,4.]])
    print(np.mean(x_numpy, axis=0), torch.mean(x_torch, dim=0))
```
* Tensor View
```python
    # "MNIST"
    N, C, W, H = 10000, 3, 28, 28
    X = torch.randn((N, C, W, H))

    print(X.shape)
    print(X.view(N, C, 784).shape)
    print(X.view(-1, C, 784).shape) #automatically choose the 0th dimension
```
* Computation graphs  
  ```python
    a = torch.tensor(2.0, requires_grad=True) # we set requires_grad=True to let PyTorch know to keep the graph  
    b = torch.tensor(1.0, requires_grad=True)  
    c = a + b  
    d = b + 1  
    e = c * d  
    print('c', c)  
    print('d', d)  
    print('e', e)  
  ```
    c tensor(3., grad_fn=\<AddBackward0>)  
    d tensor(2., grad_fn=\<AddBackward0>)  
    e tensor(6., grad_fn=\<MulBackward0>)  
    \* We can see that PyTorch kept track of the computation graph for us.