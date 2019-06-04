[点这里查看ipyb文件](https://nbviewer.jupyter.org/github/Jweeeeee/ML/blob/master/%E9%9A%8F%E6%9C%BA%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B%E6%A8%A1%E6%8B%9F.ipynb)

[随机微分方程（SDE）的蒙特卡洛模拟（Python实现）](https://www.jianshu.com/p/74b956f6eb63)

这学期上了**随机过程**这门课，为完成老师布置的作业：模拟常微分方程(ODE)和随机微分方程(SDE)的图像![ODE](https://upload-images.jianshu.io/upload_images/17813773-0301047e53bff514.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)![SDE](https://upload-images.jianshu.io/upload_images/17813773-b2b6e33cedf28e93.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


看了一些文章，并在周二课上受到一些启发，在这里趁热总结一下用**Python**模拟常微分方程/随机微分方程的方法，事实上不管用什么语言实现都可以，思路是完全一样的。其中，

学习了大神[dalalaa](https://www.jianshu.com/u/32db699162d4)的[Python数学建模极简入门（三）简单动力系统](https://www.jianshu.com/p/e5bb573f9d05)

数学方面主要是看教材[《随机过程 方兆本》](https://max.book118.com/html/2019/0309/8043000015002012.shtm)


阅读了这篇文章[关于布朗运动两类积分的比较](https://wenku.baidu.com/view/5eda1c35a58da0116d1749e1.html)
### 一.数学基础

![5.3.2 二阶矩过程的微分](https://upload-images.jianshu.io/upload_images/17813773-f3bddc28b471e4e9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![5.4.1 伊藤微分公式](https://upload-images.jianshu.io/upload_images/17813773-846b3f4236e9cc7f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们可以得到$dW(t)=W(t+{\Delta}t)-W(t)$~$N(0,dt)$


### 一.离散化

$df(t)=af(t)dt$可以离散化为$f(k+1)=f(k)+af(k){\Delta}t$

$dS(t)={\mu}S(t)dt+{\sigma}S(t)dBt$可以离散化为$dS(t_{k+1})={\mu}S(t_{k}){\Delta}t+{\sigma}S(t)dY{_k}$其中$Y{_k}$~$N(0,{\Delta}t)$
对于$dt$也就是${\Delta}t$我们可以取为$\frac{b-a}{n}$，$b-a$是区间长度，$n$是点的个数
对于$dW(t)=W(t+{\Delta}t)-W(t)$我们可以生成一个服从$N(0,dt)$的随机数$dnorm$

### 二.根据离散化的表达式生成（x,y）
这是散点图但是可以看成连续的，因为$dt$可以满足任意精度

```
# 引入需要的包和模块
import numpy as np
from matplotlib import pyplot as plt

# 函数定义部分
# ODE
num = 1000
long = 10
X1 = np.linspace(0,long,num) # x的取值范围，即取0到10范围内1000个点
def Y1(y0=1,a=0.618):
    dt = long/num
    Y = np.zeros((1,num)) # 取与x相同数量的y
    Y[(0,0)] = y0
    for i in range(0,num-1):
        Y[(0,i+1)] = Y[(0,i)] + a*Y[(0,i)]*dt # 根据表达式生成相应的y
    return Y[0]

# SDE,表达式部分需要一些变化
num = 1000
long = 10
X2 = np.linspace(0,long,num)1000个点
def Y2(y0=2,mu=0.618,sigma=3.9):
    dt = long/num
    dYt = np.random.normal(0,dt,(1,num)) # 生成和x数量相同的dYt，Yt~N(0,dt)
    Y = np.zeros((1,num)) 
    Y[(0,0)] = y0
    for i in range(0,num-1):
        Y[(0,i+1)] = Y[(0,i)] + mu*Y[(0,i)]*dt + sigma*Y[(0,i)]*dYt[0][i] 
    return Y[0]
```

### 三.画出图像

```
# 绘制图像部分
def pict(X=X1,Y=Y1()):
    plt.plot(X,Y) # X,Y点的横、纵坐标的集合 ps.这里X,Y是都是numpy.ndarray对象
    plt.show() # 显示图像

# 画图
pict(X=X1,Y=Y1())
pict(X=X2,Y=Y2())
```

![ODE](https://upload-images.jianshu.io/upload_images/17813773-461d82c6e87aff76.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![SDE](https://upload-images.jianshu.io/upload_images/17813773-f4d3f1b36ef38d0b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


如果有更高的绘图需求比如绘制更多的曲线可以通过基本方法实现，可以参考[Python的中文文档](https://docs.python.org/zh-cn/3/)，以及matplotlib和numpy的相关文档或教程

更加完整的代码和结果我放在Github上了：[ODE和SDE模拟](https://github.com/Jweeeeee/SDE)

ipynb文件打不开也可以看 [这个网页](https://nbviewer.jupyter.org/github/Jweeeeee/ML/blob/master/%E9%9A%8F%E6%9C%BA%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B%E6%A8%A1%E6%8B%9F.ipynb)

**参考：**
[《随机过程 方兆本》](https://max.book118.com/html/2019/0309/8043000015002012.shtm)
[关于布朗运动两类积分的比较](https://wenku.baidu.com/view/5eda1c35a58da0116d1749e1.html)
[Python数学建模极简入门（三）简单动力系统](https://www.jianshu.com/p/e5bb573f9d05)
[蒙特卡洛估值几种不同的计算方式(Python)](https://blog.csdn.net/u014281392/article/details/76285280)
