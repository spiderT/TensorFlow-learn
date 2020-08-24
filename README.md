# TensorFlow-learn

## 1. 安装环境

### 1.1. 安装python开发环境

安装python：  
brew install python@2  

安装pip：  
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py  
sudo python get-pip.py  
sudo pip install -U virtualenv  

### 1.2. 创建python虚拟环境

virtualenv --system-site-packages -p python2.7 ./venv  
source ./venv/bin/active  
pip install --upgrade pip  
pip list  

之后可以使用以下命令退出 virtualenv：  
deactivate  

### 1.3. 安装TensorFlow包

pip install --upgrade tensorflow

## 2. TensorFlow基础概念解析

### 2.1. TensorFlow模块与架构介绍

TensorFlow 模块与 APIs  
![模块](images/ts1.png)  

TensorFlow 架构  
![架构](images/ts2.png)  

### 2.2. TensorFlow数据流图介绍

TensorFlow数据流图是一种声明式编程范式  

![数据流图](images/ts3.png)  
![数据流图](images/ts4.png)

TensorFlow数据流图优势：  

- 并行计算快
- 分布式计算快(CPUs, GPUs TPUs)
- 预编译优化(XLA)
- 可移植性好(Language-independent representation)

### 2.3. 张量Tensor

在数学里，张量是一种几何实体，广义上表示任意形式的“数据”，张量可以理解为0阶标量，1阶向量和2阶矩阵在高维空间上的推广，张量的阶描述它表示数据的最大维度。

![张量Tensor](images/ts5.png)

tf.Tensor是TensorFlow.js中的最重要的数据单元，它是一个形状为一维或多维数组组成的数值的集合。tf.Tensor和多维数组其实非常的相似。  

一个tf.Tensor还包含如下属性:  

rank: 张量的维度  
shape: 每个维度的数据大小  
dtype: 张量中的数据类型  

可以用tf.tensor()方法将一个数组(array)创建为一个tf.Tensor：  

```js
// 从一个多维数组创建一个rank-2的张量矩阵
const a = tf.tensor([[1, 2], [3, 4]]);
console.log('shape:', a.shape);
a.print();
// 或者您可以用一个一维数组并指定特定的形状来创建一个张量
const shape = [2, 2];
const b = tf.tensor([1, 2, 3, 4], shape);
console.log('shape:', b.shape);
b.print();
```

在默认的情况下，tf.Tensor的数据类型也就是 dtype为32位浮点型(float32)。当然tf.Tensor也可以被创建为以下数据类型：布尔(bool), 32位整型(int32), 64位复数(complex64), 和字符串(string)：  

```js
const a = tf.tensor([[1, 2], [3, 4]], [2, 2], 'int32');
console.log('shape:', a.shape);
console.log('dtype', a.dtype);
a.print();
```

#### 修改张量的形状  

tf.Tensor中的元素数量是这个张量的形状的乘积(例如一个形状为[2,3]的张量所含有的元素个数为2*3=6个)。所以说在大部分时候不同形状的张量的大小却是相同的,那么将一个tf.Tensor改变形状(reshape)成为另外一个形状通常是有用且有效的。上述操作可以用reshape() 方法实现:  

```js
const a = tf.tensor([[1, 2], [3, 4]]);
console.log('a shape:', a.shape);
a.print();

const b = a.reshape([4, 1]);
console.log('b shape:', b.shape);
b.print();
```

#### 获取张量的值

可以使用Tensor.array() or Tensor.data()这两个方法:  

```js
 const a = tf.tensor([[1, 2], [3, 4]]);
 //返回多维数组的值
 a.array().then(array => console.log(array));
 // 返回张量所包含的所有值的一维数组
 a.data().then(data => console.log(data));
```

### 2.4. 变量Variable

TensorFlow变量(Variable)的主要作用是维护特定节点的状态，如深度学习或机器学习的模型参数。  
tf.Variable方法是操作，返回值是变量（特殊张量）。  

通过tf.Variable方法创建的变量与张量一样，可以作为操作的输入和输出。不同在于：  

- 张量的生命周期通常随依赖的计算完成而结束，内存也随即释放。
- 变量则常驻内存，在每一步的训练时不断更新其值，以实现模型参数的更新。  

```js
const x = tf.variable(tf.tensor([1, 2, 3]));
x.assign(tf.tensor([4, 5, 6]));

x.print();
```

TensorFlow变量使用流程

![变量](images/ts6.png)

### 2.5. 操作Operation

### 2.6. 会话Session

### 2.7. 优化器Optimizer






