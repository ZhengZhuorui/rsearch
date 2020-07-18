# rsearch
##安装环境和方式

### C++安装

需要安装下列库，将include和lib放置于thirdparty文件夹下：

```
faiss
gtest
```

需要环境如下：

```
ubuntu 16.04
CPU: Intel 
Flags:sse3 avx2
cmake >= 2.8.12
mkl blas(安装faiss需要)
```

安装如下：

```
mkdir build
```

建立build文件夹

```
cd build
cmake ..
make( && make install)
```

### Python安装

需要安装下列软件

```
pcre
swig >= 3.0
python == 3.5.2
```

需要安装下列python库

```
numpy
```

需要对makefile.inc配置环境参数（懒得写configure了）

首先需要执行**C++安装**，然后进入python文件夹，进行编译

```
cd python
make
```

然后得到rsearch文件夹

样例见python文件夹下的demo.py

## 文件结构

~~~
|-include(头文件)
|  |-rsearch_type(用到的数据类型)
|  |-rsearch_def(所有定义的类)
|  |-gallery(存储数据的类)
|  |  |-cpu_base_gallery
|  |  |-pqivf_gallery
|  |-probe(解决k近邻的方法的类)
|  |  |-cpu_base_probe
|  |  |-pqivf_probe(乘积量化 + 粗度量器)
|  |-matrix(矩阵乘法+topk选择)
|  |  |-base_matrix_mul
|  |  |-rapid_matrix_mul
|  |-other(主要是增加时间空间索引)
|  |  |-simple_gallery
|  |  |-simple_index
|  |-utils
|  |  |-cluster(聚类)
|  |  |-helpers(一些类型转字符串)
|  |  |-avx2_asm(利用avx2和sse3指令集加速计算)
|  |  |-utils(一些通用函数)
|  |  |-ThreadPool(线程池,多线程运行)
|  |-rsearch
|-src(代码实现)
|-python(Python接口)
|-frontend(前端+Django接收前端的数据进行处理)
|-CMakeLists.txt(cmake配置文件)
|-README.md(程序说明)
|-makefile.inc(配置文件)
|-configure.h.in(c++配置头文件)
|-env.sh(初始环境设置)
~~~

test下包括unit_test和benchmark还有demo，不过现在还在调试尚未完成