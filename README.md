# rsearch
##安装环境和方式

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
cmake
```



安装如下：

```
mkdir build
```

建立build文件夹

```
cd build
cmake ..
make
```

## 程序架构

~~~
|-rsearch_type(用到的数据类型)
|-rsearch_def(所有定义的类)
|-gallery(存储数据的类)
|  |-cpu_base_gallery
|  |-pqivf_gallery
|-probe(解决k近邻的方法的类)
|  |-cpu_base_probe
|  |-pqivf_probe(乘积量化 + 粗度量器)
|-matrix(矩阵乘法+topk选择)
|  |-base_matrix_mul
|  |-rapid_matrix_mul
|-utils
|  |-cluster(聚类)
|  |-helpers(一些类型转字符串)
|  |-avx2_asm(利用avx2和sse3指令集加速计算)
|  |-utils(一些通用函数)
|  |-ThreadPool(线程池,多线程运行)
|-pipe(用于连接不同方法的查询，还没完成)
~~~

