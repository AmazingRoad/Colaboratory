# GTSRB交通标志识别

## 环境

Ubuntu

Python3+

[pytorch](http://pytorch.org/)

## 文件目录
```
.
├── core
│   ├── GTSRBloader.py
│   └── lenet.py
├── data
├── eval.py
├── feature
├── LICENSE.md
├── README.md
├── show.py
├── test.py
├── tool
│   └── parser.py
└── train.py
```
## 数据

GTSRB

放在data文件夹下，目录级别：

```
data/
├── GTSRB_Final_Training_Images
│   └── GTSRB
│         └── Final_Training
│                  └── Images
```

详细可以参见`data/train.txt`文件

## 训练

建议采用GPU训练

训练集上的识别率：99.93%

```shell
python train.py
```

## 验证

```shell
python eval.py
```

## 提取特征

默认提取`data/feature.jpg`的特征图，按需修改

```shell
python test.py
```

## 查看特征图

```shell
python show.py
```