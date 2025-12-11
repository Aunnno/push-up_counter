# push-up_counter  
> *A project that provide a way to count push-ups in real time.* 
---
## 环境配置 
### 运行环境  
以下是开发者自己的运行环境：  
* GPU:`Nvidia Geforece RTX 3070 Laptop` 
* GPU RAM:`8GB` 
* RAM:`16GB` 
* CUDA version:`12.1` 
* python version:`3.10.11` 
 --- 
* CPU:`i5-10500` 
* RAM:`8GB` 
* python version:`3.12.0` 
上面的是我的个人电脑,下面的是学校电脑。*由于使用的是yolo11x-pose，所以我强烈推荐你的电脑配置尽量高一点，因为yolo官方模型除了x模型之外其他模型效果都不好*~~（当然你愿意的话可以去网上下载别的模型）~~ 
### Python 依赖 
* `ultralytics`
* `scikit-learn`
其实你也可以直接 
```shell
pip install -r requirements.txt -y
```   
如果你有N卡的话可以下个CUDA，然后把pytorch版本换成CUDA版，这样可以快很多很多 

---

## 代码介绍  
### 原理介绍  
使用了yolo11x-pose实现了姿态检测，yolo-pose为现成的姿态检测模型，这里不作赘述，详见其官方文档<https://docs.ultralytics.com/zh/tasks/pose/> 
使用了随机森林实现了状态分类，因为随机森林是机器学习，性能要求和数据集大小要求都很低，以及这种特征明显的分类任务没必要用神经网络  
### 整体思路 
分别封装了训练器，识别器，计数器以及标注器为单独的模块供调用  
我们提取了做俯卧撑时变化最大的几个角度作为特征进行训练，数据集大小为325条  
### 参数 
* Train 
  * `n_estimators=100`  
  * `max_depth=15` 
* Count
  * `confidence=0.6`
### 其他
自己翻代码去我懒得写了  

---

## 使用方法 

运行`annotate.py`启动标注器，记得改下路径 
运行`realtime_count.py`启动计数器，注意，`count.py`已被废弃，不要运行那个 
部分路径未被暴露，请自行更改源代码  

--- 

## 鸣谢
小陈，帮我们借了个相机采集了训练数据和测试数据  
小熊，做了15个俯卧撑 
大为，给我喝了两口咖啡吃了一口肥牛饭里的肥牛 