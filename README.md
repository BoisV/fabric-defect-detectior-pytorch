# fabric-defect-detectior-pytorch
## 环境需求
pytorch 1.17 pillow 8.0.1


文件目录结构如下
```python
├─.gitignore
├─main_train.py
├─README.md
├─lib
|  ├─__init__.py
|  ├─utils
|  |   ├─dataset.py
|  |   ├─data_split.py
|  |   ├─__init__.py
|  ├─models
|  |   ├─model.py
|  |   ├─__init__.py
├─docs
├─data
|  ├─fabric_data
|  └─DIY_fabric_data
```

## 运行步骤
1. 安装所需包
2. 将下载下来的fabric_data文件夹放到/data/目录下，执行data_split.py文件
3. 运行main_train.py，开始训练
   
注意：有些图片打不开，执行要删除/data/DIY_fabric_data/list.txt中图片对应行即可。


