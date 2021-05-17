## 实体抽取
本代码是基于BERT+BiLSTM+CRF进行实体抽取。本代码实现了模型转pb，采用tensorflow 
serving加载预测。采用了一部分demo数据进行模型训练和测试，后续会更新实体数据集的模型
实验结果。本项目的模型预测采用了最大长度截断分段进行预测，并且是按照最大长度附近的标
点符号截断，模型预测新增去掉韩文的处理。

### 代码结构如下
```
train.py
predict.py
app.py
model.py
eval.py
utils.py
save_file_config.py
rnncell.py
context_preprocessing.py
bert_data_utils.py
bert_data_loader.py
bert_export_pb.py
CoNLLeval.py
```

### demo测试集上的结果
```
acc: 98.04, pre: 79.09, rec: 87.00, f1: 82.86
```