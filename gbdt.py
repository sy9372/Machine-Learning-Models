# 本来是在jupyter notebook上测试的，可以在线测试，现转移到pycharm中，可以测出数据的均方误差和R方

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error  # 平均相对误差绝对值
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import r2_score
from joblib import dump

# 1.导入数据
data = pd.read_csv('./一系列矿浆(泵A流量）模型_3std去异_均值滤波.csv')

# 2.数据处理及调用gbdt模型
data1 = data.iloc[:, :-1]  # iloc函数：通过行号来取行数据（贴片：最后一列提取出来）
y = data.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(data1, y, test_size=0.2)
# 在sklearn包中调用梯度提升决策树的模型
gbdt = GradientBoostingRegressor(
    # boosting参数
    init=None,
    n_estimators=100,   # 弱学习器最大迭代次数
    learning_rate=0.1,  # 学习率
    subsample=0.8,   # 子采样，不放回抽样，默认为1即不使用子采样
    loss="squared_error",  # lr/squared_error
    alpha=0.7,  # 噪点多，可以降低这个值
    # 分割参数
    max_features=None,   # 默认是None,意味着划分时考虑所有的特征数（样本特征数不多，比如小于50，选默认None）
    criterion='friedman_mse',
    # 分割停止参数
    min_samples_split=2, 
    min_impurity_decrease=0,
    max_depth=5,   # 可以修改，数据少或特征少时可以不管这个值；数据多，常用取值10-100
    max_leaf_nodes=None,
    # 剪枝参数，保持默认值
    min_samples_leaf=1,
    warm_start=False,
    random_state=10
)
gbdt.fit(x_train, y_train)   # fit函数
y_pred = gbdt.predict(x_test)  

# 3.画图（pycharm上显示不了图，可以在jupyter notebook上查看预测结果对比图）
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
f, ax = plt.subplots(1)
f.set_figheight(6)
f.set_figwidth(10)
ax.grid(True)
ax.plot(range(len(y_test)),y_test,label= "磷酸根浓度测量值")
ax.plot(range(len(y_pred)),y_pred,label= "磷酸根浓度预测值")
ax.legend()
ax.set_xlabel('预测样本') 
ax.set_ylabel('磷酸根浓度')

# 4.预测精度分析
max_relative_error = np.max(np.abs(y_test - y_pred) / y_test, axis=0) # 最大相对误差
print('最大相对误差：', max_relative_error)
print('平均相对误差：', mean_absolute_percentage_error(y_test, y_pred))
print('均方误差：', mean_squared_error(y_test, y_pred))
print('r2',r2_score(y_test,y_pred))

# 5.绘制相对误差曲线
f, ax = plt.subplots(1)
f.set_figheight(6)
f.set_figwidth(10)
ax.grid(True)
ax.plot(range(len(abs(y_test - y_pred) / y_test)),abs(y_test - y_pred) / y_test,label = "相对误差（绝对值）")
ax.legend()
ax.set_xlabel('预测样本') 
ax.set_ylabel('最大相对误差')
plt.show()

# 6.模型生成
# num = 3

# dump(gbdt, "gbdt_trained" + str(num)+'.joblib')
# '算法名+输入个数_trained'
# 'gbdt3_trained'
# ''