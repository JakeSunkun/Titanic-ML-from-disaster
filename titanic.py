import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
# ————————————————1.1 简介————————————————
# kaggle竞赛数据：Titanic: Machine Learning from Disaster

sns.set(style='white', context='notebook', palette='deep')
# ————————————————2.1 导入数据————————————————
# load data
train = pd.read_csv("D:/workstation/kaggle_list/Titanic-ML-from-disaster/train.csv")
test = pd.read_csv("D:/workstation/kaggle_list/Titanic-ML-from-disaster/test.csv")
# IDtest = test["PassengerID"]


# ————————————————2.2 离群样本检测————————————————
# outlier detection：异常值检测
# df：展示col中的数据？
# n:col的阈值？
def detect_outliers(df, n, features):

    outlier_indices = []
    # iterate over features(colums)
    for col in features:
        # print(col)
        # print(df(col))
        # 计算四分位数
        # 四分位数也称为四分位点，它是将全部数据分成相等的四部分，其中每部分包括25%的数据，处在各分位点的数值就是四分位数
        Q1 = np.percentile(df[col], 25)     # 较小四分位数
        Q3 = np.percentile(df[col], 75)     # 较大四分位数
        IQR = Q3 - Q1
        # print(Q1, Q3, IQR)

        # outlier_step：用于确定内限范围
        outlier_step = 1.5 * IQR
        # 确定异常值的index
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        # 将outlier_list_col中的所有的index进行组合
        outlier_indices.extend(outlier_list_col)
    # 计算拓展后的异常值outlier_indices数目
    outlier_indices = Counter(outlier_indices)
    # k：key；v：value
    # multiple_outliers：离群值，当某一个样本在indices中出现超过两次时记录k值
    # 分析：也就是特征中超过两个值是离群值
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    return multiple_outliers


# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train, 2, ["Age", "SibSp", "Parch", "Fare"])

# ————————————————2.3 组合训练和测试数据集————————————————
# 连接测试集和训练集获得绝对的数据版本
train_len = len(train)
dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

# ————————————————2.4 检测空值和缺失值————————————————
# 填充NaN，方便后续统计NaN个数
dataset = dataset.fillna(np.nan)

# 检测Null值,并计数
info_null_sum = dataset.isnull().sum()

# ————————Info——————————
# info_a = train.info()          # 显示train的概要信息
info_b = train.isnull().sum()  # 统计train中的空缺值
info_c = train.head()          # 显示前五行数据
info_d = train.describe()      # 显示描述信息

# ————————————————3 特征分析——————————————————
# ————————————————3.1 数值分析————————————————
# 数值之间的关联度
plt.figure()
g_all = sns.heatmap(train[["Survived", "SibSp", "Parch", "Age", "Fare"]].corr(),
                        annot=True, fmt=".2f", cmap="coolwarm")
g_all = g_all.set_title("Graph3.1 Correlation matrix between numerical values ")
# g_heatmap.show()
plt.show()

# 探究SibSp特征和survived之间的关系
# plt.figure()
g_sibsp = sns.factorplot(x="SibSp", y="Survived", data=train, kind="bar", size=6, palette="muted")
# g_factorplot.despine(letf=True)
g_sibsp = g_sibsp.set_ylabels("survival probability")
plt.show()

# Parch和survived之间的关系
g_parch = sns.factorplot(x="Parch", y="Survived", data=train, kind="bar", size=6, palette="muted")
# g_parch.despine(letf=True)
g_parch = g_parch.set_ylabels("survival probabitlity")
plt.show()

# Age和survived的关系
g_age = sns.FacetGrid(train, col='Survived')
g_age = g_age.map(sns.distplot, "Age")
plt.show()

# Age曲线分布
g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade = True)
g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax =g, color="Blue", shade= True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])
plt.show()

# Fare船费统计数据
info_fare = dataset["Fare"].isnull().sum()
# print(info_fare)
# 以median进行填充
# 报错：没有找到filena方法
# dataset["Fare"] = dataset["Fare"].filena(dataset["Fare"].median())
# for index, value in dataset["Fare"]:
#     if value==0:
#         dataset["Fare"][index]=0

# print(type(dataset["Fare"]))

g_fare = sns.distplot(train["Fare"], color="m", label="Skewness: %.2f"%(dataset["Fare"].skew()))
g_fare = g_fare.legend(loc="best")
plt.show()
# 平滑处理
train["Fare"] = train["Fare"].map(lambda i:np.log(i) if i > 0 else 0)
g_fare_log = sns.distplot(train["Fare"], color="m", label="Skewness: %.2f"%(dataset["Fare"].skew()))
g_fare_log = g_fare_log.legend(loc="best")
plt.show()




