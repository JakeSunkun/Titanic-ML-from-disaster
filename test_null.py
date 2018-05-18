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

import xgboost as xgb
from xgboost import XGBClassifier
# ————————————————1.1 简介————————————————
# kaggle竞赛数据：Titanic: Machine Learning from Disaster

sns.set(style='white', context='notebook', palette='deep')
# ————————————————2.1 导入数据————————————————
# load data
train = pd.read_csv("D:/workstation/kaggle_list/Titanic-ML-from-disaster/train.csv")
test = pd.read_csv("D:/workstation/kaggle_list/Titanic-ML-from-disaster/test.csv")
IDtest = test["PassengerId"]


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

train = train.drop(Outliers_to_drop, axis=0).reset_index(drop=True)

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

# ————————————————3 Feature Analysis:特征分析——————————————————
# ————————————————3.1 数值分析————————————————
# 数值之间的关联度
plt.figure()
g_all = sns.heatmap(train[["Survived", "SibSp", "Parch", "Age", "Fare"]].corr(),
                    fmt=".2f", cmap="coolwarm")
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

g_fare = sns.distplot(train["Fare"], color="m", label="Skewness: %.2f" % (dataset["Fare"].skew()))
g_fare = g_fare.legend(loc="best")
plt.show()
# 平滑处理
train["Fare"] = train["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
g_fare_log = sns.distplot(train["Fare"], color="m", label="Skewness: %.2f" % (dataset["Fare"].skew()))
g_fare_log = g_fare_log.legend(loc="best")
plt.show()

# ————————————————3.2 类别值分析————————————————
# sex性别分析
g_sex = sns.barplot(x="Sex", y="Survived", data=train)
g_sex = g_sex.set_ylabel("Sruvival Probability")
plt.show()

info_sex = train[["Sex", "Survived"]].groupby('Sex').mean()
# print(info_sex)

# Pclass:乘客舱等级划分
g_pclass = sns.factorplot(x="Pclass", y="Survived", data=train, kind="bar", size=6, palette="muted")
g_pclass = g_pclass.set_ylabels("survival probability")
plt.show()

# 不同乘客舱中性别得生存比例
g_pclass_sex = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train,
                   size=6, kind="bar", palette="muted")
g_pclass_sex = g_pclass_sex.set_ylabels("survival probability")
plt.show()

# 登船港口数据分析
# 填充空缺数据为登船人数最多的港口
info_embarked = dataset["Embarked"].isnull().sum()
dataset["Embarked"] = dataset["Embarked"].fillna("S")

g_embarked = sns.factorplot(x="Embarked", y="Survived", data=train, size=6, kind="bar", palette="muted")
g_embarked = g_embarked.set_ylabels("surveved probability")
plt.show()
# col:不同登船港口分别划分，每个条目里面是Pclass船舱等级的划分
g_embarked_count = sns.factorplot("Pclass", col="Embarked", data=train, size=6, kind="count", palette="muted")
g_embarked_count = g_embarked_count.set_ylabels("Count")
plt.show()

# ————————————————4 Filling missing Valuest:填补空缺值————————————————
# Age：年龄和各个特征之间的数值分析
g = sns.factorplot(y="Age", x="Sex", data=dataset, kind="box")
plt.show()
g = sns.factorplot(y="Age", x="Sex", hue="Pclass", data=dataset, kind="box")
plt.show()
g = sns.factorplot(y="Age", x="Parch", data=dataset, kind="box")
plt.show()
g = sns.factorplot(y="Age", x="SibSp", data=dataset, kind="box")
plt.show()

# 将性别转换为0或者1，male：0；female：1
dataset["Sex"] = dataset["Sex"].map({"male":0, "female":1})
g = sns.heatmap(dataset[["Age", "Sex", "SibSp", "Parch", "Pclass"]].corr(), cmap="BrBG", annot=True)
plt.show()

# 填充缺失的年龄信息
# NaN age的行的列表
index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)

for i in index_NaN_age:
    age_med = dataset["Age"].median()
    age_pred = dataset["Age"][((dataset["SibSp"] == dataset.iloc[i]["SibSp"])
                               & (dataset["Parch"] == dataset.iloc[i]["Parch"])
                               & (dataset["Pclass"] == dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred):
        dataset["Age"].iloc[i] = age_pred
    else:
        dataset['Age'].iloc[i] = age_med

dataset = dataset.where(pd.notna(dataset), dataset.mean(), axis='columns')

# 重新绘Survived和Age的关系图，制箱式和琴式图
g = sns.factorplot(x="Survived", y="Age", data=train, kind="box")
plt.show()
g = sns.factorplot(x="Survived", y="Age", data=train, kind="violin" )
plt.show()

# ————————————————5 Feature engineering: 特征工程————————————————
# 5.1 Name/Title
info_name = dataset["Name"].head()
# print(info_name)

# 从Name中获取Title
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(dataset_title)
info_title = dataset["Title"].head()
# print(info_title)

g = sns.countplot(x="Title", data=dataset)
g = plt.setp(g.get_xticklabels(), rotation=45)
plt.show()

# 统计改进
dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don',
                                             'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
dataset["Title"] = dataset["Title"].astype(int)

g = sns.countplot(dataset["Title"])
g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])
plt.show()

# Title和Survived之间的关系图
g = sns.factorplot(x="Title", y="Survived", data=dataset, kind="bar")
g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])
g = g.set_ylabels("sruvival probability")
plt.show()

# 去掉无关紧要的Name
dataset.drop(labels = ["Name"], axis = 1, inplace = True)

# 5.2 Family Size
# 新建family总数
dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1

g = sns.factorplot(x="Fsize", y="Survived", data=dataset)
g = g.set_ylabels("Survival Probability")
plt.show()

# 优化统计展示效果
dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if s == 2 else 0)
dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <=s<= 4 else 0)
dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)

g = sns.factorplot(x="Single", y="Survived", data=dataset, kind="bar")
g = g.set_ylabels("Survived Probability")
plt.show()
g = sns.factorplot(x="SmallF", y="Survived", data=dataset, kind="bar")
g = g.set_ylabels("Survived Probability")
plt.show()
g = sns.factorplot(x="MedF", y="Survived", data=dataset, kind="bar")
g = g.set_ylabels("Survived Probability")
plt.show()
g = sns.factorplot(x="LargeF", y="Survived", data=dataset, kind="bar")
g = g.set_ylabels("Survived Probability")
plt.show()

# print(dataset.head())
# 原始数据中的添加准备好的相关数据
dataset = pd.get_dummies(dataset, columns=["Title"])
dataset = pd.get_dummies(dataset, columns=["Embarked"], prefix="Em")
# print(dataset.head())

# 5.3 Cabin
# Cabin基本信息分析
# print(dataset["Cabin"].head())
# print(dataset["Cabin"].describe())
# print(dataset["Cabin"].isnull().sum())
# print(dataset["Cabin"][dataset["Cabin"].notnull()].head())

# print(dataset["Cabin"])

# dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])
# g = sns.countplot(dataset["Cabin"],order=['A','B','C','D','E','F','G','T','X'])
# plt.show()
# g = sns.factorplot(y="Survived",x="Cabin",data=dataset,kind="bar",order=['A','B','C','D','E','F','G','T','X'])
# g = g.set_ylabels("Survival Probability")
# plt.show()
#
# # 将Cabin中的数据加入dataset中
# dataset = pd.get_dummies(dataset, columns=["Cabin"],prefix="Cabin")
dataset.drop(labels=["Cabin"], axis=1, inplace=True)

# 5.4 Ticket
# print(dataset["Ticket"].head())

# 提取Ticket中的prefix并替换列中数据
Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit():
        Ticket.append(i.replace(".", "").replace("/", "").strip().split(' ')[0])
    else:
        Ticket.append("X")

dataset["Ticket"] = Ticket

# print(dataset["Ticket"].head())

# 相应数据添加进待训练数据集
dataset = pd.get_dummies(dataset, columns=["Ticket"], prefix="T")
# dataset.drop(labels = ["Ticket"], axis = 1, inplace = True)

# 为Pclass创建catgorical values
dataset["Pclass"] = dataset["Pclass"].astype("category")
dataset = pd.get_dummies(dataset, columns=["Pclass"], prefix="Pc")
dataset.drop(labels=["PassengerId"], axis=1, inplace=True)
head5 = dataset.head(5)
# print(head5)

# # 针对Fare进行处理
# dataset['FareBand'] = pd.qcut(dataset['Fare'], 4)
# g = sns.factorplot(x="FareBand", y="Survived", data=dataset, kind="bar")
# g = g.set_ylabels("Survived Probability")
# plt.show()

# g = sns.factorplot(x="Fare4", y="Survived", data=dataset, kind="bar")
# g = g.set_ylabels("Survived Probability")
# plt.show()
# g = sns.factorplot(x="Fare3", y="Survived", data=dataset, kind="bar")
# g = g.set_ylabels("Survived Probability")
# plt.show()
# g = sns.factorplot(x="Fare2", y="Survived", data=dataset, kind="bar")
# g = g.set_ylabels("Survived Probability")
# plt.show()
# g = sns.factorplot(x="Fare1", y="Survived", data=dataset, kind="bar")
# g = g.set_ylabels("Survived Probability")
# plt.show()
# Fare_level = []
# for i in dataset['Fare']:
#     if i <= 7.91:
#         Fare_level.append(1)
#     elif 7.91<i<=14.454:
#         Fare_level.append(2)
#     elif 14.454<i<=31:
#         Fare_level.append(3)
#     else:
#         Fare_level.append(4)
# dataset['Fare_level'] = Fare_level


# ————————————————6 modeling：建模————————————————
# 将dataset分为train和test
train = dataset[:train_len]
test = dataset[train_len:]
test.drop(labels=["Survived"], axis=1, inplace=True)

# train中将features和labels分开
train["Survived"] = train["Survived"].astype(int)
Y_train = train["Survived"]
X_train = train.drop(labels=["Survived"], axis=1)

# 6.1 simplt modeling
# 6.11 cross validate models
kfold = StratifiedKFold(n_splits=10)

# 测试不同算法
# random_state是随机数生成器产生的结果
# 保证重新生成的随机数是相同的，只要内部数字一样
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),
                                      random_state=random_state, learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state=random_state))
classifiers.append(LinearDiscriminantAnalysis())
classifiers.append(XGBClassifier())


cv_results = []
# scoring：用户决定输出评分的格式；cv：交叉验证；n_jobs:使用的CPU核心数量

for classifiers in classifiers:
    cv_results.append(cross_val_score(classifiers, X_train, y=Y_train,
                                      scoring="accuracy", cv=kfold, n_jobs=-1))
cv_means = []
cv_std = []
for cv_results in cv_results:
    cv_means.append(cv_results.mean())  # 计算均值
    cv_std.append(cv_results.std())     # 标准差(Standard Deviation)描述各数据偏离平均数的距离（离均差）的平均数

cv_res = pd.DataFrame({"CrossValMeans": cv_means, "CrossValerrors": cv_std,
                       "Algorithm": ["SVC", "DecisionTree", "AdaBoost", "RandomForest", "ExtraTrees",
                                     "GradientBoosting", "MultipleLayerPerceptron", "KNeighboors", "LogisticRegression",
                                     "LinearDiscriminantAnalysis", "XGBClassifier"]})

g = sns.barplot("CrossValMeans", "Algorithm", data=cv_res, palette="Set3",
                orient="h", **{'xerr': cv_std})
g.set_xlabel("Mean Accruacy")
g = g.set_title("Cross validation scores")
plt.show()

