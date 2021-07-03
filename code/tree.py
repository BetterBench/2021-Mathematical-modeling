
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree,linear_model
import graphviz 
import pydotplus
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=cmap(idx),marker=markers[idx], label=cl)
    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1, marker='o', s=55, label='test set')

from matplotlib.colors import ListedColormap
# 定义函数，用于绘制决策边界。 
def plot_decision_boundary(model, X, y): 
    color = ["r", "g", "b"] 
    marker = ["o", "v", "x"] 
    class_label = np.unique(y) 
    cmap = ListedColormap(color[: len(class_label)]) 
    x1_min, x2_min = np.min(X[:,0:2], axis=0) 
    x1_max, x2_max = np.max(X[:,1:3], axis=0) 
    x1 = np.arange(x1_min - 1, x1_max + 1, 0.02) 
    x2 = np.arange(x2_min - 1, x2_max + 1, 0.02) 
    X1, X2 = np.meshgrid(x1, x2,x3) 
    Z = model.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape) 
# 绘制使用颜色填充的等高线。 
    plt.contourf(X1, X2, Z, cmap=cmap, alpha=0.5) 
    for i, class_ in enumerate(class_label): 
        plt.scatter(x=X[y == class_, 0], y=X[y == class_, 1], 
                c=cmap.colors[i], label=class_, marker=marker[i])
    plt.legend()
    plt.show()

def plot_every_DRAG_fee(data):
    category={}
    cate_box = pd.DataFrame()
    datalen = len(data)
    for i in range(0,datalen):
        adrgid = data['adrgid'][i]
        data_fee = data['fee'][i]
        if adrgid in category.keys():
            templist = list(category[adrgid])
            templist.append(data_fee)
            category[adrgid] =list(set(templist))
        else:
            category[adrgid] = [data_fee]

    # for  k in category.keys():
    #     maxn =max(category[k])
    #     minn =min(category[k])
    #     low = min() 
    #     mid = 
    #     high =
    #     inteval_n = [mink,]
    plt.figure()
    x = list(range(len(category['BU1'])))
    y = list(sorted(category['BU1']))
    plt.xlabel('BU1')
    plt.ylabel('Medical fees')
    plt.subplot(221) 
    plt.scatter(x, y, alpha=0.9)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）

    x2 = list(range(len(category['ER1'])))
    y2 = list(sorted(category['ER1']))
    plt.xlabel('ER1')
    plt.ylabel('Medical fees')
    plt.subplot(222)

    plt.scatter(x2, y2, alpha=0.9)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
    x2 = list(range(len(category['GR1'])))
    y2 = list(sorted(category['GR1']))
    plt.xlabel('GR1')
    plt.ylabel('Medical fees')
    plt.subplot(223) 
    plt.scatter(x2, y2, alpha=0.9)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）

    x2 = list(range(len(category['QS3'])))
    y2 = list(sorted(category['QS3']))
    plt.xlabel('QS3')
    plt.ylabel('Medical fees')
    plt.subplot(224) 
    plt.scatter(x2, y2, alpha=0.9)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
    plt.show()
    print()

def interval_deal(data):
    category={}
    cate_box = pd.DataFrame()
    datalen = len(data)
    for i in range(0,datalen):
        adrgid = data['adrgid'][i]
        data_fee = data['fee'][i]
        if adrgid in category.keys():
            templist = list(category[adrgid])
            templist.append(data_fee)
            category[adrgid] =list(set(templist))
        else:
            category[adrgid] = [data_fee]
    mdict ={}
    inter_fee =[]
    inter_adrg = []
    inter_df = pd.DataFrame()
    low_fee =[]
    mid_fee =[]
    high_fee = []
    for  k in category.keys():
        maxn =max(category[k])
        minn =min(category[k])
        avg = maxn-minn
        low = avg*(3/10)
        mid = avg*(5/10)
        high =avg*(2/10)
        inteval_n = [minn,minn+low,minn+low+mid,minn+low+mid+high]
        low_fee.append([minn,minn+low])
        mid_fee.append([minn+low,minn+low+mid])
        high_fee.append([minn+low+mid,minn+low+mid+high])
        inter_adrg.append(k)
        # inter_fee.append(inter_temp)
        mdict[k] = inteval_n
    inter_df['ADRG'] = inter_adrg
    inter_df['低费用区间'] = low_fee
    inter_df['中费用区间'] = mid_fee
    inter_df['高费用区间'] = high_fee
    inter_df.to_csv('ADRG_interval.csv')
    print()
    labelfee = []
    for i in range(0,datalen):
        adrgid = data['adrgid'][i]
        dfee = float(data['fee'][i])
        minn = float(mdict[adrgid][0])
        lowd = float(mdict[adrgid][1])
        midd= float(mdict[adrgid][2])
        highd = float(mdict[adrgid][3])
        if dfee>=minn and lowd>=dfee:
            labelfee.append(1)
        elif dfee>lowd and midd>=dfee:
            labelfee.append(2)    
        else:
            labelfee.append(3) 
    # print(labelfee)
    return labelfee
    # print()
def plot_confusion_matrix(cm, classes,savename, title='Confusion Matrix'):

    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    # plt.savefig(savename, format='png')
    plt.show()
def decision_tree(X,Y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0) 
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    # 模型预测
    y_pred = clf.predict(X_test)
    # 计算准确率
    score = accuracy_score(y_test, y_pred)
    print(score)
    classes = [1,2,3]
    # 获取混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm,classes, 'confusion_matrix.png', title='confusion matrix')
    print()
    fn=['是否手术','ADRG','并发症']
    cn=['低','中', '高']
    # fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
    # tree.plot_tree(clf,
    #             feature_names = fn, 
    #             class_names=cn,
    #             filled = True)
    # 决策树可视化
    # fig.savefig('imagename.png')
    # dot_data = tree.export_graphviz(clf,
    #             feature_names = fn, 
    #             class_names=cn,
    #             out_file=None)
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.write_pdf('iris.pdf')
    print()
def scatter_plot(data):
    plt.scatter(data['surgery'],data['surgery'],  # 按照经纬度显示
    s = data['complication'],  # 按照单价显示大小
    c = data['参考总价'],  # 按照总价显示颜色
    alpha = 0.4, cmap = 'Reds')  
    plt.grid()
if __name__ =="__main__":
    # 读取清洗后的数据
    train_data_file = './clear_data.csv'
    t_data = pd.read_csv(train_data_file)#, names=['id', 'sex','born','intime','outtime','maindiag','elsediag','surgery','fee','days','drgsid','drgs','adrgid','adrg','highfee'])
    # 数据标准化
    sc = StandardScaler()
    data = pd.read_csv('main_label3.csv',index_col=None)
    X_i = data[['surgery','adrgid','complication']]
    sc.fit(X_i)
    X = sc.transform(X_i)
    Y = data['label']
    decision_tree(X,Y)

    # generate_csv(t_data)
    # interval_deal(t_data)
    # interval_static(t_data)
    # bayes_cla(X,Y)
    # logistic_model(X,Y)
    # scatter_point(X,Y)