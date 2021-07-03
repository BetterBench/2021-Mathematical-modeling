
import numpy as np
import pandas as pd
# import tensorflow as tf
from category_encoders.target_encoder import TargetEncoder
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import datetime

train_data_file = './clear_data.csv'
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题


def muti_logistic():
    data = pd.read_csv('feature_data.csv',index_col=0)
    cols = list(data.columns)
    for i in range(len(cols)-3):  
        data1 = data[cols[:-1]]  
        x = sm.add_constant(data1) #生成自变量  
        y = data['fee'] #生成因变量  
        model = sm.OLS(y, x) #生成模型  
        result = model.fit() #模型拟合  
        pvalues = result.pvalues #得到结果中所有P值  
        pvalues.drop('const',inplace=True) #把const取得  
        pmax = max(pvalues) #选出最大的P值  
        if pmax>0.05:  
            ind = pvalues.idxmax() #找出最大P值的index  
            cols.remove(ind) #把这个index从cols中删除  
        else:  
            result.summary()
            print()
# 汇总是否手术、ADRG编码、并发症发咋程度、
def decison_tree_csv(data):
    g_data = pd.DataFrame()
    g_data['surgery'] = surgery_code(data)
    g_data['label'] = label_code(data)
    encoder=TargetEncoder(cols='adrgid') 
    g_data['complication'] = complicate_code(data)
    # 类别特征采用目标编码
    mglist = encoder.fit_transform(data['adrgid'],g_data['label'])
    g_data['adrgid'] = mglist

    g_data.to_csv('main_label3.csv')
    print()
# 汇总年龄、性别、住院天数三个特征和费用成一个CSV文件，为分析亚群特征做特征工程
def generate_csv(data):
    g_data = pd.DataFrame()
    g_data['age'] = age_code(data)
    
    g_data['gender'] = gender_code(data)
    g_data['days'] = data['days']
    g_data['complication'] = complicate_code(data)
    
    g_data['fee'] = data['fee']
    g_data.to_csv('muti_features.csv')
    # g_data.to_csv('age_features.csv')
    # g_data.to_csv('gender_features.csv')
    # g_data.to_csv('days_features.csv')
    # g_data.to_csv('complicate_features.csv')
    print()
def decision_tree_csv(data):
    g_data = pd.DataFrame()
    g_data['surgery'] = surgery_code(data)
    
    g_data['adrgid'] = data['adrgid']
    g_data['complication'] = complicate_code(data)
    g_data['fee'] = data['fee']
    g_data.to_csv('decision_tree.csv')
    # g_data.to_csv('age_features.csv')
    # g_data.to_csv('gender_features.csv')
    # g_data.to_csv('days_features.csv')
    # g_data.to_csv('complicate_features.csv')
    print()
# 是否手术编码
def surgery_code(data):
    datalen = len(data)
    sur =[]
    for i in range(0,datalen):
        s = data['surgery'][i]
        if pd.isnull(s):
            sur.append(0)
        else:
            sur.append(1)
    return sur
# 性别编码，男为1，女为0
def gender_code(data):
    datalen = len(data)
    gender =[]
    for i in range(0,datalen):
        sex = data['sex'][i]
        if sex=='未知':
            continue
        elif sex=='男':
            gender.append(1)
        else:
            gender.append(0)
    return gender
# 年龄计算
def age_code(data):
    age_list =[]
    datalen = len(data)
    for i in range(0,datalen):
        born_year = data['born'][i]
        if born_year=='0 AM':
            continue
        else:
            intime = ''.join(data['intime'][i])
            in_year = intime.strip().split("/")
            age = int(in_year[2])-int(born_year)
            age_list.append(age)
    return age_list
# 并发症严重程度，根据DRGS编码规则知道编码的最后一位，1表示严重、5表示一般、7表示无
def complicate_code(data):
    text_len =[]
    datalen = len(data)
    g_data = pd.DataFrame()
    for i in range(0,datalen):
        one_lines = data['drgsid'][i]
        text_len.append(one_lines[-1])
    return text_len
def clear_data(data):
    c_data = data
    datalen = len(data)
    for i in range(0,datalen):
        born_year = c_data['born'][i]
        sex = c_data['sex'][i]
        if born_year=='0 AM' or sex=='未知':
            c_data.drop([i],inplace=True)
            print()
    c_data.to_csv('clear_data.csv')
    print()
# 给样本打标签
def label_code(data):
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
    for  k in category.keys():
        maxn =max(category[k])
        minn =min(category[k])
        avg = maxn-minn
        low = avg*(4/10)
        mid = avg*(4/10)
        high =avg*(2/10)
        inteval_n = [minn,minn+low,minn+low+mid,minn+low+mid+high]
        mdict[k] = inteval_n
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
# 计算ADRG组下的高中低费用的三个区间
def interval_code(data):
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
    klist =[]
    mlist=[]
    for  k in category.keys():
        maxn =max(category[k])
        minn =min(category[k])
        avg = maxn-minn
        low = avg*(1/10)
        mid = avg*(7/10)
        high =avg*(2/10)
        inteval_n = [minn,minn+low,minn+low+mid,minn+low+mid+high]
        klist.append(k)
        mlist.append(inteval_n)
    new_data = pd.DataFrame()
    new_data['ADRG'] = klist
    new_data['inter'] = mlist
    new_data.to_csv('ADRG的区间.csv')
    print()
if __name__ =="__main__":
    t_data = pd.read_csv(train_data_file)#, names=['id', 'sex','born','intime','outtime','maindiag','elsediag','surgery','fee','days','drgsid','drgs','adrgid','adrg','highfee'])
    t_data.columns = ['id', 'sex','born','intime','outtime','maindiag','elsediag','surgery','fee','days','drgsid','drgs','adrgid','adrg','highfee']
    
    # n_data = t_data
    # n_data['nfee'] = avg_fee(t_data)
    # muti_logistic()
    decison_tree_csv(t_data)
    # generate_csv(t_data)
    # clear_data(t_data)

