import numpy as np
import pandas as pd
# import tensorflow as tf
from category_encoders.target_encoder import TargetEncoder
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import datetime

train_data_file = './cdata.csv'
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题
# DRGS编码与费用，进行目标编码
def maindiag_extract(data):
    text_len =[]
    datalen = len(data)
    g_data = pd.DataFrame()
    for i in range(0,datalen):
        one_lines = data['drgsid'][i]
        text_len.append(one_lines[-1])

    g_data['fee'] =data['fee']
    encoder=TargetEncoder(cols='maindiag') 
    maindiaglist = encoder.fit_transform(data['maindiag'],data['fee'])
    g_data['maindiag'] = maindiaglist
    g_data['complication'] =text_len
    g_data.to_csv('price.csv')
    print()
def elsediag_extract(data):
    text_len =[]
    datalen = len(data)
    for i in range(0,datalen):
        nontext = data['elsediag'][i]
        if pd.isnull(nontext):
            continue
        one_lines = ''.join(list(nontext))
        text = one_lines.strip().split(",")
        for j in range(len(text)):
            text_id = text[j].strip().split("|")
            text_len.append(text_id[0])
    all_category = list(set(text_len))
    print(all_category)
    print(len(all_category))
    print()
def drgs_extract(data):
    text_len =[]
    datalen = len(data)
    for i in range(0,datalen):
        text_id = data['drgsid'][i]
        text_len.append(text_id)
    all_category = list(set(text_len))
    print(all_category)
    print(len(all_category))
    print()
def feature_extract(data):
    text_len =[]
    datalen = len(data)
    for i in range(0,datalen):
        nontext = data['elsediag'][i]
        if pd.isnull(nontext):
            continue
        one_lines = ''.join(list(nontext))
        text = one_lines.strip().split(",")
        for j in range(len(text)):
            text_id = text[j].strip().split("|")
            text_len.append(text_id[0])
    all_category = list(set(text_len))
    print(all_category)
    print(len(all_category))
    arr = np.zeros((len(data),803))
    n_data = pd.DataFrame(arr,columns = all_category)
    n_data['maindiag'] = maindiag_extract(t_data)
    n_data['fee'] = t_data['fee']
    
    # 实现把803个特征进行编码
    for i in range(0,datalen):
        nontext = data['elsediag'][i]
        if pd.isnull(nontext):
            continue
        # one_lines = ''.join(list(nontext))
        text = nontext.strip().split(",")
        for j in range(len(text)):
            text_id = text[j].strip().split("|")
            n_data.loc[i,text_id[0]] = 1
    n_data.to_csv('feature_data.csv')
    print()

def fee_range(data):
    text_len =[]
    # category =[]
    category={}
    feelist =[]
    datalen = len(data)
    for i in range(0,datalen):
        text_id = data['drgsid'][i]
        data_fee = data['fee'][i]
        feelist.append(data_fee)
        category[text_id] =list(set(feelist))   
    ncate ={} 
    for k in category.keys():
        ncate[k] = np.mean(category[k])
    
    a_cate = dict(sorted(ncate.items(), key=lambda x: x[1], reverse=True))
    x = list(a_cate.keys())
    y = list(a_cate.values())
    plt.xlabel('DRGs')
    plt.ylabel('Medical fees')
    plt.xlabel('DRGs')
    plt.title('Average cost of different DRGS groups')
    plt.xticks([])
    plt.scatter(x, y, alpha=0.9)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
    plt.show()
    print(a_cate)  
    print()

def box_line(data):
    text_len =[]
    # category =[]
    category={}
    cate_box = pd.DataFrame()
    datalen = len(data)
    # feelist =[]
    for i in range(0,datalen):
        text_id = data['adrgid'][i]
        data_fee = data['fee'][i]
        if text_id in category.keys():
            templist = list(category[text_id])
            templist.append(data_fee)
            category[text_id] =list(templist)
        else:
            category[text_id] = [data_fee]
    pxy = {}
    for k in category.keys():
        pxy[k] = np.mean(category[k])
        # print(k,len(category[k]))
    sordict = dict(sorted(pxy.items(), key=lambda x: x[1]))
    resultxy ={}
    for k in sordict.keys():
        resultxy[k] = category[k]
        print(k)
    
    x = list(resultxy.keys())
    y = list(resultxy.values())
    for j in resultxy.keys():
        print(j,resultxy[j])
    plt.xlabel('DRGs')
    plt.title('Distribution of the number of grouping categories ')
    plt.ylabel('The amount of DRGS')
    plt.xticks([])
    # x = [i for i in range(len(y))]
    plt.scatter(x, y, alpha=0.9)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
    plt.show()
    print()
    
    for k in resultxy.keys():
        templi = list(resultxy[k])
        templen = len(templi)
        if 4272 > templen:
            for i in range(4272-templen):
                templi.append(np.nan)
        cate_box[k] = templi
    cate_box.plot.box(title="Fee-categroy")
    plt.grid(linestyle="--", alpha=0.3)
    plt.title('Relationship between DRGS and medical fee')
    plt.xlabel('DRGS')
    plt.xticks([])
    plt.ylabel('medical fee')
    plt.show()
    
    print()  
def highfee(data):
    text_len =[]
    datalen = len(data)
    for i in range(0,datalen):
        one_lines = ''.join(list(data['highfee']))
        text_id = one_lines.strip().split("|")
        text_len.append(text_id[0])
    all_category = list(set(text_len))
    print(all_category)
    print(len(all_category))
    encoder=TargetEncoder(cols='maindiag') 
    maindiaglist = encoder.fit_transform(data['maindiag'],data['fee'])
    return maindiaglist
def age_static(data):
    age_fee ={}
    datalen = len(data)
    for i in range(0,datalen):
        born_year = data['born'][i]
        if born_year=='0 AM':
            continue
        else:
            intime = ''.join(data['intime'][i])
            in_year = intime.strip().split("/")
            age = int(in_year[2])-int(born_year)
            data_fee = data['fee'][i]
            if age in age_fee.keys():
                templist = list(age_fee[age])
                templist.append(data_fee)
                age_fee[age] =list(templist)
            else:
                age_fee[age] = [data_fee] 
    avg_ls =[]
    age_ls =[]
    # 计算平均费用
    avg_age_fee ={}
    for k in age_fee.keys():
        avg = np.mean(list(age_fee[k]))
        avg_age_fee[k] = avg
    sort_avg_fee = dict(sorted(avg_age_fee.items(), key=lambda x: x[0]))
    
    # for m in sort_avg_fee.keys():
    #     age_ls.append(m)
    #     avg_ls.append(sort_avg_fee[m])
    # avg_df = pd.DataFrame()
    # avg_df['年龄'] = age_ls
    # avg_df['费用'] = avg_ls
    
    # avg_df.to_csv('age_fee.csv')

    # 绘制折线图
    print(sort_avg_fee)
    x = list(sort_avg_fee.keys())
    y = list(sort_avg_fee.values())
    plt.plot(x,y,'b--',label='age-fee')
    plt.title('Relationship between age and cost')
    plt.xlabel('age')
    plt.ylabel('medical-fee')
    plt.show()
    
    #绘制直方图，阶段年龄与平均费用的
    li30 =[]
    li40 =[]
    li50 =[]
    li60 =[]
    li70 =[]
    limax =[]
    n_age_fee = {}
    for k in age_fee.keys():
        age = int(k)
        if age <=30:
            li30.extend(age_fee[k])
        elif age <=40:
            li40.extend(age_fee[k])
        elif age <=50:
            li50.extend(age_fee[k])
        elif age <=60:
            li60.extend(age_fee[k])
        elif age <=70:
            li70.extend(age_fee[k])
        else:
            limax.extend(age_fee[k])
    '''
    n_age_fee[30] = len(li30)
    n_age_fee[40] = len(li40)
    n_age_fee[50] = len(li50)
    n_age_fee[60] = len(li60)
    n_age_fee[70] = len(li70)
    n_age_fee[90] = len(limax)
    '''
    
    n_age_fee[30] = np.mean(li30)
    n_age_fee[40] = np.mean(li40)
    n_age_fee[50] = np.mean(li50)
    n_age_fee[60] = np.mean(li60)
    n_age_fee[70] = np.mean(li70)
    n_age_fee[90] = np.mean(limax)
    
    # 计算平均费用
    sort_level_age_fee = dict(sorted(n_age_fee.items(), key=lambda x: x[0]))
    x = ['<=30','31-40','41-50','51-60','61-70','>=70']
    y = list(sort_level_age_fee.values())
    # plt.title('Relationship between age_range and populatioin')
    plt.title('Relationship between age_range and medical fee')
    plt.xlabel('age_range')
    # plt.ylabel('populatioin')
    plt.ylabel('medical fee')
    for a,b in zip(x,y):  
        plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=11)
    plt.bar(x, y)
    plt.show()
    print()
def sex_static(data):
    sex_fee ={}
    datalen = len(data)
    male =[]
    female =[]
    for i in range(0,datalen):
        sex = data['sex'][i]
        if sex=='未知':
            continue
        elif sex=='男':
            male.append(data['fee'][i])
        else:
            female.append(data['fee'][i])
        sex_fee['male'] = male
        sex_fee['female'] = female
    # 计算平均费用
    avg_sex_fee ={}
    for k in sex_fee.keys():
        avg = np.mean(list(sex_fee[k]))
        avg_sex_fee[k] = avg
    n_sex_fee={}
    n_sex_fee['male'] = len(sex_fee['male'])
    n_sex_fee['female'] = len(sex_fee['female'])
  
    x = ['male','female']
    y1 = list(avg_sex_fee.values())
    y2 = list(n_sex_fee.values())
    plt.figure()
    plt.title('Relationship between gender and cost')
    plt.xlabel('gender')
    plt.ylabel('medical-fee')
    for a,b in zip(x,y1):  
        plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=11)
    plt.bar(x, y1)
    plt.figure()
    plt.title('Relationship between gender and population')
    plt.xlabel('gender')
    plt.ylabel('population')
    for a,b in zip(x,y2):  
        plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=11)
    plt.bar(x, y2)
    plt.show()
    print()
def duration_static(data):
    duration_fee ={}
    datalen = len(data)
    for i in range(0,datalen):
        intime  = ''.join(data['intime'][i]).strip()
        outtime = ''.join(data['outtime'][i]).strip()
        data_fee = data['fee'][i]
        date1=datetime.datetime.strptime(outtime[0:10],"%m/%d/%Y")
        date2=datetime.datetime.strptime(intime[0:10],"%m/%d/%Y")
        day =(date1-date2).days
        if int(day)>300:
            continue
        if day in duration_fee.keys():
            templist = list(duration_fee[day])
            templist.append(data_fee)
            duration_fee[day] =list(templist)
        else:
            duration_fee[day] = [data_fee] 
    # 计算平均费用
    avg_duration_fee ={}
    for k in duration_fee.keys():
        avg = np.mean(list(duration_fee[k]))
        avg_duration_fee[k] = avg
    sort_avg_fee = dict(sorted(avg_duration_fee.items(), key=lambda x: x[0]))
    #绘制直方图，阶段年龄与平均费用的
    li01 =[]
    li25 =[]
    li69 =[]
    li60 =[]
    li100 =[]
    n_duration_fee = {}
    a_duration_fee ={}
    for k in duration_fee.keys():
        day = int(k)
        if day ==0 or day ==1:
            li01.extend(duration_fee[k])
        elif day <=5:
            li25.extend(duration_fee[k])
        elif day <=9:
            li69.extend(duration_fee[k])
        elif day <=60:
            li60.extend(duration_fee[k])
        else:
            li100.extend(duration_fee[k])
    maxday = max(duration_fee.keys())
    '''
    n_duration_fee[1] = len(li01)
    n_duration_fee[25] = len(li25)
    n_duration_fee[69] = len(li69)
    n_duration_fee[60] = len(li60)
    n_duration_fee[100] = len(li100)
    '''
    # 计算平均费用
    a_duration_fee[1] = np.mean(li01)/2
    a_duration_fee[25] = np.mean(li25)/4
    a_duration_fee[69] = np.mean(li69)/4
    a_duration_fee[60] = np.mean(li60)/50
    a_duration_fee[100] = np.mean(li100)/(maxday-60)
    sort_level_duration_fee = dict(sorted(a_duration_fee.items(), key=lambda x: x[0]))
    '''
    x1 = list(sort_avg_fee.keys())
    x2 = ['0-1','2-5','6-9','10-60','>=60']
    y1 = list(sort_avg_fee.values())
    y2 = list(sort_level_duration_fee.values())
    plt.title('Relationship between hospital-time and medical fee')
    plt.xlabel('day range')
    plt.ylabel('medical fee')
    # plt.xticks([])
    plt.scatter(x1, y1, alpha=0.9)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
    plt.figure()
    plt.title('Relationship between hospital-time and population')
    plt.xlabel('day range')
    plt.ylabel('population')
    for a,b in zip(x2,y2):  
        plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=11)
    plt.bar(x2, y2)
    plt.show()
    '''
    x3 = ['0-1','2-5','6-9','10-60','>=60']
    y3 = list(sort_level_duration_fee.values())
    plt.title('Relationship between hospital-time and medical fee')
    plt.xlabel('day range')
    plt.ylabel('medical fee')
    for a,b in zip(x3,y3):  
        plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=11)
    plt.bar(x3, y3)
    plt.show()
    print()
def complication_static(data):
    com_fee ={}
    datalen = len(data)
    serious =[]
    general =[]
    non = []
    for i in range(0,datalen):
        drgsid = ''.join(data['drgsid'][i]).strip()
        lastid = int(drgsid[-1])
        if lastid==1:
            serious.append(data['fee'][i])
        elif lastid==3:
            general.append(data['fee'][i])
        elif lastid==5:
            non.append(data['fee'][i])
        else:
            continue
        com_fee['serious'] = serious
        com_fee['general'] = general
        com_fee['non'] = non
    # 计算平均费用
    avg_com_fee ={}
    for k in com_fee.keys():
        avg = np.mean(list(com_fee[k]))
        avg_com_fee[k] = avg
    n_com_fee={}
    n_com_fee['serious'] = len(com_fee['serious'])
    n_com_fee['general'] = len(com_fee['general'])
    n_com_fee['non'] = len(com_fee['non'])

    x = ['serious','general','non']
    y1 = list(avg_com_fee.values())
    y2 = list(n_com_fee.values())
    plt.figure()
    plt.title('Relationship between complication and medical fee')
    plt.xlabel('complication')
    plt.ylabel('medical fee')
    for a,b in zip(x,y1):  
        plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=11)
    plt.bar(x, y1)
    plt.figure()
    plt.title('Relationship between complication and population')
    plt.xlabel('complication')
    plt.ylabel('population')
    for a,b in zip(x,y2):  
        plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=11)
    plt.bar(x, y2)
    plt.show()
    print()
def drgs_box_line(data):
    category={}
    cate_box = pd.DataFrame()
    datalen = len(data)
    # feelist =[]
    for i in range(0,datalen):
        text_id = data['adrgid'][i]
        data_fee = data['fee'][i]
        if text_id in category.keys():
            templist = list(category[text_id])
            templist.append(data_fee)
            category[text_id] =list(set(templist))
        else:
            category[text_id] = [data_fee]
    pxy = {}

    for k in category.keys():
        pxy[k] = np.mean(category[k])
        # print(k,len(sordict[k]))
    sordict = dict(sorted(pxy.items(), key=lambda x: x[1]))
    resultxy ={}
    for k in sordict.keys():
        resultxy[k] = category[k]
        # print(k,sordict[k])
    for k in resultxy.keys():
        templi = list(resultxy[k])
        templen = len(templi)
        if 4682 > templen:
            for i in range(4682-templen):
                templi.append(np.nan)
        cate_box[k] = templi
    cate_box.plot.box(title="Fee-categroy")
    plt.grid(linestyle="--", alpha=0.3)
    plt.title('Relationship between ADRG and medical fee')
    plt.xlabel('ADRG')
    plt.ylabel('medical fee')
    plt.show()
    
    print()  
def highfee_static(data):
    datalen = len(data)
    high=[]
    mid =[]
    low=[]
    high_dict={}
    for i in range(0,datalen):
        judge = data['highfee'][i]
        adrg = data['adrgid'][i]
        # datafee = data['fee'][i]
        if pd.isnull(judge):
            mid.append(adrg)
        elif '高' in judge :
            high.append(adrg)
        elif '低' in judge:
            low.append(adrg)
    high_dict['high'] =list(set(high))
    high_dict['mid'] =list(set(mid))
    high_dict['low'] =list(set(low))      
    for k in high_dict.keys():
        print(k,high_dict[k])
    print()
def avg_fee(data):
    datalen = len(data)
    fee_list=[]
    for i in range(0,datalen):
        fee = float(data['fee'][i])
        days = int(data['days'][i])
        avg = fee/days
        fee_list.append(avg)
    return fee_list
def avg_complication_static(data):
    com_fee ={}
    datalen = len(data)
    serious =[]
    general =[]
    non = []
    for i in range(0,datalen):
        drgsid = ''.join(data['drgsid'][i]).strip()
        lastid = int(drgsid[-1])
        if lastid==1:
            serious.append(data['nfee'][i])
        elif lastid==3:
            general.append(data['nfee'][i])
        elif lastid==5:
            non.append(data['nfee'][i])
        else:
            continue
        com_fee['serious'] = serious
        com_fee['general'] = general
        com_fee['non'] = non
    # 计算平均费用
    avg_com_fee ={}
    for k in com_fee.keys():
        avg = np.mean(list(com_fee[k]))
        avg_com_fee[k] = avg
    n_com_fee={}
    n_com_fee['serious'] = len(com_fee['serious'])
    n_com_fee['general'] = len(com_fee['general'])
    n_com_fee['non'] = len(com_fee['non'])

    x = ['serious','general','non']
    y1 = list(avg_com_fee.values())
    y2 = list(n_com_fee.values())
    plt.figure()
    plt.title('Relationship between complication and medical fee')
    plt.xlabel('complication')
    plt.ylabel('medical fee')
    for a,b in zip(x,y1):  
        plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=11)
    plt.bar(x, y1)
    plt.figure()
    plt.title('Relationship between complication and population')
    plt.xlabel('complication')
    plt.ylabel('population')
    for a,b in zip(x,y2):  
        plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=11)
    plt.bar(x, y2)
    plt.show()
    print()
def adrgid_plot(data):
    text_len =[]
    category={}
    datalen = len(data)
    for i in range(0,datalen):
        text_id = data['adrgid'][i]
        if text_id in category.keys():
            aid = category[text_id]+1
            category[text_id] =aid
        else:
            category[text_id] = 1

    sordict = dict(sorted(category.items(), key=lambda x: x[1]))
    
    x = list(sordict.keys())
    y = list(sordict.values())
    plt.xlabel('ADRG categories')
    plt.title('Distribution of the number of grouping categories ')
    plt.ylabel('The amount of ADRG')
    for a,b in zip(x,y):  
        plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=11)
    plt.bar(x, y)
    plt.show()
    print()
if __name__ =="__main__":
    t_data = pd.read_csv('./clear_data.csv')#, names=['id', 'sex','born','intime','outtime','maindiag','elsediag','surgery','fee','days','drgsid','drgs','adrgid','adrg','highfee'])
    t_data.columns = ['id', 'sex','born','intime','outtime','maindiag','elsediag','surgery','fee','days','drgsid','drgs','adrgid','adrg','highfee']
    
    # n_data = t_data
    # n_data['nfee'] = avg_fee(t_data)

    # maindiag_extract(t_data)
    # elsediag_extract(t_data)
    # drgs_extract(t_data)
    # feature_extract(t_data)
    # muti_logistic()
    # fee_range(t_data)
    # box_line(t_data)
    # age_static(t_data)
    # sex_static(t_data)
    # duration_static(t_data)
    # complication_static(t_data)
    # drgs_box_line(t_data)
    # highfee_static(t_data)
    # avg_complication_static(n_data)
    adrgid_plot(t_data)
    print()

