import argparse
import math
import os
import string
import warnings

import gensim
import numpy as np
import pandas as pd
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer


warnings.filterwarnings("ignore")

##控制台，有许多需要修改的变量，只需要在前台输出即可
parser = argparse.ArgumentParser(description='')
parser.add_argument("--model_path", type=str, default="model/word2vec.model", help="path of the model")
parser.add_argument("--data_path", type=str, default="data/ec_sp500_all.csv", help="path of the data")
parser.add_argument("--threshold", type=float, default=0.5, help="the threshold of counting or not")
parser.add_argument("--vector_size", type=int, default=100, help="how many sizes of your vector, the bigger the more precise")
parser.add_argument("--window", type=int, default=8, help="the size of the window")
parser.add_argument("--topk", type=int, default=10, help="how many companies that you want to know")
parser.add_argument("--min_count", type=int, default=0, help="if smaller, delete")

args = parser.parse_args()


def normalize_document(doc):
    bs = BeautifulSoup(doc, 'html.parser')
    doc = bs.get_text()
    doc=re.sub(r'[^a-zA-Z\s]','',doc)
    doc = re.sub(r' +', ' ', doc)
    doc=doc.lower()
    doc=doc.strip()
    tokens=nltk.WordPunctTokenizer().tokenize(doc)
    doc=' '.join(tokens)
    # print(doc)
    return doc

def readFile(file):
    #如此便将所有数据保存在了df里面
    df = pd.read_csv(file, header=0, sep=",", encoding="ISO-8859-1")
    return df

def PreText(df_all):
    #去掉文本里面的\r \n
    df_all["content"] = df_all["content"].replace(r"\r", " ", regex=True)
    df_all["content"] = df_all["content"].replace(r"\n", " ", regex=True)
    #文本里所有的数据大写
    df_all["content"] = df_all["content"].str.upper()
    # 将标点符号都删除
    df_all["content"] = df_all["content"].str.replace(r"\s+", " ", regex=True)
    df_all["content_new"] = df_all["content"].replace(r"[{}]".format(string.punctuation), " ", regex=True).str.split(" ")

    #文本中有多余的空格，比如连续的空格或者行首和行尾的空格，可以去掉，以减少噪音。
    df_all["content"] = df_all["content"].str.split(r"[{}]".format(string.punctuation)).replace(" ", "")

    return df_all

def Train_model(df_Pre_all):
    # df_Pre_all["content"] = df_Pre_all["content"].astype(str)
    #将每一个独立的文本用.进行分割，否则模型会混淆不同的句子。
    # sentences = df_Pre_all["content"].str.cat(sep=".")
    sentences = df_Pre_all["content_new"].to_numpy()
    model_path = args.model_path
    if os.path.exists(model_path):
        model = gensim.models.word2vec.Word2Vec.load(model_path)
    else:
        model = gensim.models.word2vec.Word2Vec(sentences, vector_size = args.vector_size, window = args.window, min_count=args.min_count)
        if not os.path.exists("model") :
            os.makedirs("model")
        model.save(model_path)

    return model

def sumTheWord(df_Pre_all, model):

    first = []
    second = []
    third = []
    fourth = []
    fifth = []

    #先找出词汇表里面大于0.5的相关词汇，
    similar_words_temp = model.wv.most_similar("INNOVATION")

    first = ['Creativity', 'Innovative', 'Innovate', 'Innovation',
    'Creative', 'Excellence', 'Passion', 'World-class', 'Technology', 'Operational_excellence',
    'Passionate', 'Product_innovation', 'Capability', 'Customer_experience', 'Thought_leadership',
    'Expertise', 'Agility', 'Efficient', 'Technology_innovation', 'Competency', 'Know-how', 'Cutting-edge',
    'Agile', 'Creatively', 'Customer-centric', 'Enable', 'Value_proposition', 'Reinvent', 'Focus']
    second = ['Integrity', 'Accountability', 'Ethic', 'Integrity', 'Responsibility',
    'Transparency', 'Accountable', 'Governance', 'Ethical', 'Transparent', 
    'Trust', 'Responsible', 'Oversight', 'Independence', 'Objectivity', 
    'Moral', 'Trustworthy', 'Fairness', 'Hold_accountable', 'Corporate_governance', 
    'Autonomy', 'Core_value', 'Assure', 'Stakeholder', 'Fiduciary_responsibility',
    'Continuity', 'Credibility', 'Privacy', 'Fiduciary_duty', 'Honesty']
    third = ["Quality", "Dedicated", "Quality", "Dedication", "Customer_service", "Customer", "Dedicate", "Service_level", "Mission", "Service_delivery", "Reliability", "Service", "Commitment", "Customer_need", "Customer_support", "High-quality", "Ensure", "Quality_service", "Capable", "Product_quality", "End_user", "Quality_product", "Service_quality", "Service_capability", "Quality_level", "Customer_expectation", "Customer_satisfaction", "Client", "Customer_requirement", "Customer_relationship"]
    fourth = ["Respect", "Talented", "Talent", "Empower", "Employee", "Empowerment", "Leadership", "Culture", "Entrepreneurial", "Executive", "Management_team", "Skill", "Professionalism", "Staff", "Skill_set", "Competent", "High-caliber", "Highly_skilled", "Experienced", "Technologist", "Entrepreneur", "Manager", "Best_brightest", "Energize", "Energetic", "Empowered", "Leader", "Cultural", "Entrepreneurial_spirit", "Focused"]
    fifth = ["Teamwork", "Collaborate", "Cooperation", "Collaboration", "Collaborative", "Cooperative", "Team", "Partnership", "Cooperate", "Team_member", "Partner", "Collaboratively", "Engage", "Jointly", "Relationship", "Interaction", "Coordination", "Joint", "Cooperatively", "Collaborator", "Engagement", "Alliance", "Working_relationship", "Coordinate", "Business_partner", "Dialogue", "Association", "Technology_partnership", "Team_up", "Communication"]


    firstDic = pd.DataFrame(columns=first)
    for i in range(len(first)):
        first[i] = first[i].upper()
        firstDic[first[i]] = np.zeros(len(df_Pre_all))

    secondDic = pd.DataFrame(columns=second)
    for i in range(len(second)):
        second[i] = second[i].upper()
        secondDic[second[i]] = np.zeros(len(df_Pre_all))

    thirdDic = pd.DataFrame(columns=third)
    for i in range(len(third)):
        third[i] = third[i].upper()
        thirdDic[third[i]] = np.zeros(len(df_Pre_all))

    fourthDic = pd.DataFrame(columns=fourth)
    for i in range(len(fourth)):
        fourth[i] = fourth[i].upper()
        fourthDic[fourth[i]] = np.zeros(len(df_Pre_all))

    fifthDic = pd.DataFrame(columns=fifth)
    for i in range(len(fifth)):
        fifth[i] = fifth[i].upper()
        fifthDic[fifth[i]] = np.zeros(len(df_Pre_all))

    for i in range(len(df_Pre_all)):
        for word in first:
            try:
                firstDic[word][i] += df_Pre_all["content_new"][i].count(word)
            except IOError:
                firstDic[word][i] += 0
        for word in second:
            try:
                secondDic[word][i] += df_Pre_all["content_new"][i].count(word)
            except IOError:
                secondDic[word][i] = 0
        for word in third:
            try:
                thirdDic[word][i] += df_Pre_all["content_new"][i].count(word)
            except IOError:
                thirdDic[word][i] = 0
        for word in fourth:
            try:
                fourthDic[word][i] += df_Pre_all["content_new"][i].count(word)
            except IOError:
                fourthDic[word][i] = 0
        for word in fifth:
            try:
                fifthDic[word][i] += df_Pre_all["content_new"][i].count(word)
            except IOError:
                fifthDic[word][i] = 0

    firstDic["company"] = df_Pre_all["company"]
    secondDic["company"] = df_Pre_all["company"]
    thirdDic["company"] = df_Pre_all["company"]
    fourthDic["company"] = df_Pre_all["company"]
    fifthDic["company"] = df_Pre_all["company"]

    firstDic["fqtr"] = df_Pre_all["fqtr"]
    secondDic["fqtr"] = df_Pre_all["fqtr"]
    thirdDic["fqtr"] = df_Pre_all["fqtr"]
    fourthDic["fqtr"] = df_Pre_all["fqtr"]
    fifthDic["fqtr"] = df_Pre_all["fqtr"]

    return df_Pre_all, firstDic, secondDic,thirdDic,fourthDic, fifthDic


# 定义一个计算某个词的tf-idf值的函数
def tfidf(corpus, word):
    # 词汇表
    vocab = set()
    for doc in corpus:
        vocab.update(doc.split())
    vocab = list(vocab)
    # 词汇表大小
    vocab_size = len(vocab)
    # 文档数
    doc_num = len(corpus)
    # 判断词汇是否在词汇表中
    if word not in vocab:
        return 0 # 如果不在，返回0
    else:
        # 获取词汇的索引
        index = vocab.index(word)
        # 计算逆文档频率
        count = 0 # 统计包含该词的文档数
        for doc in corpus:
            if word in doc:
                count += 1
        idf = math.log(doc_num / (count + 1)) # 加1是为了避免分母为0
        # 计算TF-IDF值
        tfidf_value = 0
        for doc in corpus:
            if doc == " " or doc == "": continue
            words = doc.split()
            tf = words.count(word) / len(words) # 计算词频
            tfidf_value += tf * idf # 累加TF-IDF值
        # 返回TF-IDF值
        return tfidf_value

def getTFDic(firstDic, contents):

    for j in range(len(contents)):
        for column in firstDic.columns[0:firstDic.shape[1]-2]:
            firstDic[column][j] = tfidf(contents[j], column)

    return firstDic

##核心目的： 讨论关税话题最多的十家公司
# 一共需要做几件事情：

    # 1. 读入文件，将文件的行列分好, 我只需要将数据分为公司, 内容即可。
    # 2. 老师规定要有一些简单的数据预处理部分
        # a.奇奇怪怪的\r \n类似的字符，
        # b.将所有字母统一大写
        # c.将标点符号删除
    # 3. 处理过后文本数据，放到word2vec模型进行训练: 结果就是：得到所有文本对应的向量。tariff:[0,1,2], tax:[1,2,3]
    # 4. 需要算每一个单词和这四个单词之间的cos similarity， 排序，选最高的几个，统计个数。然后再输出公司是哪些。


# 那么首先就需要先指定文件是哪个文件
file = args.data_path
# 专门写一个函数来处理文本，使其精简成为我需要的东西
df_all = readFile(file)

# 数据进行预处理，这里主要是做了去掉一些：
# a.奇奇怪怪的\r \n类似的字符，
# b.将所有字母统一大写
# c.将标点符号删除
df_Pre_all = PreText(df_all)

# 将数据扔进模型中进行训练
model = Train_model(df_Pre_all)

# if os.path.exists("results/firstDic.csv"):
if 0:
    firstDic = pd.read_csv("results/firstDic.csv")
    secondDic = pd.read_csv("results/secondDic.csv")
    thirdDic = pd.read_csv("results/thirdDic.csv")
    fourthDic = pd.read_csv("results/fourthDic.csv")
    fifthDic = pd.read_csv("results/fifthDic.csv")
else:
    results, firstDic, secondDic,thirdDic,fourthDic, fifthDic  = sumTheWord(df_Pre_all, model)

    ##计算tf-idf
    firstDic = getTFDic(firstDic, results["content"])
    secondDic = getTFDic(secondDic, results["content"])
    thirdDic = getTFDic(thirdDic, results["content"])
    fourthDic = getTFDic(fourthDic, results["content"])
    fifthDic = getTFDic(fifthDic, results["content"])

    firstDic["sumFirst"] = firstDic[firstDic.columns[0:firstDic.shape[1]-2]].sum(axis=1)
    secondDic["sumSecond"] = secondDic[secondDic.columns[0:secondDic.shape[1]-2]].sum(axis=1)
    thirdDic["sumThird"] = thirdDic[thirdDic.columns[0:thirdDic.shape[1]-2]].sum(axis=1)
    fourthDic["sumFourth"] = fourthDic[fourthDic.columns[0:fourthDic.shape[1]-2]].sum(axis=1)
    fifthDic["sumFifth"] = fifthDic[fifthDic.columns[0:fifthDic.shape[1]-2]].sum(axis=1)

    firstDic.to_csv("results_new/firstDic.csv")
    secondDic.to_csv("results_new/secondDic.csv")
    thirdDic.to_csv("results_new/thirdDic.csv")
    fourthDic.to_csv("results_new/fourthDic.csv")
    fifthDic.to_csv("results_new/fifthDic.csv")


df2 = pd.concat([df['star_rating'],
                 pd.DataFrame(tv_matrix.toarray(), columns=tv.get_feature_names())
                 ], axis=1)



results = pd.concat([df_Pre_all["company"], df_Pre_all["fqtr"], df_Pre_all["fyr"], firstDic["sumFirst"], secondDic["sumSecond"], thirdDic["sumThird"], fourthDic["sumFourth"], fifthDic["sumFifth"]], axis=1,
                    keys = ["company", "fqtr", "fyr", "Innovation", "Integrity", "Quality", "Respect", "Teamwork"])

Allyears = results["fyr"].drop_duplicates().to_numpy()
for year in Allyears:
    resultsTemp = results[results["fyr"] == year]
    resultsTemp.to_csv("results_new/AllScore" + str(year) + ".csv")
results.to_csv("results_new/AllScore.csv")