
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import *
get_ipython().magic(u'matplotlib inline')


# In[2]:


df_stage1_train = pd.read_csv('data/training_variants')

df_stage1_train_text = pd.read_csv("data/training_text", sep="\|\|",engine="python",
                         header=None,skiprows=1, names = ["ID","Text"])

df_stage1_train = pd.merge(df_stage1_train,df_stage1_train_text,how='left',on='ID')

df_stage1_train =  df_stage1_train.fillna('')



# In[3]:


df_stage1_test = pd.read_csv('data/test_variants')

df_stage1_test_text = pd.read_csv("data/test_text", sep="\|\|",engine="python",
                         header=None,skiprows=1, names = ["ID","Text"])

df_stage1_test = pd.merge(df_stage1_test,df_stage1_test_text,how='left',on='ID')

df_stage1_test = df_stage1_test.fillna('')




# In[4]:


# Reading the filtered solutions
df_solution_stage1 = pd.read_csv('data/stage1_solution_filtered.csv')
df_solution_stage1['Class'] = pd.to_numeric(df_solution_stage1.drop('ID',axis=1).idxmax(axis=1).str[5:])


# In[5]:


df_stage1_test = pd.merge(df_stage1_test,df_solution_stage1[['ID','Class']], on='ID',how='left')


# In[6]:


df_stage1_test.drop('ID',axis = 1)
df_stage1_test = df_stage1_test[df_stage1_test['Class'].notnull()]


# In[7]:


df_stage2_train = pd.concat([df_stage1_train, df_stage1_test])


# In[8]:


df_stage2_test = pd.read_csv('data/stage2_test_variants.csv')

df_stage2_test_text = pd.read_csv("data/stage2_test_text.csv", sep="\|\|",engine="python",
                         header=None,skiprows=1, names = ["ID","Text"])

df_stage2_test = pd.merge(df_stage2_test,df_stage2_test_text,how='left',on='ID')

df_stage2_test = df_stage2_test.fillna('')

pid = df_stage2_test['ID'].values
df_stage2_train.reset_index(drop=True, inplace=True)

train = df_stage2_train
test = df_stage2_test

train_y = train['Class'].values

df_all = pd.concat((df_stage2_train,df_stage2_test),axis=0)


# In[9]:


train



# Checking number of unique genes
# In[10]:


train_gene_list = train.Gene.unique()
print("There are {} different number of genes in training dataset".format(len(train_gene_list)))

test_gene_list = test.Gene.unique()
print("There are {} different number of genes in test dataset".format(len(train_gene_list)))


# In[11]:


common_genes = set(train_gene_list) & set(test_gene_list)
print("There are {} number of common genes in train and test datasets".format(len(common_genes)))


# In[12]:


#Top genes in the training dataset
train_gene_count = train.Gene.value_counts()
train_gene_count.columns = ["Gene","Count"]
train_gene_count[train_gene_count > 50]

train_gene_count = pd.DataFrame(train_gene_count)
train_gene_count.columns = ["Count"]
train_gene_count_sorted = train_gene_count.sort_values("Count",ascending=0)

train_gene_count_sorted[:10]


# In[13]:


#Top genes in the test dataset
test_gene_count = test.Gene.value_counts()
test_gene_count[test_gene_count >30]


# In[14]:


# Plotting the number of data observations w.r.t class
plt.figure(figsize=(12,8))
sns.set(style = "white",palette="muted",color_codes=True)
sns.countplot(x="Class", data=train)
plt.show()


# In[15]:



fig, ax = plt.subplots(nrows =3,ncols =3,figsize=(18,18))

for i in range(3):
    for j in range(3):
        train_data = train[train['Class'] == i*3+j+1].groupby('Gene')["ID"].count().reset_index()
        train_data=train_data.sort_values('ID', ascending=False)[:10]
        sns.barplot(x="Gene", y="ID",data=train_data,ax=ax[i][j] )
    
    


# In[16]:



def get_gene_text_count(dframe):
    dframe['Gene_count'] = dframe.apply(lambda r: sum([1 for w in r['Gene'].split(' ')
                            if w in r['Text'].split(' ')]), axis=1)

    dframe['Text_count'] = dframe["Text"].apply(lambda x: len(x.split(' ')))
    
    Labelencoder = preprocessing.LabelEncoder()
    dframe['Gene_encoder'] = Labelencoder.fit_transform(df_all['Gene'].values)
    dframe['Variation'] = Labelencoder.fit_transform(df_all['Variation'].values)
    
    return dframe



# In[17]:


first_and_pos = df_all.Variation.str.extract('^(?=.{1,7}$)([a-zA-Z]+)([0-9].*)$',expand=True)


first_and_pos = first_and_pos.rename(columns={0:'First_letter',1:'Dummy'})
del first_and_pos['Dummy']
threshold = 10 # Anything that occurs less than this will be removed.
value_counts = first_and_pos.First_letter.value_counts() # Entire DataFrame 
to_remove = value_counts[value_counts <= threshold].index
first_and_pos.First_letter.replace(to_remove,'', inplace=True)

Firstletter_onehot = pd.get_dummies(first_and_pos,prefix=['First_letter_'])


# In[18]:


pos_and_last = df_all.Variation.str.extract('([0-9]+)([a-zA-Z]|.*)$',expand=True)
pos_and_last = pos_and_last.rename(columns={0:'Gene_Position',1:'Last_letter'})
pos_and_last.head(1)
del pos_and_last['Gene_Position']
threshold = 30 # Anything that occurs less than this will be removed.
value_counts = pos_and_last.Last_letter.value_counts() # Entire DataFrame 
to_remove = value_counts[value_counts <= threshold].index
pos_and_last.Last_letter.replace(to_remove,'', inplace=True)

Lastletter_onehot = pd.get_dummies(pos_and_last,prefix=['Last_letter_'])


# In[19]:


variation_feat = pd.DataFrame()


# In[20]:



variation_feat["is_del"] = df_all.Variation.str.contains('del',case=False)
variation_feat["is_ins"] = df_all.Variation.str.contains('ins',case=False)
variation_feat["is_fus"] = df_all.Variation.str.contains('fus',case=False)
variation_feat["is_trunc"] = df_all.Variation.str.contains('trunc',case=False)
variation_feat["is_methyl"] = df_all.Variation.str.contains('methyl',case=False)
variation_feat["is_amp"] = df_all.Variation.str.contains('amp',case=False)
variation_feat["is_sil"] = df_all.Variation.str.contains('sil',case=False)
variation_feat["is_splice"] = df_all.Variation.str.contains('splice',case=False)
variation_feat["is_exon"] = df_all.Variation.str.contains('exon',case=False)


# In[21]:


variation_feat = variation_feat.astype(int)


# In[22]:


df_all = pd.concat((df_all,Firstletter_onehot,Lastletter_onehot,
                               variation_feat),axis=1)


# In[23]:



df_all = get_gene_text_count(df_all)
#train = get_gene_text_count(train)
train = df_all.iloc[:len(train)]
test = df_all.iloc[len(train):]



# In[24]:


class_group = train.groupby('Class')["Text_count"]
class_group.describe()


# In[25]:


# checking whethere there are
# any null values for the clinical text report
train[train["Text_count"] < 200]


# In[26]:


# Printing the rows which have null in Text value

train[train["Text_count"] == 1]


# In[27]:


# Checking whether test data have null values for text or not
test[test["Text_count"] == 1]


# In[28]:


# Plotting the text count distribution w.r.t a single class
plt.figure(figsize=(12,8))
sns.violinplot(x="Class", y="Text_count", data=train,inner=None)
plt.show()


# In[29]:


# Calculating Tf-Idf matrix for the clinical text

#Initializing the tfidf instance from sklearn

tfidf = TfidfVectorizer(max_df = 0.95,
    min_df=5,strip_accents='unicode',lowercase =True,
    analyzer='word', token_pattern=r'\w+', use_idf=True, 
    smooth_idf=True, sublinear_tf=True,ngram_range=(1,2), stop_words = 'english')

train_tfidf = tfidf.fit_transform(train["Text"])

test_tfidf = tfidf.transform(test["Text"])
feature_names = tfidf.get_feature_names()


# In[30]:


'''from sklearn.feature_extraction.text import CountVectorizer
n_features = 50
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=50,
                                stop_words='english')
train_tf = tf_vectorizer.fit_transform(train['Text'])
test_tf = tf_vectorizer.transform(test['Text'])

from sklearn.decomposition import  LatentDirichletAllocation
n_components = 9
unique_items, counts = np.unique(train_y, return_counts=True)
priors = counts.astype(float)/np.sum(counts)

lda = LatentDirichletAllocation(n_topics=n_components,max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda2 = LatentDirichletAllocation(n_topics=n_components,max_iter=5,doc_topic_prior=priors,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
train_tfidf = lda.fit_transform(train_tf)
test_tfidf = lda.transform(test_tf)'''


# In[31]:


np.shape(train_tfidf)


# In[32]:


# Reducing the dimensions of the tfidf matrix 
# by using singular value decomposition
# 20 and 30
svd = TruncatedSVD(n_components=40,n_iter = 50,random_state = 42)

train_svd = svd.fit_transform(train_tfidf)
test_svd = svd.transform(test_tfidf)


# In[33]:


np.shape(test_svd)


# In[34]:



train_full = pd.concat((train,pd.DataFrame(train_svd)),axis=1)
test_full = pd.concat((test,pd.DataFrame(test_svd)),axis=1)



# In[35]:


train


# In[36]:


train_y = train_y - 1


# In[37]:


train_full = train_full.drop(['Gene','Variation','ID','Text','Class'],axis=1).values

test_full = test_full.drop(['Gene','Variation','ID','Text','Class'],axis=1).values


# In[38]:


np.shape(train_full)


# In[39]:


np.shape(test_full)


# In[40]:


from sklearn import *
import xgboost as xgb
denom = 0
fold = 5 #Change to 5, 1 for Kaggle Limits
for i in range(fold):
    params = {
        'eta': 0.02333,
        'max_depth': 7,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 9,
        'seed': i,
        'silent': True
    }
    x1, x2, y1, y2 = model_selection.train_test_split(train_full, train_y, test_size=0.18, random_state=i)#0.18
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), 1000,  watchlist, verbose_eval=50, early_stopping_rounds=100)
    score1 = metrics.log_loss(y2, model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit), labels = list(range(9)))
    print(score1)
    #if score < 0.9:
    if denom != 0:
        pred = model.predict(xgb.DMatrix(test_full), ntree_limit=model.best_ntree_limit+80)
        preds += pred
    else:
        pred = model.predict(xgb.DMatrix(test_full), ntree_limit=model.best_ntree_limit+80)
        preds = pred.copy()
    denom += 1
    submission = pd.DataFrame(pred, columns=['class'+str(c+1) for c in range(9)])
    submission['ID'] = pid
    submission.to_csv('submission_xgb_fold_'  + str(i) + '.csv', index=False)
preds /= denom
submission = pd.DataFrame(preds, columns=['class'+str(c+1) for c in range(9)])
submission['ID'] = pid
submission.to_csv('subbmission_xgb.csv', index=False)



