import numpy as np
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from sklearn.externals import joblib
import os
train_data_path = './data/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv'
# test_data_path = './data/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv'
val_data_path = './data/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv'

df = pd.read_csv(train_data_path)
val_df = pd.read_csv(val_data_path)
labels=[ 'location_traffic_convenience',
 'location_distance_from_business_district',
 'location_easy_to_find',
 'service_wait_time',
 'service_waiters_attitude',
 'service_parking_convenience',
 'service_serving_speed',
 'price_level',
 'price_cost_effective',
 'price_discount',
 'environment_decoration',
 'environment_noise',
 'environment_space',
 'environment_cleaness',
 'dish_portion',
 'dish_taste',
 'dish_look',
 'dish_recommendation',
 'others_overall_experience',
 'others_willing_to_consume_again']
 
 X_train, X_test, Y_train, Y_test = df['content'].fillna(' '),val_df['content'].fillna(' '),df.loc[:,labels],val_df.loc[:,labels]
 Y_test = val_df.loc[:,labels]
 Y_test.columns.values.tolist()
 data_x_char = pd.concat([X_train,X_test])
 data_x_char = data_x_char.fillna(' ')
 char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    #stop_words='stopwords_list',    
    ngram_range=(2, 4),
    max_features=50000)
  
  char_vectorizer.fit(data_x_char)
  X_train_word = [' '.join(jieba.cut(x)) for x in X_train]
  X_test_word = [' '.join(jieba.cut(x)) for x in  X_test]
  data_x_word = X_train_word+X_test_word
  
  word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    #stop_words=stopwords_list,
    ngram_range=(1, 3),
    max_features=10000)
word_vectorizer.fit(data_x_word)

train_word_features = word_vectorizer.transform(X_train_word)
test_word_features = word_vectorizer.transform(X_test_word)

train_char_features = char_vectorizer.transform(X_train)
test_char_features = char_vectorizer.transform(X_test)

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

Y_test = val_df.loc[:,labels]
print(test_features.get_shape(),Y_test.shape)

def one_hot_encod(index):
    l = [1, 0, -1, -2]    
    return l[index]
score=[]
test_score=[]
S = [1, 0, -1, -2]
for class_name in labels:
    train_target = Y_train[class_name].values.tolist()
    train_target = (np.array(train_target)[:,None] == S).astype(np.int)
    test_target = Y_test[class_name].values.tolist()
#     test_target = (np.array(test_target)[:,None] == [1, 0, -1, -2]).astype(np.int)
    
    scores = []
    test = []
    #test_scores=[]
    
    for i in ([0,1,2,3]):
        
        if os.path.exists(str(class_name)+'_'+str(S[i])+'_clf.pkl'):
            classifier = joblib.load(str(class_name)+'_'+str(S[i])+'_clf.pkl')
        else:
            Y = train_target[:,i]
            classifier = LogisticRegression(C=0.1, solver='sag')
            tmp = cross_val_score(classifier, train_features, Y, cv=3, scoring='roc_auc')
            cv_score = np.mean(tmp)
            scores.append(cv_score)       

            classifier.fit(train_features, Y)     
            joblib.dump(classifier, str(class_name)+'_'+str(S[i])+'_clf.pkl')

        Y_test_hat = classifier.predict_proba(test_features)[:,1]
        test.append(Y_test_hat)        
        
  
    Y_test_label = np.argmax(np.array(test),axis=0)
    test_label = [[1, 0, -1, -2][r] for r in Y_test_label]
    accuracy = np.sum((np.array(test_target) ==  np.array(test_label)))/len(test_target)
    print('【Train】 CV score for class {} is {}'.format(class_name, np.mean(scores)))
    print('【Test】 CV score for class {} is {}'.format(class_name, accuracy))
    score.append(scores)
    test_score.append(accuracy)
# print('【Train】 Total CV score is {}'.format(np.mean(score)))
print('【Test】 Total CV score is {}'.format(np.mean(test_score)))






  
  
  
  
  
  
  
  
  
  
  
    
 
 
 
 
