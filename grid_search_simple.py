import pandas as pd
from preparing_data_set import dataset_preparation
from building_model import build_model_cnn_lstm
import numpy as np
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV




# df=pd.read_excel()



df= pd.read_excel('./InputData/BHR CC_h_20170501-20190704.xlsx')

cnn_filters = [ 20, 40 , 60, 80]
dense_neur = [ 40, 60, 80, 100]


x_tr,x_test,y_tr,y_ts = dataset_preparation(df)

def model_configs():
    # define scope of configs
    n_epoch = [24,24,24,24]
    cnn_filters = [32,36,40,48,56,64,73]
    
    dense_neur = [30,34,42,48,56,60,72,80]
    
    configs = list()
    for i in n_epoch:
        
        for j in cnn_filters:
            
            for k in dense_neur:
                
                cfg = [i, j, k]
                configs.append(cfg)
    
    print('Total configs: %d' % len(configs))
    return configs

confs= model_configs()


def test_score(model):
    
    score = np.mean((abs(model.predict(x_test[0::7]).reshape(16,7) - y_ts[0::7])))
    score1 = np.median((abs(model.predict(x_test[0::7]).reshape(16,7) - y_ts[0::7])))
    return [score,score1]

score_list = []

for conf in confs:
    model_name= 'Bahreain' + str(conf) + '.h5'
    
    model = build_model_cnn_lstm(x_tr,y_tr,conf)
    
    score = test_score(model)
    
    score_list.append((model_name,score[0],score[1]))
    
    
    model.save(model_name)

df_score = pd.DataFrame(score_list,columns=['ModelName','ScoreMean','ScoreMedian'])
df_score.to_csv('score_of_models.csv',index=False)  

