import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from calculate_SMV import splitting 

def dataset_preparation(df):
    
    """
    For given DataFrame, it returns x_train,x_test,y_train,y_test
    x_train.shape = [_,7,4] and y_train.shape=[_,7]
    """
    df = df.reset_index().rename(index=str, columns={'Unnamed: 0':'TimeStamp'})
    
    shop_name = df.columns[df.columns.str.contains(" COGS")][0].replace("COGS","")
    
    df.columns = df.columns.str.replace(shop_name,"")
    
    df['TimeStamp']=pd.to_datetime(df['TimeStamp'])
    df1 = df[['TimeStamp','Traffic']].copy()
    
    df1['Year']=df1.TimeStamp.dt.year
    df1['Month']=df1.TimeStamp.dt.month
    df1['Week']=df1.TimeStamp.dt.week
    df1['Day'] = df1.TimeStamp.dt.weekday
    df1['Date']= df1['TimeStamp'].dt.date
    
    df1['Week']=df1['Week']*10
    
    df1['Day']=(df1['Day']+1)*100
    
    df2 = df1.groupby(['Year','Month','Week','Day','Date']).apply(np.sum)[['Traffic']].sort_values(by='Date').reset_index().copy()
    print(df2)
    for ind, _ in df2[df2['Traffic']==0].iterrows():
        range1 = range(ind-7,ind)
        
        df2.at[ind,'Traffic'] = df2.iloc[range1]['Traffic'].mean()
        
    df2['IsHoliday'] = 0
    
    
    x = df2[['Date','Year','Month','Week','Day','IsHoliday','Traffic']]
    
    x.at[x['Traffic']>2.4*x['Traffic'].mean(),'IsHoliday']=x['Traffic'].max()
    
    print(x['Traffic'].mean())
    
    training, testing, _, _ = train_test_split(x,x,train_size=0.8,shuffle=False)
    
    x_train,y_train = splitting(training)
    
    x_test,y_test = splitting(testing)
    
    return x_train,x_test,y_train,y_test
    
    # now, there is a final step!!! To convert these dataframes into correctly
    # shaped 
    
    
    
    
    
    
    
    
    
    