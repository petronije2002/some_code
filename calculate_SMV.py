import pandas as pd
import numpy as np
import datetime


def calculate_SMV(df): # provide DataFrame as an input of the function
    
    # Extract the shop name, and remove it from all column names
    
    shop_name = df.columns[df.columns.str.contains("COGS")][0].replace("COGS","")
    
    df.columns = df.columns.str.replace(shop_name,"")
    
    # Rename column index, to 'TimeStampe
    df=df.rename(index=str, columns={'index':"TimeStamp"})

    df=df[['TimeStamp','Traffic','Conversion Rate','Staff Hours','Staff Minutes per Visitor','Amount Sold']]


    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    df['Week'] = df['TimeStamp'].dt.week
    df['Year']=df['TimeStamp'].dt.year
    df['Month']=df['TimeStamp'].dt.month
    df['Date'] = df['TimeStamp'].dt.date
    df['Day'] = df['TimeStamp'].dt.weekday

    df =df.fillna(0)

    df.head()

    # There is a mistake in DATA!!! Staff Minutes Per Visitor need to be
    # recalculated in minutes !!!!
    
    df['Staff Minutes per Visitor'] = df["Staff Minutes per Visitor"]/60
    
    # Here we take only those hours when the shop was open
    
    df = df[(df['TimeStamp'].dt.time>=datetime.time(10,0))&(df['TimeStamp'].dt.time<=datetime.time(23,59))]
    
    # In the DataSet, there are some record where the shop was open, and 
    # Staff Hours were zero, or less than 1...that is technically impossible
    
    df = df[df['Staff Hours']>0]

    df1 = df.copy()
    
    #Take quantile of 95% of records, in terms of Traffic ( eliminate records,
    # with 'crazy' traffic 
    
    traf_limit = df1['Traffic'].quantile(0.95)
    
    df2 = df1[(df1['Traffic']<traf_limit)].copy()
    
    # Remove records with negative revenues ( returns ), and take only hours
    # with staff hours >= 1 ( at least one staffer had to be in 'open shop')
    
    
    df2 = df2[df2['Amount Sold']>=0].copy()
    df2 = df2[df2['Staff Hours']>=1].copy()
    
    
    # Look at the Staff Minute Per Visito!! there are some skew! 
    # Again, we will take quantile with 85% ( it is from experience)
    
    
    SMV_limit = df2['Staff Minutes per Visitor'].quantile(0.85)
    
    df2 = df2[df2['Staff Minutes per Visitor']<SMV_limit]
    
    # Now we need to calculate OPTIMALITY...It is coeff describing how succesfull
    # staff performed 
    
    df2['Optim'] = df2['Conversion Rate'] * df2['Amount Sold']/df2['Staff Hours']
    
    # We will take quantile of 95% of that coeff. ( To eliminate 'crazy' values')
    
    limes_optim = df2['Optim'].quantile(0.9)
    
    df2=df2[(df2['Optim']<limes_optim)].copy()
    
    
    # Now, when we have cleaned our data, we need to pay attention of 
    # 'better' part of dataset. It meanse we will take 25% of dataset with 
    #highest values of OPTIMALITY. 
    
    description =df2.describe()
    OPTIM_LIMES= description.loc['75%','Optim']
    df3 = df2[df2['Optim']>OPTIM_LIMES].copy()
    
    description3 = df3.describe()
    
    OPTIM_LIMES= description3.loc['50%','Staff Minutes per Visitor']
    
    return OPTIM_LIMES, description3\

   
def splitting(data_set):
    _inputs = list()
    _outputs = list()
    
    #here we reset index, since in the testing index will starto from 628...!!!
    data_set1 = data_set.reset_index().copy()
    
    try:
        for ind,row in data_set1.iterrows():
            end_ind = ind + 7

#             week_in = data_set1.iloc[ind:end_ind][['WeekNorm','DayNorm', 'Dubai Mall Traffic Norm']].values

            #take the traffic for previous 7 days
    
            week_in_traffic = data_set1.iloc[ind:end_ind][['Traffic']].values
    
            #these are values we expect our model to predict
        
            week_out = data_set1.iloc[end_ind:end_ind+7]['Traffic'].values
            
            #check if there is gonna be some holiday during a next week
            
            is_holiday = data_set1.iloc[end_ind:end_ind+7][['IsHoliday']].values
            
            #what will be week days 
            
            daynorm = data_set1.iloc[end_ind:end_ind+7][['Day']].values
            
            #what will be week number
            
            weeknorm = data_set1.iloc[end_ind:end_ind+7][['Week']].values
            
            
            #find
            
            week_in=np.concatenate((week_in_traffic,is_holiday,daynorm,weeknorm,), axis=1)
            
            if (week_in.shape[0]<7) | (week_out.shape[0]<7) :

                break



            else:
                _inputs.append(week_in)
                _outputs.append(week_out)


    except:




        pass

    finally:
        inputs= np.stack( _inputs, axis=0 )
        outputs= np.stack( _outputs,axis=0)
        
        return inputs, outputs
    
    
    
    
    
    
    

