#%%
import pandas as pd 
import os 
location_path = 'E:\SeqtoSeq\data\process_english_data\\en'
#%%
def tamil_data_process():
    '''
    This function is to split the tamil data into small csv file 
    Total number of line in ta.txt file is 31542969
    '''
    df = pd.DataFrame(columns=['text'],index=list(range(0,6500000)))
  
    file_no = 0
    index=0
    with open('E:\SeqtoSeq\data\\tamil\\ta.txt',encoding='utf8') as data:     
        for i,line in enumerate(data):
            
            df["text"][index] = [line]
            index+=1
        
            if (i%6500000==0 and i!=1):
                print(df)
                df.to_csv(location_path+str(file_no)+".csv")
                df = pd.DataFrame(columns=['text'],index=list(range(0,6500000)))
                print(f"Steps :{file_no} finished")
                file_no+=1
                index=0
        
    print(file_no)
# %%

def english_data_process():
    '''
    This function is to split tne english data into small size data csv file 
    Total number of line in en.txt is 54250995
    '''
    df = pd.DataFrame(columns=['text'],index=list(range(0,14000000)))
  
    file_no = 0
    index=0
    with open('E:\SeqtoSeq\data\english\en.txt',encoding='utf8') as data:     
        for i,line in enumerate(data):
            
            df["text"][index] = [line]
            index+=1
        
            if (i%14000000==0 and i!=1):
                print(df)
                df.to_csv(location_path+str(file_no)+".csv")
                df = pd.DataFrame(columns=['text'],index=list(range(0,14000000)))
                print(f"Steps :{file_no} finished")
                file_no+=1
                index=0
        
    print(index)
english_data_process()
