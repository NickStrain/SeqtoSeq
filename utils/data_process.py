import pandas as pd 
import os 
df = pd.DataFrame(columns=['text'],index=list(range(0,500000)))
location_path = 'E:\SeqtoSeq\data\process_tamil_data\\'
file_no = 0
index=0
with open('E:\SeqtoSeq\data\\tamil\\ta.txt',encoding='utf8') as data:     
    for i,line in enumerate(data):
        
        df["text"][index] = [line]
        index+=1
        
        if (i%500000==0):
            print(df)
            df.to_csv(location_path+str(file_no)+".csv")
            df = pd.DataFrame(columns=['text'],index=list(range(0,500000)))
            print(f"Steps :{file_no}")
            file_no+=1
            index=0
        

