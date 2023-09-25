# %% 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 


# %% 
class Encoder(nn.Module):
    '''
    The encoder part of the seqtoseq model
    '''
    def __init__(self,input_size,hidden_size,dropout_value=0.1):
        super(Encoder,self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size,hidden_size)
        self.lstm = nn.LSTM(hidden_size,hidden_size)
        self.dropout = nn.Dropout(dropout_value)
    
    def forward(self,x):
        x = self.dropout(self.embedding(x))
        output,hidden = self.lstm(x)
        
        return output,hidden
    
class Decoder(nn.Module):
    '''
    The decoder part of the seqtoseq model
    Without attention...;
    '''
    def __init__(self,hidden_size,output_size):
        super(Decoder,self).__init__()
        self.embedding = nn.Embedding(output_size,hidden_size)
        self.lstm = nn.LSTM(hidden_size,hidden_size)
        self.out = nn.Linear(hidden_size,output_size)
        
    
    def forward(self,x,hidden):
        x = self.embedding(x)
        x = F.relu(x)
        x,hidden= self.lstm(x,hidden)
        x = self.out(x)
        
        return x,hidden
    

class SeqtoSeq(nn.Module):
    '''
    These part is to combine the encoder and decoder
    and this is SeqtoSeq without attention  
    '''
    def __init__(self,encoder,decoder):
        super(SeqtoSeq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward()
    
            