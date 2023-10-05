# %% 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

MAX_Length = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SOS_token = 0
EOS_token = 1
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
        
    def forward(self,inputs,target_tensor=None):
        '''
        Teaching force when target tensor is availble
        Teaching force is Feeding the target tensor to the next input of lstm  
        '''
    
        encoder_output,encoder_hidden = self.encoder(inputs) #Encoder part
        
        #Decoder
        
        batch_size = encoder_output.size(0)
        decoder_input  = torch.empty(batch_size,1,dtype=torch.long,device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        
        for i in range(MAX_Length):
            decoder_output,decoder_hidden  = self.decoder(decoder_input,decoder_hidden)
            decoder_outputs.append(decoder_output)
            
            if target_tensor is not None:
                decoder_input = target_tensor[:,i].unsqueeze(1)
            else:
                _,topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()
                
        decoder_outputs = torch.cat(decoder_outputs,dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs,dim=-1)
        
        return decoder_outputs,decoder_hidden
        
                
                                                            
                                                            
            
# %%
