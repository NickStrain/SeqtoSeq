import torch 
import numpy as np

from torch.utils.data import TensorDataset,DataLoader,RandomSampler

SOS_token = 0
EOS_token = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGHT  = 50 
def indextoSenctence(lang,sentence):
    return [lang.word2index[word] for word in sentence[0].split(' ')]

def tensortoSentence(lang,sentence):
    
    indexes = indextoSenctence(lang,sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes,dtype=torch.long,device=device).view(-1,1)

def tensortopair(pair,input_lang,output_lang):
    input_tensor = tensortoSentence(input_lang,pair[0])
    target_tensor = tensortoSentence(output_lang,pair[0])
    return(input_tensor,target_tensor)

def get_dataloader(batch_size,input_lang,output_lang,pair_eng,pair_ta):
    n = len(pair_eng)
    input_ids = np.zeros((n,MAX_LENGHT),dtype=np.int32)
    target_ids = np.zeros((n,MAX_LENGHT),dtype=np.int32)
    
    for idx, inp in enumerate(pair_eng):
        inp_ids = indextoSenctence(input_lang,inp)
        #tgt_ids = indextoSenctence(output_lang,tgt)
        inp_ids.append(EOS_token)
        #tgt_ids.append(EOS_token)
        
        input_ids[idx, :len(inp_ids)] = inp_ids
        
        #target_ids[idx, :len(tgt_ids)] = tgt_ids
    
    for idx, tgt in enumerate(pair_ta):
        #inp_ids = indextoSenctence(input_lang,inp)
        tgt_ids = indextoSenctence(output_lang,tgt)
        #inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
    
        #input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids  
        
    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))
    train_sampler = RandomSampler(train_data)
    
    train_dataloader = DataLoader(train_data,sampler=train_sampler,batch_size=batch_size)
    
    return input_lang,output_lang,train_dataloader


def train_epoch(dataloader,model,optimizer,loss_fn):
    total_loss = 0
    print("training...")
    for input_tensor,target_tensor in dataloader:
        optimizer.zero_grad()
        #model_output,decoder_hidden = model(input_tensor,target_tensor)
        model_output = model(input_tensor,target_tensor)
        
        loss = loss_fn(model_output.view(-1,model_output.size(-1)),target_tensor.view(-1))
        loss.backward()
        
        optimizer.step()
        
        total_loss +=loss.item()
        
    return total_loss / len(dataloader)

