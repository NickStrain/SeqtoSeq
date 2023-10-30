# %%
from utils.data_process1 import prepareData
from utils.data_pipline import get_dataloader,train_epoch
#from src.model import SeqtoSeq,Decoder,Encoder
from src.model import Seq2Seq,Decoder,Encoder

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# hidden_size = 512
batch_size = 32
# n_layers = 2
# embedding_size = 256
eng_lang,ta_lang,pair_eng,pair_ta  = prepareData("eng","ta")
# dropout = 0.5
# input_size = eng_lang.n_words
# output_size = ta_lang.n_words


# model = SeqtoSeq(input_size,embedding_size,hidden_size,n_layers,output_size,dropout)
INPUT_DIM = eng_lang.n_words
OUTPUT_DIM = ta_lang.n_words
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)

input_lang,output_lang,train_loader = get_dataloader(batch_size,eng_lang,ta_lang,pair_eng,pair_ta)

optimizer = optim.Adam(model.parameters(),lr=0.001)
#loss_fn = nn.CrossEntropyLoss()

# train_epoch(train_loader,model,)

def train(train_loader,model,epochs):
    loss_fn = nn.CrossEntropyLoss()
    
    for e in range(epochs):
        loss = train_epoch(train_loader,model,optimizer,loss_fn)
        print(f"Loss:{loss}")
        
train(train_loader,model,1)

# def train(model,train_loader,optimizer,loss_fn):
#     model.train()
    
#     for i,()


