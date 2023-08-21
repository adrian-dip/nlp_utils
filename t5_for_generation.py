
INPUT_DIR = './'
OUTPUT_DIR = './synthetic_data/'

import random
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = 'google/flan-t5-large'
class config:
    TRAIN_BATCH_SIZE = 2    
    VALID_BATCH_SIZE = 2    
    TRAIN_EPOCHS = 6        
    VAL_EPOCHS = 1 
    LEARNING_RATE = 2e-4   
    SEED = 42               
    IN_LEN = 64
    OUT_LEN = 64 
    BEAMS = 8

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data['text']
        self.ctext = self.data['in']

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    losses = AverageMeter()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    for _, data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        labels = y[:, 1:].clone().detach()
        labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=labels)
        loss = outputs[0]
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        losses.update(loss.item(), config.TRAIN_BATCH_SIZE)
    return losses.avg
        
def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=config.OUT_LEN, 
                num_beams=config.BEAMS,
                repetition_penalty=3.0, 
                length_penalty=1.25, 
                early_stopping=True
                )
            
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals


def engine(df, generation_df):
    tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=config.IN_LEN)

    training_set = CustomDataset(df, tokenizer, config.IN_LEN, config.OUT_LEN)
    val_set = CustomDataset(generation_df, tokenizer, config.IN_LEN, config.OUT_LEN)

    train_params = {
        'batch_size': config.TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
        }

    val_params = {
        'batch_size': config.VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }

    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)

    optimizer = torch.optim.Adam(params =  model.parameters(), lr=config.LEARNING_RATE)

    # Training loop

    for epoch in range(config.TRAIN_EPOCHS):
        print(f'Epoch: {epoch}')
        avg_loss = train(epoch, tokenizer, model, device, training_loader, optimizer)
        print(f'Epoch: {avg_loss}')
        predictions, actual = validate(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({'generated':predictions, 'actual':actual, 'precursor':list(generation_df['in'])})
        final_df.to_csv(OUTPUT_DIR + 'es_predictions_epoch' + str(epoch) + '.csv', index=False)
        print('Output Files generated for review')


if __name__ == '__main__':
    seed_everything()
    engine(df, gen_df)


