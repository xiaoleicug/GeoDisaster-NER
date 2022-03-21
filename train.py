import os
import torch
import numpy as np
import logging
import torch.nn as nn
from tqdm import tqdm

import config
from model import BertNER
from metrics import f1_score, f1_score_overlapping, bad_case, get_entities, get_entities_name
from transformers import BertTokenizer


def train_epoch(train_loader, model, optimizer, scheduler,epoch):
    # set model to training mode
    model.train()
    # step number in one epoch: 336
    train_losses = 0
    for idx, batch_samples in enumerate(tqdm(train_loader)):
        batch_data, batch_token_starts, batch_labels = batch_samples
        batch_masks = batch_data.gt(0)  # get padding mask
        # compute model output and loss
        loss = model((batch_data, batch_token_starts),
                     token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)[0]
        train_losses += loss.item()
        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()
        loss.backward()
        # gradient clipping 如果所有参数的gradient组成的向量的L2 norm 大于max norm，那么需要根据L2 norm/max_norm 进行缩放。从而使得L2 norm 小于预设的 clip_norm
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
        # performs updates using calculated gradients
        optimizer.step()
        scheduler.step()
    train_loss = float(train_losses) / len(train_loader)
    logging.info("Epoch: {}, train loss: {}".format(epoch, train_loss))


def train(train_loader, dev_loader, model, optimizer, scheduler, model_dir):
    """train the model and test model performance"""
    # reload weights from restore_dir if specified
    if model_dir is not None and config.load_before:
        model = BertNER.from_pretrained(model_dir)
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(model_dir))
    #best_val_f1 = config.best_val_f1  #Load best_val_f1 from config
    best_val_f1=0.0
    patience_counter = 0
    # start training
    for epoch in range(1, config.epoch_num + 1):
        train_epoch(train_loader, model, optimizer, scheduler, epoch)
        val_metrics = evaluate(dev_loader, model, mode='dev')
        val_loss=val_metrics['loss']
        val_f1 = val_metrics['f1']
        logging.info("Epoch: {}, dev loss: {}, f1 score: {}".format(epoch, val_metrics['loss'], val_f1))
        '''if val_loss < config.best_val_loss:
            config.best_val_loss=val_loss
            model.save_pretrained(model_dir)
            logging.info("--------Save best model!--------")
        '''
        improve_f1 = val_f1 - best_val_f1
        if improve_f1 > 1e-5:
            best_val_f1 = val_f1
            model.save_pretrained(model_dir)
            logging.info("--------Save best model!--------")
            if improve_f1 < config.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1
        # Early stopping and logging best f1
        if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num:
            logging.info("Best val f1: {}".format(best_val_f1))
            break

    logging.info("Training Finished!")
    config.best_val_f1=best_val_f1  #save best_val_f1 to config

def evaluate(dev_loader, model, mode='dev'):
    # set model to evaluation mode
    model.eval()
    if mode == 'test':
        tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True, skip_special_tokens=True)
    id2label = config.id2label
    true_tags = []
    pred_tags = []
    sent_data = []
    dev_losses = 0

    with torch.no_grad():
        for idx, batch_samples in enumerate(dev_loader):
            batch_data, batch_token_starts, batch_tags = batch_samples
            if mode == 'test':
                sent_data.extend([[tokenizer.convert_ids_to_tokens(idx.item()) for idx in indices
                                   if (idx.item() > 0 and idx.item() != 101)] for indices in batch_data])
            batch_masks = batch_data.gt(0)  # get padding mask, gt(x): get index greater than x
            label_masks = batch_tags.gt(-1)  # get padding mask, gt(x): get index greater than x
            # compute model output and loss
            loss = model((batch_data, batch_token_starts),
                         token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)[0]
            dev_losses += loss.item()
            # (batch_size, max_len, num_labels)
            batch_output = model((batch_data, batch_token_starts),
                                 token_type_ids=None, attention_mask=batch_masks)[0]
            # (batch_size, max_len - padding_label_len)
            batch_output = model.crf.decode(batch_output, mask=label_masks)
            # (batch_size, max_len)
            batch_tags = batch_tags.to('cpu').numpy()
            pred_tags.extend([[id2label.get(idx) for idx in indices] for indices in batch_output])
            # (batch_size, max_len - padding_label_len)
            true_tags.extend([[id2label.get(idx) for idx in indices if idx > -1] for indices in batch_tags])

    assert len(pred_tags) == len(true_tags)
    if mode == 'test':
        assert len(sent_data) == len(true_tags)

    # logging loss, f1 and report
    metrics = {}
    if mode == 'dev':
        # f1 = f1_score(true_tags, pred_tags, mode)
        score = f1_score(true_tags, pred_tags, mode)
        metrics['f1'] = score['f1']
        metrics['p'] = score['p']
        metrics['r'] = score['r']
    else:
        bad_case(true_tags, pred_tags, sent_data)
        if not os.path.exists(config.case_dir):
            os.system(r"echo test {}".format(config.case_dir))  # 调用系统命令行来创建文件
        output = open(config.test_result_dir, 'w')
        for idx, (t, p) in enumerate(zip(true_tags, pred_tags)):
            output.write("Test case " + str(idx) + ": \n")
            output.write("sentence: " + str(sent_data[idx]) + "\n")
            # output.write("golden label: " + str(t) + "\n")
            # output.write("model pred: " + str(p) + "\n")
            output.write(str(get_entities(p)) + "\n")
            # output.write(str(get_entities_name(get_entities(p)), sent_data[idx]))
        '''print(get_entities(pred_tags))
        entities = get_entities_name(get_entities(pred_tags), sent_data)
        print(entities)'''
        # f1_labels, f1 = f1_score(true_tags, pred_tags, mode)
        label_score, score = f1_score(true_tags, pred_tags, mode)
        metrics['label_score'] = label_score
        metrics['f1'] = score['f1']
        metrics['p'] = score['p']
        metrics['r'] = score['r']

        label_score_op, score_op = f1_score_overlapping(true_tags, pred_tags, mode)
        metrics['label_score_op'] = label_score_op
        metrics['f1_op'] = score_op['f1']
        metrics['p_op'] = score_op['p']
        metrics['r_op'] = score_op['r']


    # metrics['f1_labels'] = f1_labels
    # metrics['f1'] = f1
    metrics['loss'] = float(dev_losses) / len(dev_loader)
    return metrics

def predict(sentences,model):
    tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True, skip_special_tokens=True)
    id2label = config.id2label
    pred_tags = []
    
    sent_data = []
    
    words = []
    #word_lens = []
    for sentence in sentences:
        #for word in sentence:
            #word_lens.append(len(word))
        words = ['[CLS]'] + [item for token in sentence for item in token]
        #token_start_idxs = 1 + np.cumsum([0] + word_lens[:-1])
        sent_data.append(tokenizer.convert_tokens_to_ids(words))
        
    pred_scores=[]
    with torch.no_grad():
        for sent in sent_data:
            t=torch.tensor(sent,dtype=torch.long).unsqueeze(0) # [1,57]
            token_starts=np.ones(len(sent))
            token_starts[0]=0
            token_starts=torch.tensor(token_starts,dtype=torch.long).unsqueeze(0) #[1,57]
            t, token_starts = t.to(config.device), token_starts.to(config.device)
            emissons=model((t,token_starts))[0] #(1,56,10)
            #pred_score.extend([[indices] for indices in score]) #[[56,10],[54,10]]
            pred_tag=model.crf.decode(emissons) #[[56]]
            output = torch.tensor(pred_tag, dtype=torch.long).to(config.device)
            #log_likelihood=model.crf.forward(emissons,output,reduction='none')
            log_likelihood=model.crf(emissons,output)
            pred_score=log_likelihood.exp().item()
            #for index,idx in enumerate(output[0]):
                #s=log_likelihood.exp()
                #pred_score+=s
            #pred_score=float(pred_score)/len(sent)
            pred_scores.append(pred_score)
            
            pred_tags.extend([[id2label.get(idx) for idx in indices] for indices in pred_tag])
    
    
            
        
    return pred_tags, pred_scores        
        
    
        
    
   
        
        

if __name__ == "__main__":
    a = [101, 679, 6814, 8024, 517, 2208, 3360, 2208, 1957, 518, 7027, 4638,
         1957, 4028, 1447, 3683, 6772, 4023, 778, 8024, 6844, 1394, 3173, 4495,
         807, 4638, 6225, 830, 5408, 8024, 5445, 3300, 1126, 1767, 3289, 3471,
         4413, 4638, 2767, 738, 976, 4638, 3683, 6772, 1962, 511, 0, 0,
         0, 0, 0]
    t = torch.tensor(a, dtype=torch.long)
    tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True, skip_special_tokens=True)
    word = tokenizer.convert_ids_to_tokens(t[1].item())
    sent = tokenizer.decode(t.tolist())
    print(word)
    print(sent)
    
