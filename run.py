import utils
import config
import logging
import numpy as np
import torch
from data_process import Processor
from data_loader import NERDataset
from model import BertNER
from train import train, evaluate, predict


from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers.optimization import get_linear_schedule_with_warmup, AdamW

from collections import Counter

import warnings

warnings.filterwarnings('ignore')


def dev_split(dataset_dir):
    """split dev set"""
    data = np.load(dataset_dir, allow_pickle=True)
    words = data["words"]
    labels = data["labels"]
    x_train, x_dev, y_train, y_dev = train_test_split(words, labels, test_size=config.dev_split_size, random_state=0)
    return x_train, x_dev, y_train, y_dev


def test():
    data = np.load(config.test_dir, allow_pickle=True)
    word_test = data["words"]
    label_test = data["labels"]
    test_dataset = NERDataset(word_test, label_test, config)
    logging.info("--------Dataset Build!--------")
    # build data_loader
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size,
                             shuffle=False, collate_fn=test_dataset.collate_fn)
    logging.info("--------Get Data-loader!--------")
    # Prepare model
    if config.model_dir is not None:
        model = BertNER.from_pretrained(config.model_dir)
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(config.model_dir))
    else:
        logging.info("--------No model to test !--------")
        return

    val_metrics = evaluate(test_loader, model, mode='test')
    val_f1 = val_metrics['f1']
    val_p = val_metrics['p']
    val_r = val_metrics['r']
    val_f1_op = val_metrics['f1_op']
    val_p_op = val_metrics['p_op']
    val_r_op = val_metrics['r_op']
    logging.info("test loss: {}, f1 score: {}, p: {}, r:{}".format(val_metrics['loss'], val_f1, val_p, val_r))
    logging.info("Overlapping f1 score: {}, p: {}, r:{}".format(val_f1_op, val_p_op, val_r_op))
    val_score_labels = val_metrics['label_score']
    val_score_labels_op = val_metrics['label_score_op']
    for label in config.labels:
        logging.info("f1 score of {}: {}, overlapping f1 score :{} ".format(label, val_score_labels[label]['f1'], val_score_labels_op[label]['f1']))
        logging.info("p of {}: {}, overlapping p :{}".format(label, val_score_labels[label]['p'], val_score_labels_op[label]['p']))
        logging.info("r of {}: {}, overlapping r :{}".format(label, val_score_labels[label]['r'], val_score_labels_op[label]['r']))


def teacher_unlabel():
    data = np.load(config.unlabel_dir, allow_pickle=True)
    words_unlabel = data["words"]
    # label_unlabel=data["labels"]
    # unlabel_dataset=NERDataset(word_unlable,label_unlabel,config)
    logging.info("--------Dataset Build!--------")

    # Prepare model
    if config.model_dir is not None:
        model = BertNER.from_pretrained(config.model_dir)
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(config.model_dir))
    else:
        logging.info("--------No model to get the soft labels of the unlabeled data !--------")
        return
    labels_unlabel, unlabel_score = predict(words_unlabel, model)
    return words_unlabel, labels_unlabel, unlabel_score


def self_training(words_L, labels_L, words_U, labels_U, scores_U):
    words_U = words_U.tolist()
    words_L = words_L.tolist()
    labels_L = labels_L.tolist()

    words = []
    labels = []

    words.extend(words_L)
    labels.extend(labels_L)

    # print(words_U)
    del_w = []
    del_l = []
    for idx, score in enumerate(scores_U):
        len_tag_O = Counter(labels_U[idx])['O']
        if len_tag_O < len(labels_U[idx]):
            if score >= 0.95:
                word = words_U[idx]
                label = labels_U[idx]
                words.append(words_U[idx])
                labels.append(labels_U[idx])
                del_w.append(word)
                del_l.append(label)
                # words_U.remove(words_U[idx])
                # labels_U.remove(labels_U[idx])
    for d_w in del_w:
        words_U.remove(d_w)
    for d_l in del_l:
        labels_U.remove(d_l)

    words = np.array(words)
    labels = np.array(labels)

    # 分离出新的训练集和验证集
    word_train, word_dev, label_train, label_dev = train_test_split(words, labels, test_size=config.dev_split_size,
                                                                    random_state=0)
    # word_train = words
    # label_train = labels
    # data = np.load(config.dev_dir, allow_pickle=True)
    # word_dev = data['words']
    # label_dev = data['labels']

    train_dataset = NERDataset(word_train, label_train, config)
    dev_dataset = NERDataset(word_dev, label_dev, config)
    logging.info("--------Dataset Build!--------")
    train_size = len(train_dataset)
    print(train_size)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=True, collate_fn=dev_dataset.collate_fn)
    logging.info("--------Get Dataloader!--------")

    device = config.device
    # model = BertNER.from_pretrained(config.roberta_model, num_labels=len(config.label2id))
    '''model = BertNER.from_pretrained(config.bert_model, num_labels=len(config.label2id))
    model.to(device)
    

    # Train the model
    logging.info("--------Start Training!--------")
    train(train_loader, dev_loader, model, optimizer, scheduler, config.model_dir)'''
    if config.model_dir is not None:
        model = BertNER.from_pretrained(config.model_dir)
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(config.model_dir))
    train_steps_per_epoch = train_size // config.batch_size
    # Prepare optimizer
    if config.full_fine_tuning:
        # model.named_parameters(): [bert, bilstm, classifier, crf]
        bert_optimizer = list(model.bert.named_parameters())
        lstm_optimizer = list(model.bilstm.named_parameters())
        classifier_optimizer = list(model.classifier.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in lstm_optimizer if not any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 10, 'weight_decay': config.weight_decay},
            {'params': [p for n, p in lstm_optimizer if any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 10, 'weight_decay': 0.0},
            {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 100, 'weight_decay': config.weight_decay},
            {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 100, 'weight_decay': 0.0},
            {'params': model.crf.parameters(), 'lr': config.learning_rate * 5}
        ]
    # only fine-tune the head classifier
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, correct_bias=False)
    train_steps_per_epoch = train_size // config.batch_size
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=(config.epoch_num // 10) * train_steps_per_epoch,
                                                num_training_steps=config.epoch_num * train_steps_per_epoch)
    logging.info("--------Start Training!--------")
    train(train_loader, dev_loader, model, optimizer, scheduler, config.model_dir)
    #train(train_loader, dev_loader, model, optimizer, config.model_dir)
    words_U = np.array(words_U)

    return words, labels, words_U


def load_dev(mode):
    if mode == 'train':
        # 分离出验证集
        word_train, word_dev, label_train, label_dev = dev_split(config.train_dir)
        #np.savez_compressed(config.train_dir, words=word_train, labels=label_train)
        #np.savez_compressed(config.dev_dir, words=word_dev, labels=label_dev)
    elif mode == 'test':
        train_data = np.load(config.train_dir, allow_pickle=True)
        dev_data = np.load(config.test_dir, allow_pickle=True)
        word_train = train_data["words"]
        label_train = train_data["labels"]
        word_dev = dev_data["words"]
        label_dev = dev_data["labels"]
    else:
        word_train = None
        label_train = None
        word_dev = None
        label_dev = None
    return word_train, word_dev, label_train, label_dev


def run():
    """train the model"""
    # set the logger
    # utils.set_logger(config.log_dir)
    logging.info("device: {}".format(config.device))
    # 处理数据，分离文本和标签

    # processor = Processor(config)
    # processor.process()
    # logging.info("--------Process Done!--------")
    # 分离出验证集
    word_train, word_dev, label_train, label_dev = load_dev('train')
    # build dataset

    train_size = len(word_train)   #325
    print(train_size)
    train_dataset = NERDataset(word_train[0:train_size], label_train[0:train_size], config)
    dev_dataset = NERDataset(word_dev, label_dev, config)
    logging.info("--------Dataset Build!--------")
    # get dataset size
    train_size = len(train_dataset)
    # build data_loader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=True, collate_fn=dev_dataset.collate_fn)
    logging.info("--------Get Dataloader!--------")
    # Prepare model
    device = config.device
    # model = BertNER.from_pretrained(config.roberta_model, num_labels=len(config.label2id))
    model = BertNER.from_pretrained(config.bert_model, num_labels=len(config.label2id))
    model.to(device)
    # Prepare optimizer
    if config.full_fine_tuning:
        # model.named_parameters(): [bert, bilstm, classifier, crf]
        bert_optimizer = list(model.bert.named_parameters())
        lstm_optimizer = list(model.bilstm.named_parameters())
        classifier_optimizer = list(model.classifier.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in lstm_optimizer if not any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 10, 'weight_decay': config.weight_decay},
            {'params': [p for n, p in lstm_optimizer if any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 10, 'weight_decay': 0.0},
            {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 100, 'weight_decay': config.weight_decay},
            {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 100, 'weight_decay': 0.0},
            {'params': model.crf.parameters(), 'lr': config.learning_rate * 5}
        ]
    # only fine-tune the head classifier
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, correct_bias=False)
    train_steps_per_epoch = train_size // config.batch_size
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=(config.epoch_num // 10) * train_steps_per_epoch,
                                                num_training_steps=config.epoch_num * train_steps_per_epoch)

    # Train the model
    logging.info("--------Start Training!--------")

    train(train_loader, dev_loader, model, optimizer, scheduler, config.model_dir)

    # logging.info("--------Start Self-Training!--------")
    # config.load_before=True
    # self_training(optimizer, scheduler)


if __name__ == '__main__':
    utils.set_logger(config.log_dir)
    processor = Processor(config)
    # processor.process()
    # logging.info("--------Process Done!--------")  #处理label数据

    '''data = np.load(config.label_dir, allow_pickle=True)
    words=data["words"]
    labels=data["labels"]
    
    #分离label数据集得到新的训练集和测试集
    word_train, word_test, label_train, label_test = train_test_split(words, labels, test_size=config.test_split_size, random_state=0)
    print(len(word_train)) # label = train + dev
    print(len(word_test))  # test
    
    np.savez_compressed(config.train_dir, words=word_train, labels=label_train)
    logging.info("--------{} data process DONE!--------".format('train'))    #247
    
    np.savez_compressed(config.test_dir, words=word_test, labels=label_test)
    logging.info("--------{} data process DONE!--------".format('test'))   #248'''

    '''
    #在原来的训练集中添加新标的数据作为新的训练集，测试集不变
    data1 = np.load(config.label_plus_dir, allow_pickle=True)
    data2 = np.load(config.train_dir, allow_pickle = True)
    print(len(data2["words"]))  #133
    words = np.concatenate((data1["words"],data2["words"]),axis=0)
    labels = np.concatenate((data1["labels"],data2["labels"]),axis=0)

    np.savez_compressed(config.train_dir, words= words, labels=labels)
    logging.info("--------{} data process DONE!--------".format('New train'))
    data2 = np.load(config.train_dir, allow_pickle=True)
    print(len(data2["words"])) #362
    
    data3 = np.load(config.test_dir, allow_pickle = True)
    print(len(data3["words"])) #133 
    '''


    run()  # 训练一个teacher model
    test()

    data = np.load(config.train_dir, allow_pickle=True)
    words_L = data["words"]
    labels_L = data["labels"]

    data = np.load(config.unlabel_dir, allow_pickle=True)
    words_unlabel = data["words"]
    # label_unlabel=data["labels"]
    # unlabel_dataset=NERDataset(word_unlable,label_unlabel,config)
    logging.info("--------Load unlabel words for dir! print length as follows--------")
    print(len(words_unlabel))
    logging.info("--------Get soft label of unlabel data--------")
    # words_U,labels_U,scores_U=teacher_unlabel()
    if config.model_dir is not None:
        model = BertNER.from_pretrained(config.model_dir)
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(config.model_dir))
    else:
        logging.info("--------No model to get the soft labels of the unlabeled data !--------")
        exit(0)

    labels_U, scores_U = predict(words_unlabel, model)
    logging.info("--------Label the unlabel data finished!--------")
    logging.info("length of unlabel data: {}".format(len(words_unlabel)))
    logging.info("--------self_training!--------")
    train_len = len(words_L)
    words_L, labels_L, words_U = self_training(words_L[0:train_len], labels_L[0:train_len], words_unlabel, labels_U, scores_U)
    # torch.cuda.empty_cache()
    logging.info("--------self_training finished!--------")
    test()
    logging.info("The first test finished!--------")
    L0 = len(words_U)
    count = 1
    for self_epoch in range(1, 5):
        logging.info("length of unlabel data: {}".format(len(words_U)))
        # words_U,labels_U,scores_U=teacher_unlabel()
        model = BertNER.from_pretrained(config.model_dir)
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(config.model_dir))
        logging.info("--------Get soft label of unlabel data--------")
        labels_U, scores_U = predict(words_U, model)
        logging.info("--------Label the unlabel data finished!--------")
        logging.info("--------self_training!--------")
        words_L, labels_L, words_U = self_training(words_L, labels_L, words_U, labels_U, scores_U)
        # torch.cuda.empty_cache()
        count=count+1
        logging.info("--------self_training finished!--------")
        test()
        logging.info("The {} test finished!--------".format(count))
        if len(words_U) < L0:
            L0 = len(words_U)
        else:
            logging.info("The number of self-training: {}".format(count))

            break
        if self_epoch == 4:
            logging.info("The number of self-training: {}".format(count))

