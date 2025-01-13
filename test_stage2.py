# python test_stage2.py --resume output/models/Stage2_Model1/1119_153251/checkpoint-epoch10.pth --config configs/evaluation/config_stage2.json
import argparse
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import pickle
import numpy as np
import collections
from transformers import BertModel, BertTokenizer
import random
import os

import data_loader.data_loaders as module_data
import loss.combinatorial_loss as module_loss
from model.metric import MyRetrievalMetric as module_metric
import model.model as module_arch
from trainer import stage2_inference
from model.utils.utils import sim_matrix

from parse_config import ConfigParser


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


shotTypes = ['short_service', 'long_service', 'net_kill', 'net', 'push', 'drive', 'lob', 'smash', 'drop', 'clear', 'unknown']


def create_text_tokens_BERT(caption, tokenizer, embedding_model):
    encoding = tokenizer(caption, return_tensors='pt', padding='max_length', truncation=True, max_length=25, add_special_tokens=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    input_ids = input_ids.to('cuda')
    attention_mask = attention_mask.to('cuda')
    with torch.no_grad():
        outputs = embedding_model(input_ids, attention_mask=attention_mask)
        text_token = outputs.last_hidden_state  # This contains the embeddings
    text_token = text_token.squeeze(0)
    text_mask = np.ones(25, dtype=np.float32)
    text_mask = torch.from_numpy(text_mask).float()

    return text_token, text_mask


def get_class_vector(rally_length, first_shot, last_shot, most_frequent):
    cls = []
    cls.append(rally_length)
    cls.append(first_shot+100)
    cls.append(last_shot+100)
    if rally_length >= 5 and most_frequent is not None:
        cls.append(most_frequent+200)
    return cls


def main(config):
    word_embedder = BertModel.from_pretrained('bert-base-uncased')    # dim=768
    word_embedder.eval()
    word_embedder = word_embedder.to('cuda')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of metrics
    metrics = [module_metric(met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    if config['embeddings'] is None or os.path.exists(config['embeddings']) is False:
        model_output = []
        data_path = config['data_path']
        data = pickle.load(open(data_path, 'rb'))
        for rally in tqdm(data):
            batch = collections.defaultdict(lambda: [])
            description_token, description_mask = create_text_tokens_BERT(rally['description'], tokenizer, word_embedder)
            batch['description_token'].append(description_token)
            batch['description_mask'].append(description_mask)
            batch['text_token'] = None
            output = stage2_inference(model, device, batch)['data_embed'].detach().cpu().numpy()

            model_output.append({
                'id': rally['id'],
                'videoPath': rally['videoPath'],
                'service_type': rally['service_type'],
                'last_shot_type': rally['last_shot_type'],
                'rally_length': rally['rally_length'],
                'most_frequent': rally['most_frequent'],
                'class': get_class_vector(rally['rally_length'], rally['service_type'], rally['last_shot_type'], rally['most_frequent']),
                'predicted_class': get_class_vector(rally['rally_length'], rally['stage1_prediction']['service_type'], rally['stage1_prediction']['last_shot_type'], rally['stage1_prediction']['most_frequent']),
                'embed': output
            })
            del output

        with open(config['embeddings'], 'wb') as f:
            pickle.dump(model_output, f)
    else:
        model_output = pickle.load(open(config['embeddings'], 'rb'))

    embed_arr = []
    class_arr = []
    for embed in tqdm(model_output, desc='Loading embeddings to GPU'):
        embed_arr.append(torch.from_numpy(embed['embed']))
        cls = np.array(embed['class'])
        if cls.size < 4:
            cls = np.pad(cls, (0, 4 - cls.size), 'constant', constant_values=-1)
        class_arr.append(torch.from_numpy(cls))
    embed_arr = torch.cat(embed_arr).to(device)
    class_arr = torch.stack(class_arr)

    def get_captions(filename, rally_length=None):
        output = list(open(filename, 'r', encoding='utf-8').read().splitlines())
        if rally_length is not None:
            for i in range(len(output)):
                output[i] = output[i].replace('&', str(rally_length))
        return output

    def print_metadata(meta):
        return f'Rally length: {meta[0]}, Serve type: {shotTypes[meta[1]-100]}, Last shot type: {shotTypes[meta[2]-100]}, Most frequent shot: {shotTypes[meta[3]-200] if len(meta) > 3 else "Invalid"}'

    def retrieval(queries, eval_class=-1, eval_class_arr=None):
        batch = collections.defaultdict(lambda: [])
        batch['description_token'] = None
        batch['description_mask'] = None
        for q in queries:
            query_token, query_mask = create_text_tokens_BERT(q, tokenizer, word_embedder)
            batch['text_token'].append(query_token)
            batch['text_mask'].append(query_mask)
        query_embed = stage2_inference(model, device, batch)['text_embed']

        if eval_class != -1 or eval_class_arr is not None:
            sims = {}
            sims['t2d'] = sim_matrix(query_embed, embed_arr).detach().cpu().numpy()
            if eval_class_arr is None:
                eval_class_arr = torch.full((len(queries),), eval_class, dtype=torch.int32)
            for metric in metrics:
                metric_name = metric.__name__
                res = metric(sims, class_arr, eval_class_arr)
                # metrics[metric_name] = res
                logger.info(f'    {str(metric_name):15s}: {res}')
        else:
            val, idx = torch.topk(sim_matrix(query_embed, embed_arr).detach().cpu().squeeze(0), 10)
            for score, i in zip(val, idx):
                print(f'\nSimilarity score: {score:.4f}\nVideo: {model_output[i]["videoPath"]},\nGroud Truth: {{' + print_metadata(model_output[i]['class']) + '}},\nStage1 Prediction: {{' + print_metadata(model_output[i]['predicted_class']) + '}}')

        del query_embed, query_token, query_mask

    # Rally length caption
    for rally_length in range(1, 25):
        queries = get_captions('dataset/text_data/length.txt', rally_length=rally_length)
        eval_class = rally_length
        print(f'\nRally length: {rally_length}')
        retrieval(queries, eval_class)

    # Serve type caption
    queries = get_captions('dataset/text_data/short_service.txt')
    eval_class = 0+100
    print('\nServe type: Short service')
    retrieval(queries, eval_class)

    queries = get_captions('dataset/text_data/long_service.txt')
    eval_class = 1+100
    print('\nServe type: Long service')
    retrieval(queries, eval_class)

    # Last ball type caption
    for i in range(10):
        queries = get_captions(f'dataset/text_data/end_{shotTypes[i]}.txt')
        eval_class = i+100
        print(f'\nLast ball type: {shotTypes[i]}')
        retrieval(queries, eval_class)

    # Frequent ball type caption
    for i in range(2, 10):
        queries = get_captions(f'dataset/text_data/frequent_{shotTypes[i]}.txt')
        eval_class = i+200
        print(f'\nFrequent ball type: {shotTypes[i]}')
        retrieval(queries, eval_class)

    # Calculate general metric
    # Rally length caption
    queries = []
    eval_class_arr = None
    for rally_length in range(1, 15):
        tmp = get_captions('dataset/text_data/length.txt', rally_length=rally_length)
        if eval_class_arr is None:
            eval_class_arr = torch.full((len(tmp),), rally_length, dtype=torch.int32)
        else:
            eval_class_arr = torch.cat((eval_class_arr, torch.full((len(tmp),), rally_length, dtype=torch.int32)))
        queries.extend(tmp)

    # Serve type caption
    tmp = get_captions('dataset/text_data/short_service.txt')
    eval_class_arr = torch.cat((eval_class_arr, torch.full((len(tmp),), 0+100, dtype=torch.int32)))
    queries.extend(tmp)

    tmp = get_captions('dataset/text_data/long_service.txt')
    eval_class_arr = torch.cat((eval_class_arr, torch.full((len(tmp),), 1+100, dtype=torch.int32)))
    queries.extend(tmp)

    # Last ball type caption
    for i in range(10):
        tmp = get_captions(f'dataset/text_data/end_{shotTypes[i]}.txt')
        eval_class_arr = torch.cat((eval_class_arr, torch.full((len(tmp),), i+100, dtype=torch.int32)))
        queries.extend(tmp)

    # Frequent ball type caption
    for i in range(2, 10):
        tmp = get_captions(f'dataset/text_data/frequent_{shotTypes[i]}.txt')
        eval_class_arr = torch.cat((eval_class_arr, torch.full((len(tmp),), i+200, dtype=torch.int32)))
        queries.extend(tmp)

    print('\nGeneral metric')
    retrieval(queries, eval_class_arr=eval_class_arr)

    # Metric for each type of queries
    # Rally length caption
    queries = []
    eval_class_arr = None
    for rally_length in range(1, 25):
        tmp = get_captions('dataset/text_data/length.txt', rally_length=rally_length)
        if eval_class_arr is None:
            eval_class_arr = torch.full((len(tmp),), rally_length, dtype=torch.int32)
        else:
            eval_class_arr = torch.cat((eval_class_arr, torch.full((len(tmp),), rally_length, dtype=torch.int32)))
        queries.extend(tmp)
    print('\nRally length metric')
    retrieval(queries, eval_class_arr=eval_class_arr)

    # Serve type caption
    queries = []
    eval_class_arr = None
    tmp = get_captions('dataset/text_data/short_service.txt')
    eval_class_arr = torch.full((len(tmp),), 0+100, dtype=torch.int32)
    queries.extend(tmp)
    tmp = get_captions('dataset/text_data/long_service.txt')
    tmp = random.sample(tmp, len(tmp)//2)
    eval_class_arr = torch.cat((eval_class_arr, torch.full((len(tmp),), 1+100, dtype=torch.int32)))
    queries.extend(tmp)
    print('\nServe type metric')
    retrieval(queries, eval_class_arr=eval_class_arr)

    # Last ball type caption
    queries = []
    eval_class_arr = None
    for i in range(10):
        tmp = get_captions(f'dataset/text_data/end_{shotTypes[i]}.txt')
        queries.extend(tmp)
        if eval_class_arr is None:
            eval_class_arr = torch.full((len(tmp),), i+100, dtype=torch.int32)
        else:
            eval_class_arr = torch.cat((eval_class_arr, torch.full((len(tmp),), i+100, dtype=torch.int32)))
    print('\nLast ball type metric')
    retrieval(queries, eval_class_arr=eval_class_arr)

    # Frequent ball type caption
    queries = []
    eval_class_arr = None
    for i in range(2, 10):
        tmp = get_captions(f'dataset/text_data/frequent_{shotTypes[i]}.txt')
        queries.extend(tmp)
        if eval_class_arr is None:
            eval_class_arr = torch.full((len(tmp),), i+200, dtype=torch.int32)
        else:
            eval_class_arr = torch.cat((eval_class_arr, torch.full((len(tmp),), i+200, dtype=torch.int32)))
    print('\nFrequent ball type metric')
    retrieval(queries, eval_class_arr=eval_class_arr)

    while True:
        query = input('\nEnter query, or enter \'q\' to quit: ')
        if query == 'q':
            break
        retrieval([query], -1)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)

    mp.set_start_method('spawn', force=True)

    main(config)
