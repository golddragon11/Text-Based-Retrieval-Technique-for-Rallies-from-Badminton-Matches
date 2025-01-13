# python test_stage1.py --resume output/models/Stage1_Model1/0822_132459/checkpoint-epoch56.pth --config configs/evaluation/config_stage1.json
import argparse
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import pickle
import numpy as np
import collections

import data_loader.data_loaders as module_data
import loss.combinatorial_loss as module_loss
from model.metric import MulticlassMetric as module_metric
import model.model as module_arch
from trainer import stage1_inference

from parse_config import ConfigParser


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def video_tokenization(data, n_tokens):
    if data.shape[0] <= n_tokens:
        cur_n_tokens, _ = data.shape
        token = torch.zeros(n_tokens, data.shape[-1])
        # token[:cur_n_tokens] = F.normalize(data[:cur_n_tokens], dim=1)
        token[:cur_n_tokens] = data[:cur_n_tokens]
        attention_mask = torch.zeros(n_tokens)
        attention_mask[:cur_n_tokens] = 1
    else:
        # token = torch.nn.functional.interpolate(
        #             data.permute(1, 0).unsqueeze(0),
        #             size=n_tokens,
        #             mode='nearest').squeeze(0).permute(1, 0)
        token = data[:n_tokens]
        # token = F.normalize(token, dim=1)
        attention_mask = torch.ones(n_tokens)
    return token.float(), attention_mask.float()


def tracknet_tokenization(data, n_tokens):
    token = torch.zeros(n_tokens, 12)
    if data.shape[0] <= n_tokens:
        cur_n_tokens, _ = data.shape
        # token[:cur_n_tokens] = F.normalize(data[:cur_n_tokens, 1:], dim=1)
        token[:cur_n_tokens] = data[:cur_n_tokens, 1:]
        attention_mask = torch.zeros(n_tokens)
        attention_mask[:cur_n_tokens] = data[:cur_n_tokens, 0]
    else:
        # token = F.normalize(data[:n_tokens, 1:], dim=1)
        token = data[:n_tokens, 1:]
        attention_mask = data[:n_tokens, 0]
    token += 1e-5
    return token.float(), attention_mask.float()


def generate_description(rally_length, first_shot, last_shot, most_frequent):
    name = ['short service', 'long service', 'net kill', 'net shot', 'push shot', 'drive shot', 'lob shot', 'smash', 'drop shot', 'clear']
    description = []
    description.append(f"The rally lasted for {rally_length} shots.")
    description.append(f"A total of {rally_length} shots were played.")
    if first_shot == 0:
        description.append("The player made a short service to start the game.")
        description.append("This rally started with a short service.")
    elif first_shot == 1:
        description.append("The player made a long service to start the game.")
        description.append("This rally started with a long service.")
    else:
        description.append("The model couldn't determine the type of service that was played.")
        description.append("The type of service that started the rally is unknown.")
    description.append(f"The last shot of the rally was a {name[last_shot]}.")
    description.append(f"The rally ended with a {name[last_shot]}.")
    if rally_length >= 5:
        description.append(f"{name[most_frequent]} is the most frequent shot in this rally.")
        description.append(f"The most frequent shot is {name[most_frequent]}.")
    else:
        description.append("The rally is too short to determine the most frequent shot.")
        description.append("This rally isn't long enough to determine the most frequent shot.")
    return description


def main(config):
    logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of metrics
    metrics = [module_metric(**met) for met in config['metrics']]

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

    data_path = config['data_path']
    data = pickle.load(open(data_path, 'rb'))
    stage2_input = []
    for rally in tqdm(data):
        batch = collections.defaultdict(lambda: [])
        for shot in rally['shots']:
            video_token, video_mask = video_tokenization(shot['videoFeature'], 80)
            tracknet_token, tracknet_mask = tracknet_tokenization(torch.from_numpy(shot['tracknet']), 140)
            batch['video_token'].append(video_token)
            batch['tracknet_token'].append(tracknet_token)
            batch['video_mask'].append(video_mask)
            batch['tracknet_mask'].append(tracknet_mask)
        output = stage1_inference(model, device, batch)
        target = torch.tensor([shot['shot_type'] for shot in rally['shots']]).to(device)

        for met in metrics:
            met.update(output, target)

        output = output.detach().cpu().numpy()

        first_shot = output[0]
        last_shot = output[-1]
        rally_length = len(output)
        shot_count = np.bincount(output)
        most_frequent = (np.argmax(shot_count[2:])+2) if rally_length >= 5 else None
        description = generate_description(rally_length, first_shot, last_shot, most_frequent)

        stage2_input.append({
            'id': rally['id'],
            'videoPath': rally['videoPath'],
            'service_type': rally['service_type'],
            'last_shot_type': rally['last_shot_type'],
            'rally_length': rally['rally_length'],
            'most_frequent': rally['most_frequent'],
            'stage1_prediction': {
                'service_type': first_shot,
                'last_shot_type': last_shot,
                'most_frequent': most_frequent
            },
            'description': description
        })

        del output

    log = {}
    for met in metrics:
        log.update(met.compute())
    for key, value in log.items():
        logger.info(f'    {str(key):15s}: {value}')

    with open('dataset/stage2_inference.pkl', 'wb') as f:
        pickle.dump(stage2_input, f)


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
