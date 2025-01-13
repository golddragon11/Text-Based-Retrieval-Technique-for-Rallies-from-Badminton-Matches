from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from torchvision.models.resnet import resnet152, ResNet152_Weights, resnext101_32x8d, ResNeXt101_32X8D_Weights
from torchvision.io.video import read_video
import torch.nn.functional as F
import torch
import numpy as np
import math
import pandas as pd
import subprocess
import cv2

import os
from datetime import datetime, timedelta, timezone


def extract_clip():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    match_folders = os.listdir("/home/shenkh/S2_LabelingTool_new/dist/main_widget/Data")
    weights = MViT_V2_S_Weights.DEFAULT
    model = mvit_v2_s(weights=weights)
    model.to(device)
    model.eval()

    for match in match_folders:
        print("Generating video embedding for ", match)
        label_folder = os.path.join("/home/shenkh/S2_LabelingTool_new/dist/main_widget/Data", match, "label")
        trajectory_folder = os.path.join("/home/shenkh/S2_LabelingTool_new/dist/main_widget/Data", match, "trajectory")
        clip_path = os.path.join("/home/shenkh/S2_LabelingTool_new/dist/main_widget/Data", match, "clips")
        if not os.path.exists(trajectory_folder):
            os.makedirs(trajectory_folder)
        if not os.path.exists(clip_path):
            os.makedirs(clip_path)

        video_path = os.path.join("/home/shenkh/S2_LabelingTool_new/dist/main_widget/Video", match)
        video_fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)

        for filename in os.listdir(label_folder):
            print(filename)
            df = pd.read_csv(os.path.join(label_folder, filename))

            rally_starts = df.groupby('rally')['frame_num'].first().values
            rally_ends = df.groupby('rally')['frame_num'].last().values
            # rally_starts = [datetime.fromtimestamp(time/30, timezone.utc) for time in df.groupby('rally')['frame_num'].first().values]
            # rally_starts = [datetime.strftime(time, '%H:%M:%S.%f') for time in rally_starts]
            # rally_ends = [datetime.fromtimestamp(time/30, timezone.utc) for time in df.groupby('rally')['frame_num'].last().values]
            # rally_ends = [datetime.strftime(time, '%H:%M:%S.%f') for time in rally_ends]

            # rally_starts_time = df.groupby('rally')['time'].first().values
            # rally_ends_time = [datetime.strptime(time, '%H:%M:%S') + timedelta(seconds=3) for time in df.groupby('rally')['time'].last().values]
            # rally_ends_time = [datetime.strftime(time, '%H:%M:%S') for time in rally_ends_time]

            rally_score_A = df.groupby('rally')['roundscore_A'].first().values
            rally_score_B = df.groupby('rally')['roundscore_B'].first().values

            # An_Se_Young_Ratchanok_Intanon_YONEX_Thailand_Open_2021_QuarterFinals.mp4 1_05_08.mp4
            for i, meta in enumerate(zip(rally_starts, rally_ends, rally_score_A, rally_score_B)):
                start, end, roundscore_A, roundscore_B = meta[0], meta[1], meta[2], meta[3]

                clip_name = os.path.join(clip_path, f'{filename[-5]}_{roundscore_A:02d}_{roundscore_B:02d}.mp4')
                if os.path.exists(clip_name):
                    print("Clip already exists")
                else:
                    subprocess.call(['ffmpeg', '-n', '-ss', str(start/video_fps), '-to', str(end/video_fps + 3), '-i', video_path, '-an', clip_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # Run TrackNetV3 on each rally
                if os.path.exists(os.path.join(trajectory_folder, f'{filename[-5]}_{roundscore_A:02d}_{roundscore_B:02d}_ball.csv')):
                    print("Trajectory for rally already exists")
                else:
                    print('Running TrackNetV3 on ', os.path.join(trajectory_folder, f'{filename[-5]}_{roundscore_A:02d}_{roundscore_B:02d}_ball.csv'))
                    subprocess.call(['python', 'predict.py', '--video_file', clip_name, '--tracknet_file', 'ckpts/TrackNet_best.pt', '--inpaintnet_file', 'ckpts/InpaintNet_best.pt', '--save_dir', trajectory_folder], cwd='/home/shenkh/thesis/TrackNetV3')

            for i in range(df.shape[0]):
                if df.iloc[i]['type'] == '未知球種':
                    continue
                # Extract shot clip
                roundscore_A, roundscore_B, ball_round = df.iloc[i]['roundscore_A'], df.iloc[i]['roundscore_B'], df.iloc[i]['ball_round']
                clip_name = os.path.join(clip_path, f'{filename[-5]}_{roundscore_A:02d}_{roundscore_B:02d}_{ball_round}.mp4')
                start = df.iloc[i]['frame_num']/video_fps
                if os.path.exists(clip_name):
                    print("Clip already exists")
                else:
                    if i+1 < df.shape[0] and df.iloc[i]['rally'] == df.iloc[i+1]['rally']:
                        end = df.iloc[i+1]['frame_num']/video_fps
                    else:
                        end = df.iloc[i]['frame_num']/video_fps+3
                    subprocess.call(['ffmpeg', '-y', '-ss', str(start), '-to', str(end), '-i', video_path, '-an', clip_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # Run TrackNetV3 on clip
                # if os.path.exists(os.path.join(trajectory_folder, f'{filename[-5]}_{roundscore_A:02d}_{roundscore_B:02d}_{ball_round}_ball.csv')):
                #     print("Trajectory already exists")
                # elif not os.path.exists(clip_name):
                #     print("Failed to extract clip possibly due to labeling error")
                # else:
                #     print('Running TrackNetV3 on ', os.path.join(trajectory_folder, f'{filename[-5]}_{roundscore_A:02d}_{roundscore_B:02d}_{ball_round}_ball.csv'))
                #     subprocess.call(['python', 'predict.py', '--video_file', clip_name, '--tracknet_file', 'ckpts/TrackNet_best.pt', '--inpaintnet_file', 'ckpts/InpaintNet_best.pt', '--save_dir', trajectory_folder], cwd='/home/shenkh/thesis/TrackNetV3')


def extract_clip_from_existing():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    match_folders = os.listdir("/home/shenkh/S2_LabelingTool_new/dist/main_widget/Data")
    weights = MViT_V2_S_Weights.DEFAULT
    model = mvit_v2_s(weights=weights)
    model.to(device)
    model.eval()

    for match in match_folders:
        print("Generating video embedding for ", match)
        clip_path = os.path.join("/home/shenkh/S2_LabelingTool_new/dist/main_widget/Data", match, "clips")
        for vid in os.listdir(os.path.join("/home/shenkh/S2_LabelingTool_new/dist/main_widget/Data", match, "rally_video")):
            print(vid)
            subprocess.call(['ffmpeg', '-y', '-i', os.path.join("/home/shenkh/S2_LabelingTool_new/dist/main_widget/Data", match, "rally_video", vid), '-an', os.path.join(clip_path, vid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def generate_video_embedding(directory, device="cuda"):
    match_folders = os.listdir(directory)
    if not torch.cuda.is_available():
        device = "cpu"

    weights = MViT_V2_S_Weights.DEFAULT
    model = mvit_v2_s(weights=weights)
    model.to(device)
    model.eval()

    preprocess = weights.transforms()

    for match in match_folders:
        print("Generating video embedding for ", match)
        rally_folder = os.path.join(directory, match, "rally_video")
        rally_videos = os.listdir(rally_folder)
        output_folder = os.path.join(directory, match, "rally_video_embedding")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in rally_videos:
            print(filename)
            video_path = os.path.join(rally_folder, filename)
            vid, _, _ = read_video(video_path, output_format="TCHW", pts_unit="sec")

            vid = F.pad(vid, (0, 0, 0, 0, 0, 0, 0, 16-vid.size(0) % 16))    # Pad number of frames to multiple of 16
            embedding = np.zeros((vid.size(0)//16, 400))
            for i in range(vid.size(0)//16):
                batch = preprocess(vid[i*16:(i+1)*16]).unsqueeze(0)
                batch = batch.to(device)
                print(batch.shape)
                with torch.no_grad():
                    output = model(batch)
                embedding[i] = output.detach().cpu().numpy()
            # torch.save(embedding, os.path.join(output_folder, os.path.splitext(filename)[0]+'_noSkip.pt'))
            torch.cuda.empty_cache()

            vid = vid[torch.arange(0, vid.size(0), 5)]      # Sample every 5 frames
            vid = F.pad(vid, (0, 0, 0, 0, 0, 0, 0, 16-vid.size(0) % 16))    # Pad number of frames to multiple of 16
            embedding = np.zeros((vid.size(0)//16, 400))
            for i in range(vid.size(0)//16):
                batch = preprocess(vid[i*16:(i+1)*16]).unsqueeze(0)
                batch = batch.to(device)
                with torch.no_grad():
                    output = model(batch)
                embedding[i] = output.detach().cpu().numpy()
            # torch.save(embedding, os.path.join(output_folder, os.path.splitext(filename)[0]+'.pt'))
            torch.cuda.empty_cache()


def video_embedding_resnet152(directory, device="cuda"):
    model_2d = resnet152(weights='DEFAULT')
    model_2d.to("cuda")
    model_3d = resnext101_32x8d(weights='DEFAULT')
    model_3d.to("cuda")
    model_2d.eval()
    model_3d.eval()
    preprocess_2d = ResNet152_Weights.DEFAULT.transforms()
    preprocess_3d = ResNeXt101_32X8D_Weights.DEFAULT.transforms()

    match_folders = os.listdir(directory)
    if not torch.cuda.is_available():
        device = "cpu"

    for match in match_folders:
        if match != "Kento_MOMOTA_Viktor_AXELSEN_Malaysia_Masters_2020_Finals.mp4":
            continue
        print("Generating video embedding for ", match)
        rally_folder = os.path.join(directory, match, "rally_video")
        rally_videos = os.listdir(rally_folder)
        output_folder = os.path.join(directory, match, "rally_video_embedding")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in rally_videos:
            print(filename)
            video_path = os.path.join(rally_folder, filename)
            vid, _, _ = read_video(video_path, output_format="TCHW", pts_unit="sec")
            batch = preprocess_3d(vid)
            batch = batch.to(device)
            with torch.no_grad():
                output = model_3d(batch)
                # print(output.size())
            embedding = np.zeros((math.ceil(batch.size(0)/16), 1000))
            for i in range(0, batch.size(0), 16):
                embedding[i//16] = np.mean(output[i:i+16].detach().cpu().numpy(), axis=0)
            torch.save(embedding, os.path.join(output_folder, os.path.splitext(filename)[0]+'_3d.pt'))
            torch.cuda.empty_cache()

            # batch = preprocess_2d(vid)
            # batch = batch.to(device)
            # with torch.no_grad():
            #     output = model_2d(batch)
            #     # print(output.size())
            # embedding = np.zeros((math.ceil(batch.size(0)/16), 1000))
            # for i in range(0, batch.size(0), 16):
            #     embedding[i//16] = np.mean(output[i:i+16].detach().cpu().numpy(), axis=0)
            # torch.save(embedding, os.path.join(output_folder, os.path.splitext(filename)[0]+'_2d.pt'))
            # torch.cuda.empty_cache()


if __name__ == "__main__":
    directory = "/home/shenkh/S2_LabelingTool_new/dist/main_widget/Data"
    # generate_video_embedding(directory, "cuda")
    # video_embedding_resnet152(directory, "cuda")
    extract_clip()
    # extract_clip_from_existing()
