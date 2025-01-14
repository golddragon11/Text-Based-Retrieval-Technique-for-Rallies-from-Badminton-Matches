import pandas as pd
import subprocess
import cv2

import os
from datetime import datetime, timedelta, timezone


def extract_clip(directory):
    match_folders = os.listdir(directory+"/Data")

    for match in match_folders:
        print("Generating video embedding for ", match)
        label_folder = os.path.join(directory+"/Data", match, "label")
        trajectory_folder = os.path.join(directory+"/Data", match, "trajectory")
        clip_path = os.path.join(directory+"/Data", match, "clips")
        if not os.path.exists(trajectory_folder):
            os.makedirs(trajectory_folder)
        if not os.path.exists(clip_path):
            os.makedirs(clip_path)

        video_path = os.path.join(directory+"/Video", match)
        video_fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)

        for filename in os.listdir(label_folder):
            print(filename)
            df = pd.read_csv(os.path.join(label_folder, filename))

            rally_starts = df.groupby('rally')['frame_num'].first().values
            rally_ends = df.groupby('rally')['frame_num'].last().values

            rally_score_A = df.groupby('rally')['roundscore_A'].first().values
            rally_score_B = df.groupby('rally')['roundscore_B'].first().values

            # # Extract rally clip
            # for i, meta in enumerate(zip(rally_starts, rally_ends, rally_score_A, rally_score_B)):
            #     start, end, roundscore_A, roundscore_B = meta[0], meta[1], meta[2], meta[3]

            #     clip_name = os.path.join(clip_path, f'{filename[-5]}_{roundscore_A:02d}_{roundscore_B:02d}.mp4')
            #     if os.path.exists(clip_name):
            #         print("Clip already exists")
            #     else:
            #         subprocess.call(['ffmpeg', '-n', '-ss', str(start/video_fps), '-to', str(end/video_fps + 3), '-i', video_path, '-an', clip_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            #     # Run TrackNetV3 on each rally
            #     if os.path.exists(os.path.join(trajectory_folder, f'{filename[-5]}_{roundscore_A:02d}_{roundscore_B:02d}_ball.csv')):
            #         print("Trajectory for rally already exists")
            #     else:
            #         print('Running TrackNetV3 on ', os.path.join(trajectory_folder, f'{filename[-5]}_{roundscore_A:02d}_{roundscore_B:02d}_ball.csv'))
            #         subprocess.call(['python', 'predict.py', '--video_file', clip_name, '--tracknet_file', 'ckpts/TrackNet_best.pt', '--inpaintnet_file', 'ckpts/InpaintNet_best.pt', '--save_dir', trajectory_folder], cwd='/home/shenkh/thesis/TrackNetV3')

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
                if os.path.exists(os.path.join(trajectory_folder, f'{filename[-5]}_{roundscore_A:02d}_{roundscore_B:02d}_{ball_round}_ball.csv')):
                    print("Trajectory already exists")
                elif not os.path.exists(clip_name):
                    print("Failed to extract clip possibly due to labeling error")
                else:
                    print('Running TrackNetV3 on ', os.path.join(trajectory_folder, f'{filename[-5]}_{roundscore_A:02d}_{roundscore_B:02d}_{ball_round}_ball.csv'))
                    subprocess.call(['python', 'predict.py', '--video_file', clip_name, '--tracknet_file', 'ckpts/TrackNet_best.pt', '--inpaintnet_file', 'ckpts/InpaintNet_best.pt', '--save_dir', trajectory_folder], cwd='/home/shenkh/thesis/TrackNetV3')


if __name__ == "__main__":
    directory = "/home/shenkh/S2_LabelingTool_new/dist/main_widget"
    extract_clip(directory)
