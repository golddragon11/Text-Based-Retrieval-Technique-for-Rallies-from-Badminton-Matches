import torch
import pickle
import numpy as np
import pandas as pd
import random
import os


# 發球: 短球、高球
# 回球: 擋小球、勾球、放小球、小平球、推球、撲球、挑球、防守回挑、防守回抽、平球、後場抽平球、切球、過度切球、殺球、點扣、長球
# 輸贏標記: 出界, 掛網, 未過網, 對手落地致勝, 落點判斷失誤

match = ""
ballTypes = ['發短球', '發長球', '擋小球', '勾球', '放小球', '小平球', '推球', '撲球', '挑球', '防守回挑', '防守回抽', '平球', '後場抽平球', '切球', '過度切球', '殺球', '點扣', '長球', '未知球種']
ballTypesEng = ['short_service', 'long_service', 'net_kill', 'cross_court_net', 'net', 'driven_flight', 'push', 'rush', 'lob', 'defensive_return_lob', 'defensive_return_drive', 'drive', 'backcourt_drive', 'drop', 'passive_drop', 'smash', 'wrist_smash', 'clear', 'undefined']
simplification = {'short_servide': 'short_service', 'long_service': 'long_service', 'net_kill': 'net_kill', 'cross_court_net': 'net', 'net': 'net', 'rush': 'net_kill', 'push':'push', 'driven_flight': 'push', 'lob': 'lob', 'defensive_return_lob': 'lob', 'defensive_return_drive': 'drive', 'backcourt_drive': 'drive', 'drop': 'drop', 'passive_drop': 'drop', 'smash': 'smash', 'wrist_smash': 'smash', 'clear': 'lob', 'undefined': 'undefined'}
simplified = ['short_service', 'long_service', 'net_kill', 'net', 'push', 'drive', 'lob', 'smash', 'drop', 'clear']
ballTypeDict = {ballTypes[i]: int(i) for i in range(len(ballTypes))}
simplifiedBallTypeDict = {'發短球': 0, '發長球': 1, '擋小球': 2, '勾球': 3, '放小球': 3, '小平球': 4, '推球': 4, '撲球': 2, '挑球': 6, '防守回挑': 6, '防守回抽': 5, '平球': 5, '後場抽平球': 5, '切球': 8, '過度切球': 8, '殺球': 7, '點扣': 7, '長球': 9, '未知球種': 10}

winReasonDict = {'對手犯規': [0, 'foul'], '對手未過網': [1, 'failedReturn'], '對手掛網': [2, 'net'], '落地致勝': [3, 'land'], '對手出界': [4, 'out'], '對手落點判斷失誤': [5, 'misjudgement']}


def pick_caption(filename, rally_length=None):
    output = random.choice(list(open(filename, 'r', encoding='utf-8').read().splitlines()))
    if rally_length is not None:
        output = output.replace('&', str(rally_length))
    return output


def stage1_training_data():
    def vidFeatureFilename(row):
        return "{}_{:02d}_{:02d}_{}_resnext.pt".format(row['set'], row['roundscore_A'], row['roundscore_B'], row['ball_round'])

    def vidFilename(row):
        return "{}_{:02d}_{:02d}_{}.mp4".format(row['set'], row['roundscore_A'], row['roundscore_B'], row['ball_round'])

    def tracknetFilename(row):
        return "{}_{:02d}_{:02d}_{}_ball.csv".format(row['set'], row['roundscore_A'], row['roundscore_B'], row['ball_round'])

    def tracknetFilename_rally(row):
        return "{}_{:02d}_{:02d}_ball.csv".format(row['set'], row['roundscore_A'], row['roundscore_B'])

    match_folders = os.listdir(directory)
    idx = 0
    maxVideoToken = maxTracknetToken = totalVideoToken = 0
    data = [[] for i in range(18)]
    classCount = [0] * 18
    for match in match_folders:
        for _, label_path in enumerate(os.listdir(os.path.join(directory, match, "label"))):
            df = pd.read_csv(os.path.join(directory, match, "label", label_path))
            # Drop unnecessary columns
            df = df.drop(['time', 'hit_height', 'hit_area', 'landing_height', 'getpoint_player', 'player_location_area', 'player_location_x', 'player_location_y', 'opponent_location_area', 'opponent_location_x', 'opponent_location_y', 'db'], axis=1)
            df['set'] = int(label_path[-5])

            match_path = os.path.join(directory, match)

            # Handle each rally
            for i, rally in enumerate(df['rally'].unique()):
                currRally = df[df['rally'] == rally]
                offset = df[df['rally'] == rally].iloc[0]['frame_num']

                for j, row in currRally.iterrows():
                    if row['type'] == '未知球種':
                        continue
                    try:
                        videoData = torch.load(os.path.join(match_path, "clips_embedding", vidFeatureFilename(row)))['clips']
                        videoFeature = torch.zeros(len(videoData), 2048)
                        for i in range(len(videoData)):
                            videoFeature[i] = torch.FloatTensor(videoData[i]['features'])
                        if maxVideoToken < len(videoFeature):
                            print(len(videoFeature), vidFeatureFilename(row))
                        maxVideoToken = max(maxVideoToken, len(videoFeature))

                        start_frame_num = df.iloc[j]['frame_num']
                        end_frame_num = df.iloc[j+1]['frame_num'] if j+1 < len(df) and df.iloc[j+1]['rally'] == df.iloc[j]['rally'] else df.iloc[j]['frame_num']+90
                        tracknetFeature = pd.read_csv(os.path.join(match_path, "trajectory", tracknetFilename_rally(row))).to_numpy().astype(np.float32)[start_frame_num-offset:end_frame_num-offset, 1:]
                        if tracknetFeature.shape[0] == 0:
                            continue
                        try:
                            minX = np.min(tracknetFeature[np.where(tracknetFeature[:, 1] != 0)][:, 1])
                            minY = np.min(tracknetFeature[np.where(tracknetFeature[:, 2] != 0)][:, 2])
                        except ValueError:
                            pass
                        totalVideoToken += len(videoFeature)

                        tracknetFeature = np.concatenate((tracknetFeature, tracknetFeature[:, 1:3]), axis=1)
                        tracknetFeature[:, 3] -= tracknetFeature[:, 0] * minX
                        tracknetFeature[:, 4] -= tracknetFeature[:, 0] * minY
                        diag = np.sqrt(pow(np.max(tracknetFeature[:, 3]), 2) + pow(np.max(tracknetFeature[:, 4]), 2))
                        tracknetFeature[:, 3:4] /= diag + 1e+6      # Normalize coordinates

                        displacements = np.zeros((len(tracknetFeature), 4))
                        prevRow = None
                        firstNonZero = None
                        for i, vec in enumerate(tracknetFeature):
                            if vec[0] == 1:
                                if firstNonZero is None:
                                    firstNonZero = i
                                displacements[i, :] = vec[1:5] - prevRow[1:5] if prevRow is not None else 0
                                prevRow = vec

                        distFromStart = tracknetFeature[:, 1:3] - tracknetFeature[firstNonZero if firstNonZero is not None else 0, 1:3]   # Calculate distance from starting position
                        distFromStart[:, 0] *= tracknetFeature[:, 0]
                        distFromStart[:, 1] *= tracknetFeature[:, 0]

                        totalTraveledDistance = np.cumsum(np.abs(displacements[:, 0:2]), axis=0)        # Calculate total traveled distance for each frame
                        totalTraveledDistance[:, 0] *= tracknetFeature[:, 0]
                        totalTraveledDistance[:, 1] *= tracknetFeature[:, 0]

                        tracknetFeature = np.concatenate((tracknetFeature, displacements, distFromStart, totalTraveledDistance), axis=1)
                        maxTracknetToken = max(maxTracknetToken, len(tracknetFeature))
                    except FileNotFoundError:
                        continue

                    classCount[simplifiedBallTypeDict[row['type']]] += 1
                    data[simplifiedBallTypeDict[row['type']]].append({
                        'id': '{:06d}'.format(idx),
                        'videoFeature': videoFeature,
                        'videoPath': os.path.join(match_path, "clips", vidFilename(row)),
                        'tracknet': tracknetFeature,
                        'tracknetPath': os.path.join(match_path, "trajectory", tracknetFilename(row)),
                        'shot_type': simplifiedBallTypeDict[row['type']]
                    })
                    idx += 1
    # print(idx)
    # print(maxVideoToken, maxTracknetToken, totalVideoToken/idx)
    # print(simplified)
    # print(classCount)

    train_data = []
    valid_data = []
    for subList in data:
        train_idx = np.arange(len(subList))
        np.random.shuffle(train_idx)
        len_valid = int(len(subList) * 0.2)
        valid_idx = train_idx[:len_valid]
        train_idx = np.delete(train_idx, np.arange(0, len_valid))
        train_data.extend([subList[i] for i in train_idx])
        valid_data.extend([subList[i] for i in valid_idx])

    random.shuffle(train_data)
    random.shuffle(valid_data)

    with open('stage1_training.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open('stage1_validation.pkl', 'wb') as f:
        pickle.dump(valid_data, f)


def stage1_inference_data():
    def vidFeatureFilename(row):
        return "{}_{:02d}_{:02d}_{}_resnext.pt".format(row['set'], row['roundscore_A'], row['roundscore_B'], row['ball_round'])

    def vidFilename(row):
        return "{}_{:02d}_{:02d}_{}.mp4".format(row['set'], row['roundscore_A'], row['roundscore_B'], row['ball_round'])

    def vidFilename_rally(row):
        return "{}_{:02d}_{:02d}.mp4".format(row['set'], row['roundscore_A'], row['roundscore_B'])

    def tracknetFilename(row):
        return "{}_{:02d}_{:02d}_{}_ball.csv".format(row['set'], row['roundscore_A'], row['roundscore_B'], row['ball_round'])

    def tracknetFilename_rally(row):
        return "{}_{:02d}_{:02d}_ball.csv".format(row['set'], row['roundscore_A'], row['roundscore_B'])

    match_folders = os.listdir(directory)
    idx = 0
    data = []
    for match in match_folders:
        for _, label_path in enumerate(os.listdir(os.path.join(directory, match, "label"))):
            df = pd.read_csv(os.path.join(directory, match, "label", label_path))
            # Drop unnecessary columns
            df = df.drop(['time', 'hit_height', 'hit_area', 'landing_height', 'getpoint_player', 'player_location_area', 'player_location_x', 'player_location_y', 'opponent_location_area', 'opponent_location_x', 'opponent_location_y', 'db'], axis=1)
            df['set'] = int(label_path[-5])

            match_path = os.path.join(directory, match)

            # Handle each rally
            for i, rally in enumerate(df['rally'].unique()):
                currRally = df[df['rally'] == rally]
                offset = df[df['rally'] == rally].iloc[0]['frame_num']
                rally_data = []
                classCount = [0] * 11

                for j, row in currRally.iterrows():
                    try:
                        videoData = torch.load(os.path.join(match_path, "clips_embedding", vidFeatureFilename(row)))['clips']
                        videoFeature = torch.zeros(len(videoData), 2048)
                        for i, clip in enumerate(videoData):
                            videoFeature[i] = torch.FloatTensor(clip['features'])

                        # tracknetFeature = pd.read_csv(os.path.join(match_path, "trajectory", tracknetFilename(row))).to_numpy().astype(np.float32)[:, 1:]
                        start_frame_num = df.iloc[j]['frame_num']
                        end_frame_num = df.iloc[j+1]['frame_num'] if j+1 < len(df) and df.iloc[j+1]['rally'] == df.iloc[j]['rally'] else df.iloc[j]['frame_num']+90
                        tracknetFeature = pd.read_csv(os.path.join(match_path, "trajectory", tracknetFilename_rally(row))).to_numpy().astype(np.float32)[start_frame_num-offset:end_frame_num-offset, 1:]
                        if tracknetFeature.shape[0] == 0:
                            continue
                        try:
                            minX = np.min(tracknetFeature[np.where(tracknetFeature[:, 1] != 0)][:, 1])
                            minY = np.min(tracknetFeature[np.where(tracknetFeature[:, 2] != 0)][:, 2])
                        except ValueError:
                            pass

                        tracknetFeature = np.concatenate((tracknetFeature, tracknetFeature[:, 1:3]), axis=1)
                        tracknetFeature[:, 3] -= tracknetFeature[:, 0] * minX
                        tracknetFeature[:, 4] -= tracknetFeature[:, 0] * minY
                        diag = np.sqrt(pow(np.max(tracknetFeature[:, 3]), 2) + pow(np.max(tracknetFeature[:, 4]), 2))
                        tracknetFeature[:, 3:4] /= diag + 1e+6      # Normalize coordinates

                        displacements = np.zeros((len(tracknetFeature), 4))
                        prevRow = None
                        firstNonZero = None
                        for i, vec in enumerate(tracknetFeature):
                            if vec[0] == 1:
                                if firstNonZero is None:
                                    firstNonZero = i
                                displacements[i, :] = vec[1:5] - prevRow[1:5] if prevRow is not None else 0
                                prevRow = vec

                        distFromStart = tracknetFeature[:, 1:3] - tracknetFeature[firstNonZero if firstNonZero is not None else 0, 1:3]   # Calculate distance from starting position
                        distFromStart[:, 0] *= tracknetFeature[:, 0]
                        distFromStart[:, 1] *= tracknetFeature[:, 0]

                        totalTraveledDistance = np.cumsum(np.abs(displacements[:, 0:2]), axis=0)        # Calculate total traveled distance for each frame
                        totalTraveledDistance[:, 0] *= tracknetFeature[:, 0]
                        totalTraveledDistance[:, 1] *= tracknetFeature[:, 0]

                        tracknetFeature = np.concatenate((tracknetFeature, displacements, distFromStart, totalTraveledDistance), axis=1)
                    except FileNotFoundError:
                        continue

                    classCount[simplifiedBallTypeDict[row['type']]] += 1
                    rally_data.append({
                        'shot_type': simplifiedBallTypeDict[row['type']],
                        'clipPath': os.path.join(match_path, "clips", vidFilename(row)),
                        'tracknetPath': os.path.join(match_path, "trajectory", tracknetFilename(row)),
                        'videoFeature': videoFeature,
                        'tracknet': tracknetFeature,
                    })

                rallyLength = len(currRally)
                firstShot = simplifiedBallTypeDict[currRally.iloc[0]['type']]
                lastShot = simplifiedBallTypeDict[currRally.iloc[rallyLength-1]['type']]
                mostFrequent = (np.argmax(classCount[2:-1])+2) if rallyLength >= 5 else None

                data.append({
                    'id': '{:06d}'.format(idx),
                    'videoPath': os.path.join(match_path, "rally_video", vidFilename_rally(currRally.iloc[0])),
                    'service_type': firstShot,
                    'last_shot_type': lastShot,
                    'rally_length': rallyLength,
                    'most_frequent': mostFrequent,
                    'shots': rally_data,
                })
                idx += 1
                print(data[-1]['id'], data[-1]['videoPath'], data[-1]['service_type'], data[-1]['last_shot_type'], data[-1]['rally_length'], data[-1]['most_frequent'])
    print(idx)

    with open('stage1_inference.pkl', 'wb') as f:
        pickle.dump(data, f)


def stage2_training_data():
    def vidFilename(row):
        return "{}_{:02d}_{:02d}.mp4".format(row['set'], row['roundscore_A'], row['roundscore_B'])

    dictInEffect = simplifiedBallTypeDict
    translationInEffect = simplified

    match_folders = os.listdir(directory)
    idx = 0
    data = []
    val_data = []
    countSkipped = 0
    for match in match_folders:
        for _, label_path in enumerate(os.listdir(os.path.join(directory, match, "label"))):
            df = pd.read_csv(os.path.join(directory, match, "label", label_path))
            # Drop unnecessary columns
            df = df.drop(['time', 'hit_height', 'hit_area', 'landing_height', 'getpoint_player', 'player_location_area', 'player_location_x', 'player_location_y', 'opponent_location_area', 'opponent_location_x', 'opponent_location_y', 'db'], axis=1)
            df['set'] = int(label_path[-5])

            match_path = os.path.join(directory, match)

            # Handle each rally
            for i, rally in enumerate(df['rally'].unique()):
                currRally = df[df['rally'] == rally]
                if len(currRally) <= 0:
                    continue

                videoPath = os.path.join(match_path, "clips", vidFilename(currRally.iloc[0]))
                firstShot = dictInEffect[currRally.iloc[0]['type']]
                lastShot = dictInEffect[currRally.iloc[len(currRally)-1]['type']]
                classCount = [0] * 11

                if lastShot == len(classCount)-1:
                    countSkipped += 1
                    continue

                for j, row in currRally.iterrows():
                    classCount[dictInEffect[row['type']]] += 1
                if classCount[len(classCount)-1] == len(currRally):
                    countSkipped += 1
                    continue
                classCount.pop()

                caption = []
                cls = []

                # Rally length caption
                caption.append(pick_caption('text_data/length.txt', rally_length=len(currRally)))
                cls.append(len(currRally))

                # Serve type caption
                if firstShot == 0:
                    caption.append(pick_caption('text_data/short_service.txt'))
                else:
                    caption.append(pick_caption('text_data/long_service.txt'))
                cls.append(firstShot+100)

                # Last ball type caption
                caption.append(pick_caption(f'text_data/end_{translationInEffect[lastShot]}.txt'))
                cls.append(lastShot+100)

                # Frequent ball type caption
                if len(currRally) >= 5:
                    mostFrequent = np.argmax(classCount[2:])+2
                    caption.append(pick_caption(f'text_data/frequent_{translationInEffect[mostFrequent]}.txt'))
                    cls.append(mostFrequent+200)

                description = generate_description(len(currRally), firstShot, lastShot, mostFrequent)

                tmp = []
                # To make sure every caption gets trained
                for zipCaption in zip(caption, cls):
                    tmp.append({
                        'id': '{:06d}'.format(idx),
                        'videoPath': videoPath,
                        'service_type': firstShot,
                        'last_shot_type': lastShot,
                        'rally_length': len(currRally),
                        'shot_counts': classCount,
                        'most_frequent': mostFrequent,
                        'description': description,
                        'caption': zipCaption[0],
                        'class': cls,
                        'eval_class': zipCaption[1]
                    })
                    idx += 1

                tmp_idx = random.randint(0, len(tmp)-1)
                val_data.append(tmp[tmp_idx])
                tmp.pop(tmp_idx)
                data.extend(tmp)

    # print(idx, countSkipped)
    # print(data[0])
    # print(len(data), len(val_data))

    synthesized = []
    for i in range(idx, len(data)*2):
        synthesized.extend(synthesize_data(idx, translationInEffect))
    # print(synthesized[0])

    random.shuffle(data)
    random.shuffle(val_data)

    with open('stage2_training.pkl', 'wb') as f:
        data.extend(synthesized)
        pickle.dump(data, f)
    with open('stage2_validation.pkl', 'wb') as f:
        pickle.dump(val_data, f)


def generate_description(rally_length, first_shot, last_shot, most_frequent):
    name = ['short service', 'long service', 'net kill', 'net shot', 'push shot', 'drive shot', 'lob shot', 'smash', 'drop shot', 'clear']
    description = []
    description.append(f"The rally lasted for {rally_length} shots.")
    description.append(f"A total of {rally_length} shots were played.")
    if first_shot == 0:
        description.append("The player made a short service to start the game.")
        description.append("This rally started with a short service.")
    else:
        description.append("The player made a long service to start the game.")
        description.append("This rally started with a long service.")
    description.append(f"The last shot of the rally was a {name[last_shot]}.")
    description.append(f"The rally ended with a {name[last_shot]}.")
    if rally_length >= 5:
        description.append(f"{name[most_frequent]} is the most frequent shot in this rally.")
        description.append(f"The most frequent shot is {name[most_frequent]}.")
    else:
        description.append("The rally is too short to determine the most frequent shot.")
        description.append("This rally isn't long enough to determine the most frequent shot.")
    return description


def synthesize_data(idx, translationInEffect=simplified):
    rally_length = np.random.randint(1, 51, size=1)[0]
    serve_type = last_shot = np.random.randint(0, 2, size=1)[0]
    shot_counts = [0] * 10
    shot_counts[serve_type] = 1
    for i in range(rally_length-1):
        last_shot = np.random.randint(2, 10, size=1)[0]
        shot_counts[last_shot] += 1
    most_frequent = np.argmax(shot_counts)

    caption = []
    cls = []

    # Rally length caption
    caption.append(pick_caption('text_data/length.txt', rally_length=rally_length))
    cls.append(rally_length)

    # Serve type caption
    if serve_type == 0:
        caption.append(pick_caption('text_data/short_service.txt'))
    else:
        caption.append(pick_caption('text_data/long_service.txt'))
    cls.append(serve_type+100)

    # Last ball type caption
    if last_shot != len(shot_counts)-1:
        caption.append(pick_caption(f'text_data/end_{translationInEffect[last_shot]}.txt'))
        cls.append(last_shot+100)

    # Frequent ball type caption
    if rally_length >= 5 and most_frequent > 1:
        caption.append(pick_caption(f'text_data/frequent_{translationInEffect[most_frequent]}.txt'))
        cls.append(most_frequent+200)

    desc = generate_description(rally_length, serve_type, last_shot, most_frequent)

    output = []
    for zipCaption in zip(caption, cls):
        output.append({
            'id': '{:06d}'.format(idx),
            'videoPath': None,
            'service_type': serve_type,
            'last_shot_type': last_shot,
            'rally_length': rally_length,
            'shot_counts': shot_counts,
            'most_frequent': most_frequent,
            'description': desc,
            'caption': zipCaption[0],
            'class': cls,
            'eval_class': zipCaption[1]
        })
        idx += 1

    return output


if __name__ == "__main__":
    directory = "/home/shenkh/S2_LabelingTool_new/dist/main_widget/Data"
    stage1_training_data()
    stage1_inference_data()
    stage2_training_data()
