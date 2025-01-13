import transformers
import torch

import re
import random

shotTypes = ['short_service', 'long_service', 'net_kill', 'net', 'push', 'drive', 'lob', 'smash', 'drop', 'clear']


def get_captions(filename, rally_length=None):
    output = list(open(filename, 'r', encoding='utf-8').read().splitlines())
    if rally_length is not None:
        for i in range(len(output)):
            output[i] = output[i].replace('&', str(rally_length))
    return output


def generate_message(caption):
    messages = [
        {"role": "system", "content": """
        There is a database of badminton rally clips, you will be given queries to search matching videos. The queries can be classified as one of the following classes, rally_length, service_type, last_shot_type, and most_frequent_ball_type queries. You have to classify the type of query that you are given, and extract additional information as token based on the query type.
        The token for 'rally_length' type queries should be the specified rally length in the query.
        The token for 'service_type' type queries should be either 'short_service' or 'long_service' based on the query.
        The token for 'last_shot_type' and 'most_frequent_ball_type' type queries should be the shot type specified in the query, and could be one of the following ['short_service', 'long_service', 'net_kill', 'net', 'push', 'drive', 'lob', 'smash', 'drop', 'clear']
        For example, if the query is 'Show me rallies that lasted over 10 shots', then this query should be classified as 'Class: rally_length\nToken: 10\n'. Another example, if the query is 'Find rallies that started with a long service', then this query should be classified as 'Class: service_type\nToken: long_service\n'.
        The format of your output should be 'Class: <class_name>\nToken: <token_value>\n'."""},
        {"role": "user", "content": caption},
    ]
    return messages


def main():
    count_positives = 0
    count_negatives = 0
    model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    class_pos = 0
    class_neg = 0
    for rally_length in range(1, 15):
        print(f"\nTesting query type: rally_length w/{rally_length}")
        tmp = get_captions('dataset/text_data/length.txt', rally_length=rally_length)
        for cap in tmp:
            message = generate_message(cap)
            outputs = pipeline(
                message,
                max_new_tokens=64,
            )
            reply = outputs[0]["generated_text"][-1]['content']
            print(cap, reply)
            reply = re.split('\n| ', reply)
            try:
                cls = reply[1]
                token = int(reply[3])
                if cls != 'rally_length' or token != rally_length:
                    print(f"False: {cls} {token}")
                    class_neg += 1
                    count_negatives += 1
                else:
                    print(f"True: {cls} {token}")
                    class_pos += 1
                    count_positives += 1
            except ValueError:
                print("Failed to parse generated text")
    print(f"\nRally length queries accuracy: {class_pos / (class_pos + class_neg) * 100}%")

    print("\nTesting query type: service_type w/short_service")
    tmp = get_captions('dataset/text_data/short_service.txt')
    class_pos = 0
    class_neg = 0
    for cap in tmp:
        message = generate_message(cap)
        outputs = pipeline(
            message,
            max_new_tokens=64,
        )
        reply = outputs[0]['generated_text'][-1]['content']
        print(cap, reply)
        reply = re.split('\n| ', reply)
        cls = reply[1]
        token = reply[3]
        if cls != 'service_type' or token != 'short_service':
            print(f"False: {cls} {token}")
            class_neg += 1
            count_negatives += 1
        else:
            print(f"True: {cls} {token}")
            class_pos += 1
            count_positives += 1

    print("\nTesting query type: service_type w/long_service")
    tmp = get_captions('dataset/text_data/long_service.txt')
    for cap in tmp:
        message = generate_message(cap)
        outputs = pipeline(
            message,
            max_new_tokens=64,
        )
        reply = outputs[0]['generated_text'][-1]['content']
        print(cap, reply)
        reply = re.split('\n| ', reply)
        cls = reply[1]
        token = reply[3]
        if cls != 'service_type' or token != 'long_service':
            print(f"False: {cls} {token}")
            class_neg += 1
            count_negatives += 1
        else:
            print(f"True: {cls} {token}")
            class_pos += 1
            count_positives += 1
    print(f"\nService type queries accuracy: {class_pos / (class_pos + class_neg) * 100}%")

    class_pos = 0
    class_neg = 0
    for shotType in shotTypes:
        print(f"\nTesting query type: last_shot_type w/{shotType}")
        tmp = get_captions(f'dataset/text_data/end_{shotType}.txt')
        for cap in tmp:
            message = generate_message(cap)
            outputs = pipeline(
                message,
                max_new_tokens=64,
            )
            reply = outputs[0]['generated_text'][-1]['content']
            print(cap, reply)
            reply = re.split('\n| ', reply)
            cls = reply[1]
            token = reply[3]
            if cls != 'last_shot_type' or token != shotType:
                print(f"False: {cls} {token}")
                class_neg += 1
                count_negatives += 1
            else:
                print(f"True: {cls} {token}")
                class_pos += 1
                count_positives += 1
    print(f"\nLast shot type queries accuracy: {class_pos / (class_pos + class_neg) * 100}%")

    class_pos = 0
    class_neg = 0
    for i in range(2, 10):
        print(f"\nTesting query type: most_frequent_ball_type w/{shotTypes[i]}")
        tmp = get_captions(f'dataset/text_data/frequent_{shotTypes[i]}.txt')
        for cap in tmp:
            message = generate_message(cap)
            outputs = pipeline(
                message,
                max_new_tokens=64,
            )
            reply = outputs[0]['generated_text'][-1]['content']
            print(cap, reply)
            reply = re.split('\n| ', reply)
            cls = reply[1]
            token = reply[3]
            if cls != 'most_frequent_ball_type' or token != shotTypes[i]:
                print(f"False: {cls} {token}")
                class_neg += 1
                count_negatives += 1
            else:
                print(f"True: {cls} {token}")
                class_pos += 1
                count_positives += 1
    print(f"\nMost frequent ball type queries accuracy: {class_pos / (class_pos + class_neg) * 100}%")

    print(f"\nAccuracy: {count_positives / (count_positives + count_negatives) * 100}%")


if __name__ == '__main__':
    main()

# pipeline("""<s>[INST] <<SYS>>
# There is a database of badminton rally clips, you will be given queries to search matching videos. The queries can be classified as one of the following classes, rally_length, service_type, last_shot_type, and most_frequent_ball_type queries. You have to classify the type of query that you are given, and extract additional information as token based on the query type. 
# The token for "rally_length" type queries should be the specified rally length in the query.
# The token for "service_type" type queries should be either "short_service" or "long_service" based on the query.
# The token for "last_shot_type" and "most_frequent_ball_type" type queries should be the shot type specified in the query, and could be one of the following ['short_service', 'long_service', 'net_kill', 'net', 'push', 'drive', 'lob', 'smash', 'drop', 'clear']
# For example, if the query is "Show me rallies that lasted over 10 shots", then this query should be classified as "Class: rally_length\nToken: 10\n". Another example, if the query is "Find rallies that started with a long service", then this query should be classified as "Class: service_type\nToken: long_service\n". 
# The format of your output should be "Class: <class_name>\nToken: <token_value>\n".
# <</SYS>>
# Show rallies that ended with a powerful smash. [/INST]""")


# model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
# )

# messages = [
#     {"role": "system", "content": """
#      There is a database of badminton rally clips, you will be given queries to search matching videos. The queries can be classified as one of the following classes, rally_length, service_type, last_shot_type, and most_frequent_ball_type queries. You have to classify the type of query that you are given, and extract additional information as token based on the query type.
#      The token for 'rally_length' type queries should be the specified rally length in the query.
#      The token for 'service_type' type queries should be either 'short_service' or 'long_service' based on the query.
#      The token for 'last_shot_type' and 'most_frequent_ball_type' type queries should be the shot type specified in the query, and could be one of the following ['short_service', 'long_service', 'net_kill', 'net', 'push', 'drive', 'lob', 'smash', 'drop', 'clear']
#      For example, if the query is 'Show me rallies that lasted over 10 shots', then this query should be classified as 'Class: rally_length\nToken: 10\n'. Another example, if the query is 'Find rallies that started with a long service', then this query should be classified as 'Class: service_type\nToken: long_service\n'.
#      The format of your output should be 'Class: <class_name>\nToken: <token_value>\n'."""},
#     {"role": "user", "content": "Show rallies that ended with a powerful smash."},
# ]

# outputs = pipeline(
#     messages,
#     max_new_tokens=64,
# )
# print(outputs[0]["generated_text"][-1])
