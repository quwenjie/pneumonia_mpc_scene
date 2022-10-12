import torch
import random
from pneumonia_utils import load_plain_feat_label, save_data_each_party
from utils import onehot_tensor
"""
2 users, first party have feature, second party have label
"""


def generate_pneumonia_data_config(feature_parties=[],
                                   label_parties=[],
                                   data_cnt=1000):
    data_sery = []
    party_num = 1 + max(feature_parties + label_parties)
    data_sery = [[] for _ in range(party_num)]
    for i in range(data_cnt):
        feature_party = random.choice(feature_parties)
        data_record1 = {"ID": "MEDMNIST_ID_{}".format(i)}
        data_record1["is_feature"] = True
        data_record1["name"] = "medmnist_image"
        data_sery[feature_party].append(data_record1)

        label_party = random.choice(label_parties)
        data_record1 = {"ID": "MEDMNIST_ID_{}".format(i)}
        data_record1["is_feature"] = False
        data_record1["name"] = "classification_label"
        data_sery[label_party].append(data_record1)

    return data_sery


def split_pneumonia_dataset_to_records(data_cnt=100):
    feat_plain, label_plain = load_plain_feat_label()
    feat_list = torch.split(feat_plain, 1)
    feat_list_ret = [[feat_list[i]] for i in range(data_cnt)]
    label_list = list(torch.split(label_plain, 1))
    for i in range(len(label_list)):
        if isinstance(label_list[i].item(), int):
            label_list[i] = onehot_tensor([label_list[i].item()]).long()
    return feat_list_ret, label_list


if __name__ == "__main__":
    feature_parties = [1, 2, 3]
    label_parties = [0]
    party_num = 1 + max(feature_parties + label_parties)
    data_config = generate_pneumonia_data_config(
        feature_parties=feature_parties,
        label_parties=label_parties,
        data_cnt=300)
    feat_plain, label_plain = split_pneumonia_dataset_to_records(data_cnt=300)
    for i in range(party_num):
        save_data_each_party(
            filename="pneumonia_{}.pth".format(i),
            feat_plain=feat_plain,
            label_plain=label_plain,
            data_config=data_config[i],
        )
