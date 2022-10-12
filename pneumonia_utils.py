import torch
import collections
def load_plain_feat_label(feat_name="pneumonia_data.pth", label_name="pneumonia_label.pth"):
    feat_plain = torch.load(feat_name)
    label_plain = torch.load(label_name).long()
    return feat_plain, label_plain


def get_feature_or_label(feat, label, data_rec):
    index = int(data_rec["ID"].replace("MEDMNIST_ID_", ""))
    feat_list=['medmnist_image']
    if data_rec["name"] in feat_list:
        if len(feat[index]) == 1:
            print(feat[index][0][0].shape)
            return feat[index][0][0]
    else:
        if len(label[index]) == 1:
            print('label: ',label[index][0], label[index][0].shape)
            return label[index][0]


def save_data_each_party(
    filename,feat_plain, label_plain, data_config
):  # auxillary function to store data on each device
    dataset_p = collections.UserList()  # dict.src=src needs collections.UserDict
    for data_record in data_config:
        dataset_p.append(
            {
                **data_record,
                "data": get_feature_or_label(feat_plain, label_plain, data_record),
            }
        )
    torch.save(dataset_p, filename)
