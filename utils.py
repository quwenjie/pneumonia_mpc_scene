import crypten
import torch
import crypten.communicator as comm
import io
import copy
import collections
import pickle
import colink as CL
from colink.sdk_a import CoLink


class onehot_tensor(torch.Tensor):  # we only convert when loading because
    def __init__(self, *args):
        return


def convert_onehot(targ_dict, classes, src):
    if isinstance(targ_dict, collections.UserDict) or isinstance(targ_dict, dict):
        for k, v in targ_dict.items():
            targ_dict[k] = convert_onehot(v, classes, src)
    elif isinstance(targ_dict, collections.UserList) or isinstance(targ_dict, list):
        for i in range(len(targ_dict)):
            targ_dict[i] = convert_onehot(targ_dict[i], classes, src)
    elif torch.is_tensor(targ_dict):
        if isinstance(targ_dict, onehot_tensor):
            targ_dict = convert_indices_onehot(targ_dict, classes)
    return targ_dict


def zero_out_func(targ_dict):
    if isinstance(targ_dict, collections.UserDict) or isinstance(targ_dict, dict):
        for k, v in targ_dict.items():
            targ_dict[k] = zero_out_func(v)
    elif isinstance(targ_dict, collections.UserList) or isinstance(targ_dict, list):
        for i in range(len(targ_dict)):
            targ_dict[i] = zero_out_func(targ_dict[i])
    elif torch.is_tensor(targ_dict):
        targ_dict.fill_(0)
    return targ_dict


def convert_dict_to_shared(targ_dict, src):
    if isinstance(targ_dict, collections.UserDict) or isinstance(targ_dict, dict):
        for k, v in targ_dict.items():
            targ_dict[k] = convert_dict_to_shared(v, src)
    elif isinstance(targ_dict, collections.UserList) or isinstance(targ_dict, list):
        for i in range(len(targ_dict)):
            targ_dict[i] = convert_dict_to_shared(targ_dict[i], src)
    elif torch.is_tensor(targ_dict):
        return crypten.cryptensor(targ_dict, src=src)
    return targ_dict


def download_online_file(path):
    basename = os.path.basename(path)
    if os.path.exists(basename):
        os.remove(basename)
    os.system("wget {}".format(path))
    return basename


def load_from_party_custom(
    f=None,
    cl=None,
    location="file",
    preloaded=None,
    encrypted=False,
    model_class=None,
    src=0,
    load_closure=torch.load,
    zero_out=zero_out_func,  # specify function to handle customize zero out
    convert_share=convert_dict_to_shared,  # specify function to handle customize share tensor convert
    convert_onehot_classes=0,
    data_id_filter="",
    **kwargs,
):
    """
    Loads an object saved with `torch.save()` or `crypten.save_from_party()`.

    Args:
        f: a file-like object (has to implement `read()`, `readline()`,
              `tell()`, and `seek()`), or a string containing a file name
        preloaded: Use the preloaded value instead of loading a tensor/model from f.
        encrypted: Determines whether crypten should load an encrypted tensor
                      or a plaintext torch tensor.
        model_class: Takes a model architecture class that is being communicated. This
                    class will be considered safe for deserialization so non-source
                    parties will be able to receive a model of this type from the
                    source party.
        src: Determines the source of the tensor. If `src` is None, each
            party will attempt to read in the specified file. If `src` is
            specified, the source party will read the tensor from `f` and it
            will broadcast it to the other parties
        load_closure: Custom load function that matches the interface of `torch.load`,
        to be used when the tensor is saved with a custom save function in
        `crypten.save_from_party`. Additional kwargs are passed on to the closure.
    """

    if encrypted:
        raise NotImplementedError("Loading encrypted tensors is not yet supported")
    else:
        assert isinstance(src, int), "Load failed: src argument must be an integer"
        assert (
            src >= 0 and src < comm.get().get_world_size()
        ), "Load failed: src must be in [0, world_size)"

        # source party
        if comm.get().get_rank() == src:
            assert (f is None and (preloaded is not None)) or (
                (f is not None) and preloaded is None
            ), "Exactly one of f and preloaded must not be None"

            if f is None:
                result = preloaded
            if preloaded is None:
                if location == "file":
                    result = load_closure(f, **kwargs)
                elif location == "online":
                    download_file = download_online_file(f)
                    result = load_closure(download_file, **kwargs)
                elif location == "storage":
                    payload = b""
                    b = 0
                    while True:
                        lst_key = f + ":" + str(b)
                        res = cl.read_entries(
                            [
                                CL.StorageEntry(
                                    key_name=lst_key,
                                )
                            ]
                        )
                        if res is None:
                            break
                        b += 1
                        payload = payload + res[0].payload
                    result = load_closure(io.BytesIO(payload), **kwargs)
                else:
                    raise NotImplementedError(
                        "Location {} is not supported!".format(location)
                    )
            if data_id_filter!="":
                data_ids_lst=pickle.loads(cl.read_entry(data_id_filter))
                data_ids=set(data_ids_lst)
                useful_data=collections.UserList()
                for data in result:
                    if data['ID'] in data_ids:
                        useful_data.append(data)
                result=useful_data
            # convert catalog label to one-hot form before share
            if convert_onehot_classes > 0:
                result = convert_onehot(result, convert_onehot_classes, src)
            # Zero out the tensors / modules to hide loaded data from broadcast
            if torch.is_tensor(result):
                result_zeros = result.new_zeros(result.size())
            elif isinstance(result, torch.nn.Module):
                result_zeros = copy.deepcopy(result)
                for p in result_zeros.parameters():
                    p.data.fill_(0)
            else:
                result_zeros_dict = zero_out(copy.deepcopy(result))
                # remove broadcasting unrecognized type
                result_zeros = io.BytesIO()
                torch.save(
                    result_zeros_dict, result_zeros
                )  # use torch to reduce communication size
            comm.get().broadcast_obj(result_zeros, src)

        # Non-source party
        else:
            if model_class is not None:
                crypten.common.serial.register_safe_class(model_class)
            result = comm.get().broadcast_obj(None, src)
            if isinstance(result, io.BytesIO):  # unzip
                result.seek(0)
                result = torch.load(result)
            # removed receiving unrecognized type

        result = convert_share(result, src)
        result.src = src
        return result


def convert_indices_onehot(indices, num_targets=None):
    assert indices.dtype == torch.long, "indices must be long integers"
    assert indices.min() >= 0, "indices must be non-negative"
    if num_targets is None:
        num_targets = indices.max() + 1
    onehot_vector = torch.zeros(indices.nelement(), num_targets, dtype=torch.long)
    onehot_vector.scatter_(1, indices.view(indices.nelement(), 1), 1)
    return onehot_vector
