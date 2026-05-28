import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import os

# n_bits_w = 4
# n_bits_a = 8

def calculate_t_channel_error(t_tensor, error_tensor):
    t_set = list(set(t_tensor.numpy().tolist()))
    bias_table=[[] for i in range(len(t_set))]
    channel_num = error_tensor.shape[1]
    for c in range(channel_num):
        channel_error_tensor = error_tensor[:,c,:,:]
        t_channel_error = torch.mean(channel_error_tensor, dim=(1,2))
        resultlist_list = [[] for _ in range(len(t_set))]
        for i in range(len(t_tensor)):
            resultlist_list[t_set.index(int(t_tensor[i]))].append(float(t_channel_error[i]))
        
        result_list = []
        for values in resultlist_list:
            result_list.append(sum(values)/len(values))
        
        idxs = t_set.copy()
        idxs.sort()
        for i in range(len(idxs)):
            idx=idxs[i]
            bias_table[i].append(result_list[t_set.index(idx)])
            
    bias_arr = np.array(bias_table)
    # np.save('/home/gpu1-user13/PTQD/correct_data/imagenet_20steps_w{}a{}/idx_bias.npy'.format(n_bits_w, n_bits_a), bias_arr)
    # print(bias_table)
    return bias_arr  # 添加return语句


def get_kt_dict_imagenet(data_tensor, error_tensor, t_tensor):

    t_data_dict, t_error_dict = {}, {}

    for i in range(len(t_tensor)):
        int_t = t_tensor[i].item()
        if int_t not in t_data_dict.keys():
            t_data_dict[int_t] = data_tensor[i]
            t_error_dict[int_t] = error_tensor[i]
        else:
            t_data_dict[int_t] = torch.cat([t_data_dict[int_t], data_tensor[i]], dim=0)
            t_error_dict[int_t] = torch.cat([t_error_dict[int_t], error_tensor[i]], dim=0)
    
    kt_dict = {}
    r_dict = {}
    for k in t_data_dict.keys():
        flatten_data = t_data_dict[k].flatten()
        flatten_error = t_error_dict[k].flatten()
        slope, intercept, r_value, p_value, std_err = st.linregress(flatten_data, flatten_error)
        
        kt_dict[k] = slope
        r_dict[k] = r_value
    
    # print('r_value: ', r_value)
    # print(kt_dict)
    # np.save('/home/gpu1-user13/PTQD/correct_data/imagenet_20steps_w{}a{}/kt.npy'.format(n_bits_w, n_bits_a), kt_dict, allow_pickle=True)
    return kt_dict  # 只返回需要的字典


def get_t_residualerror_std_dict(t_tensor, data_tensor, error_tensor, kt_dict):
    '''
        need kt first for calculating residual error
    '''
    # kt_dict = np.load('/home/gpu1-user13/PTQD/correct_data/imagenet_20steps_w{}a{}/kt.npy'.format(n_bits_w, n_bits_a), allow_pickle=True).item()
    t_std_dict = {}
    for i in range(len(t_tensor)):
        int_t = t_tensor[i].item()
        if int_t in kt_dict.keys():
            k = torch.tensor(kt_dict[int_t].astype('float32'))
            k = F.relu(k)
            k = k.item()
        else:
            k = 0
        residual_error = data_tensor[i] + error_tensor[i] - (1+k)*data_tensor[i]
        std = torch.std(residual_error)

        if int_t not in t_std_dict:
            t_std_dict[int_t] = [std]
        else:
            t_std_dict[int_t].append(std)
    for k in t_std_dict.keys():
        t_std_dict[k] = sum(t_std_dict[k]) / len(t_std_dict[k])
    # print(t_std_dict)
    # np.save("/home/gpu1-user13/PTQD/correct_data/imagenet_20steps_w{}a{}/t_std_dict.npy".format(n_bits_w, n_bits_a), t_std_dict, allow_pickle=True)
    return t_std_dict  # 添加return语句


def process_single_timestep(data_tensor, error_tensor, timestep):
    """
    处理单个时间步的数据，调用原有的三个函数
    
    参数:
    - data_tensor: model_output_tea_cfg (单个时间步)
    - error_tensor: model_output_stu_cfg - model_output_tea_cfg (单个时间步)
    - timestep: 当前时间步
    
    返回:
    - result: 包含处理结果的字典
    """
    # 确保数据是torch.Tensor类型
    if isinstance(data_tensor, np.ndarray):
        data_tensor = torch.from_numpy(data_tensor)
    if isinstance(error_tensor, np.ndarray):
        error_tensor = torch.from_numpy(error_tensor)
    # 将单个tensor转换为batch形式，以满足原函数的输入要求
    data_batch = data_tensor.unsqueeze(0) if len(data_tensor.shape) == 3 else data_tensor
    error_batch = error_tensor.unsqueeze(0) if len(error_tensor.shape) == 3 else error_tensor
    t_batch = torch.tensor([timestep])
    
    # 调用原有函数
    kt_dict = get_kt_dict_imagenet(data_batch, error_batch, t_batch)
    
    t_std_dict = get_t_residualerror_std_dict(t_batch, data_batch, error_batch, kt_dict)
    
    # 对于calculate_t_channel_error，获取bias信息 - 只有一个t，直接取第一个结果
    bias_arr = calculate_t_channel_error(t_batch, error_batch)
    bias_value = torch.tensor(bias_arr[0], dtype=torch.float32).view(1, -1, 1, 1)
    
    # 创建结果字典
    result = {
        'timestep': timestep,
        'slope_k': kt_dict[timestep],
        'intercept': bias_value,  # 使用线性回归得到的intercept
        'residual_std_old': t_std_dict[timestep].item() if isinstance(t_std_dict[timestep], torch.Tensor) else t_std_dict[timestep]
    }
    
    return result

# 使用示例:
# t = 某个时间步
# model_output_tea_cfg_t = 某个tensor (教师模型输出)
# diff_t = 某个tensor (模型差异)
# result_t = process_single_timestep(model_output_tea_cfg_t, diff_t, t)

# if __name__ == '__main__':
#     import os
#     folder_path = "/home/gpu1-user13/PTQD/correct_data/imagenet_20steps_w{}a{}/".format(n_bits_w, n_bits_a)
#     os.makedirs(folder_path, exist_ok=True)
#     data_error_t_list = torch.load('/home/gpu1-user13/PTQD/data_error_t_w{}a{}_scale3.0_eta0.0_step20.pth'.format(n_bits_w, n_bits_a), map_location='cpu')  ## replace error file here
#     data_list = []
#     error_list = [] 
#     t_list = []
#     for i in range(len(data_error_t_list)):
#         for j in range(len(data_error_t_list[i][0])):
#             data_list.append(data_error_t_list[i][0][j])
#             error_list.append(torch.pow(data_error_t_list[i][1][j],1))
#             t_list.append(data_error_t_list[i][2][j])

#     data_tensor = torch.stack(data_list) 
#     error_tensor = torch.stack(error_list)
#     t_tensor = torch.stack(t_list)

#     get_kt_dict_imagenet(data_tensor, error_tensor, t_tensor)
#     calculate_t_channel_error(t_tensor, error_tensor)
#     get_t_residualerror_std_dict(t_tensor, data_tensor, error_tensor)