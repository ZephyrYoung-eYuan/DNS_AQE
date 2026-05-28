import os
import sys, time
sys.path.append(".")
sys.path.append('./taming-transformers')
from taming.models import vqgan

import torch
# torch.cuda.manual_seed(3407)
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from quant_scripts.brecq_quant_model import QuantModel
from quant_scripts.brecq_quant_layer import QuantModule
from quant_scripts.brecq_adaptive_rounding import AdaRoundQuantizer

import argparse
from datetime import datetime
from diffusers import DDIMScheduler
from diffusers.training_utils import set_seed
sys.path.append("..")
from quant.dns_main import dns_cali, dns_infer 
from quant.dns_test import ori_test

n_bits_w = 4
n_bits_a = 8

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")  
    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
    return model

def get_train_samples(train_loader, num_samples):
    image_data, t_data, y_data = [], [], []
    for (image, t, y) in train_loader:
        image_data.append(image)
        t_data.append(t)
        y_data.append(y)
        if len(image_data) >= num_samples:
            break
    return torch.cat(image_data, dim=0)[:num_samples], torch.cat(t_data, dim=0)[:num_samples], torch.cat(y_data, dim=0)[:num_samples]

def get_runtime_parameters():
    parser = argparse.ArgumentParser(description='处理运行时参数')
    parser.add_argument('--timestamp', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"),help='时间戳，格式为YYYY-MM-DD-HH-MM-SS')
    parser.add_argument('--dataset_path', type=str, default='imagenet_input_20steps.pth', help='dataset_path')
    # parser.add_argument('--dataset_path', type=str, default='/home/gpu1-user13/PTQD/imagenet_input_20steps.pth', help='dataset_path')
    parser.add_argument('--ckpt_path', type=str, default='quantw{}a{}_ldm_brecq.pth'.format(n_bits_w, n_bits_a), help='ckpt_path')
    # parser.add_argument('--ckpt_path', type=str, default='/home/gpu1-user13/PTQD/quantw{}a{}_ldm_brecq.pth'.format(n_bits_w, n_bits_a), help='ckpt_path')
    parser.add_argument('--base_dir', type=str, default='dns', help='base_path')
    # parser.add_argument('--base_dir', type=str, default='/home/gpu1-user13/dns', help='base_path')
    # parser.add_argument('--std_method', type=str, default='iqr', help='噪声估计方法')
    parser.add_argument('--dcnc_method', type=str, default='uniform', help='噪声估计方法')  # uniform, triangular, bimodalgs, generalizedgs
    # parser.add_argument('--ddim_eta', type=float, default=0.0, help='eta')
    parser.add_argument('--uni_weight', type=float, default=0.2, help='uni_weight')
    parser.add_argument('--cali', action='store_true',help='是否cali')
    parser.add_argument('--test', action='store_true',help='是否test')
    parser.add_argument('--eval', action='store_true',help='是否eval')
    parser.add_argument('--only_dns', action='store_true', help='是否only_dns')
    parser.add_argument('--seed', type=int, default=14, help='seed')
    input_args = parser.parse_args()
    return input_args

if __name__ == '__main__':
    input_args = get_runtime_parameters()
    seed = input_args.seed
    set_seed(seed)
    model_ori = get_model()
    model = get_model()
    dmodel = model.model.diffusion_model
    dmodel.cuda()
    dmodel.eval()
    from quant_scripts.quant_dataset import DiffusionInputDataset
    from torch.utils.data import DataLoader

    dataset = DiffusionInputDataset(input_args.dataset_path)
    data_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True) 
    
    wq_params = {'n_bits': n_bits_w, 'channel_wise': False, 'scale_method': 'mse'}
    aq_params = {'n_bits': n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': True}
    qnn = QuantModel(model=dmodel, weight_quant_params=wq_params, act_quant_params=aq_params)
    qnn.cuda()
    qnn.eval()

    print('Setting the first and the last layer to 8-bit')
    qnn.set_first_last_layer_to_8bit()

    cali_images, cali_t, cali_y = get_train_samples(data_loader, num_samples=1024)
    device = next(qnn.parameters()).device
    # Initialize weight quantization parameters
    qnn.set_quant_state(True, True)
    # Disable output quantization because network output
    # does not get involved in further computation
    qnn.disable_network_output_quantization()

    print('First run to init model...')
    with torch.no_grad():
        _ = qnn(cali_images[:32].to(device),cali_t[:32].to(device),cali_y[:32].to(device))

    # Kwargs for weight rounding calibration
    kwargs = dict(cali_images=cali_images, cali_t=cali_t, cali_y=cali_y, iters=10000, weight=0.01, asym=True,
                    b_range=(20, 2), warmup=0.2, act_quant=False, opt_mode='mse', batch_size=8)
        
    # Start calibration
    for name, module in qnn.named_modules():
        if isinstance(module, QuantModule) and module.ignore_reconstruction is False:
            module.weight_quantizer.soft_targets = False
            module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode='learned_hard_sigmoid', weight_tensor=module.org_weight.data)

    ckpt = torch.load(input_args.ckpt_path, map_location='cpu')
    qnn.load_state_dict(ckpt)
    qnn.cuda()
    qnn.eval()
    setattr(model.model, 'diffusion_model', qnn)

    # sampler = DDIMSampler_quantCorrection_imagenet(model, num_bit=4, correct=True)


    classes = range(1000)
    # n_samples_per_class = 6

    ## Quality, sampling speed and diversity are best controlled via the `scale`, `ddim_steps` and `ddim_eta` variables
    ddim_steps = 20
    # ddim_eta = input_args.ddim_eta
    scale = 3.0   # for  guidance

    is_cali = input_args.cali
    is_eval = input_args.eval
    is_test = input_args.test
    only_dns = input_args.only_dns
    assert is_eval is not True or is_test is not True, "不能同时为True"
    uni_weight = input_args.uni_weight
    dcnc_method = input_args.dcnc_method
    latent_shape=[3,64,64]
    nbit_w = n_bits_w
    nbit_a = n_bits_a
    timestamp = input_args.timestamp
    ddim_T = 1000
    t_d = int(ddim_T/ddim_steps)
    timesteps_range = list(range(1001-t_d, 0, -t_d))  #去噪50步
    noise_scheduler = DDIMScheduler(beta_start=0.0015, beta_end=0.0195, beta_schedule = 'scaled_linear', num_train_timesteps=ddim_T, prediction_type = 'epsilon', clip_sample= False)
    net_name = 'ldm_ptqd_brecq'
    base_dir = input_args.base_dir
    base_dir = os.path.join(base_dir, f'{net_name}/eval')
    cali_dir = f'dns/{net_name}/cali'
    # cali_dir = os.path.join(base_dir, f'{net_name}/cali')
    test_dir = f'dns/{net_name}/test'
    # test_dir = os.path.join(base_dir, f'{net_name}/test')
    # init_generator = torch.Generator()
    # init_generator.manual_seed(14)
    uni_generator = torch.Generator(device=device)
    uni_generator.manual_seed(7)
    with torch.no_grad():
        # with model.ema_scope():
        with model.ema_scope(), model_ori.ema_scope():
            def model_forward_tea_stu(noisy_images, timestep, mode, is_cfg=True, uniform_scale = 0, **cond_kwargs):
                # 这一部分的apply_model不能包括cfg过程。
                c_in = cond_kwargs['c']
                timesteps = torch.full((noisy_images.size(0),), timestep, device=device)
                if mode == 'student':
                    model_output = model.apply_model(torch.cat([noisy_images, noisy_images], dim=0), torch.cat([timesteps,timesteps],dim=0), c_in)
                elif mode == 'teacher':
                    model_output = model_ori.apply_model(torch.cat([noisy_images, noisy_images], dim=0), torch.cat([timesteps,timesteps],dim=0), c_in)
                else:
                    raise TypeError  
                shape  = model_output.shape
                if True and uniform_scale != 0:
                    if dcnc_method == 'uniform':
                        uniform_noise = (2 * uniform_scale) * torch.rand(model_output.shape, device = model_output.device, generator = uni_generator) - uniform_scale
                    elif dcnc_method == 'triangular':
                        u1 = torch.rand(shape, device=device, generator=uni_generator)
                        u2 = torch.rand(shape, device=device, generator=uni_generator)
                        uniform_noise = (u1 - u2) * uniform_scale
                    elif dcnc_method == 'bimodalgs':
                        # 0.5 N(-m, s^2) + 0.5 N(m, s^2)
                        # 在 numpy 里设定 r = m^2 / s^2 = 1 ⇒ m = s
                        s = uniform_scale          # 组件标准差 s
                        m = s                      # r = 1 ⇒ m = s

                        # ±1 号
                        signs = torch.randint(
                            0, 2, shape,
                            device=device,
                            generator=uni_generator,
                            dtype=torch.float32
                        )
                        signs = signs * 2.0 - 1.0  # {0,1} → {-1, +1}

                        # 标准正态再做仿射变换
                        z = torch.randn(shape, device=device, generator=uni_generator)
                        uniform_noise = z * s + signs * m
                    elif dcnc_method == 'generalizedgs':
                        # SciPy 中的 generalized Gaussian: stats.gennorm(beta_gg, loc=0, scale=alpha)
                        # pdf: f(x) = β / (2 α Γ(1/β)) * exp( - (|x|/α)^β )
                        # 采样方法：若 G ~ Gamma(k=1/β, θ=1)，则 |X| = α * G^{1/β}，符号 ±1 各半
                        beta_gg = 4.0
                        alpha = uniform_scale      # 即 numpy 中的 alpha（scale）

                        # Gamma(1/β, 1)，PyTorch: Gamma(concentration, rate)
                        # SciPy 的 scale θ = 1 对应 rate = 1
                        conc = torch.tensor(1.0 / beta_gg, device=device)
                        rate = torch.tensor(1.0, device=device)
                        gamma_dist = torch.distributions.Gamma(conc, rate)

                        # 正半径
                        g = gamma_dist.sample(shape)              # G ~ Gamma(1/β, 1)
                        r = alpha * g.pow(1.0 / beta_gg)         # |X|

                        # 随机符号 ±1
                        signs = torch.randint(
                            0, 2, shape,
                            device=device,
                            generator=uni_generator,
                            dtype=torch.float32
                        )
                        signs = signs * 2.0 - 1.0

                        uniform_noise = r * signs
                    model_output = model_output + uniform_noise
                return model_output
            def get_noisy_images(sample_images, noise, timestep, **cond_kwargs):
                # cali中对图片加噪
                timesteps = torch.full((sample_images.size(0),), timestep, device=device)
                return noise_scheduler.add_noise(sample_images, noise, timesteps)
            def get_img_label(batch):
                # 通常不需要修改
                images, label = batch
                return images.to(device), label.to(device)
            def get_label_cfg(label):
                # 获取cfg时使用的label
                batch_size = label.shape[0]
                label_free = torch.full((batch_size,), 1000, device=label.device, dtype=label.dtype)
                label_cfg = torch.cat([label, label_free]) # 看看是否要cfg对比
                return label_cfg
            def get_encoded_label(label_cfg):
                # label嵌入
                encoded_label = model.get_learned_conditioning({model.cond_stage_key: label_cfg})
                return encoded_label
            def model_vae_decode(sample_image):
                # decode
                return model.decode_first_stage(sample_image)
            def get_test_label(mode):
                # 这里用来获取生成的label
                if mode == 'cali':
                    cali_samples_per_class = 3
                    all_class_indices = [cls for cls in classes for _ in range(cali_samples_per_class)]
                    return torch.tensor(all_class_indices, device=device)
                elif mode == 'infer':
                    if is_eval:
                        n_samples_per_class = 30
                        all_class_indices = [cls for cls in classes for _ in range(n_samples_per_class)]
                    else:   # default: is_test
                        n_samples_per_class = 5
                        classes_test = [1, 7, 10, 174, 186, 333, 795, 980, 985, 989] 
                        all_class_indices = [cls for cls in classes_test for _ in range(n_samples_per_class)]
                    return torch.tensor(all_class_indices, device=device)
                else:
                    raise KeyError
            correct_batch_size = 50
            n_correct_batch = len(get_test_label('infer'))
            testdata_name='1000classes_3'
            if is_cali:
                dns_cali(   ddim_T=ddim_T, ddim_steps=ddim_steps, scale=scale, nbit_w = nbit_w, nbit_a=nbit_a, timesteps_range=timesteps_range, 
                timestamp=timestamp, testdata_name=testdata_name, net_name=net_name, base_dir=base_dir, cali_dir=cali_dir, device=device, noise_scheduler=noise_scheduler, latent_shape=latent_shape, 
                model_forward_tea_stu=model_forward_tea_stu, get_noisy_images=get_noisy_images, model_vae_decode=model_vae_decode,
                get_img_label=get_img_label, get_label_cfg=get_label_cfg, get_encoded_label=get_encoded_label, get_test_label=get_test_label,
                correct_batch_size=correct_batch_size, uni_weight = uni_weight, dcnc_method = dcnc_method)
            if is_eval or is_test:
                if is_test:
                    base_dir=test_dir
                dns_infer(   ddim_T=ddim_T, ddim_steps=ddim_steps, scale=scale, nbit_w = nbit_w, nbit_a=nbit_a, timesteps_range=timesteps_range, 
                timestamp=timestamp, testdata_name=testdata_name, net_name=net_name, base_dir=base_dir, cali_dir=cali_dir, device=device, noise_scheduler=noise_scheduler, latent_shape=latent_shape, 
                model_forward_tea_stu=model_forward_tea_stu, get_noisy_images=get_noisy_images, model_vae_decode=model_vae_decode,
                get_img_label=get_img_label, get_label_cfg=get_label_cfg, get_encoded_label=get_encoded_label, get_test_label=get_test_label,
                correct_batch_size=correct_batch_size, uni_weight = uni_weight, seed = seed, dcnc_method = dcnc_method)

                if not only_dns:
                    ori_test(   ddim_T=ddim_T, ddim_steps=ddim_steps, scale=scale, nbit_w = nbit_w, nbit_a=nbit_a, timesteps_range=timesteps_range, 
                    timestamp=timestamp, testdata_name=testdata_name, net_name=net_name, base_dir=base_dir, cali_dir=cali_dir, device=device, noise_scheduler=noise_scheduler, latent_shape=latent_shape, 
                    model_forward_tea_stu=model_forward_tea_stu, get_noisy_images=get_noisy_images, model_vae_decode=model_vae_decode,
                    get_img_label=get_img_label, get_label_cfg=get_label_cfg, get_encoded_label=get_encoded_label, get_test_label=get_test_label,
                    correct_batch_size=correct_batch_size, uni_weight = uni_weight, seed = seed)

            if is_test:
                from PIL import Image
                teacher_image_save_path = os.path.join(base_dir, f'{timestamp}/teacher_image')
                student_image_save_path = os.path.join(base_dir, f'{timestamp}/student_image')
                infer_image_save_path = os.path.join(base_dir, f'{timestamp}/dns_image')
                concat_image_save_path = os.path.join(base_dir, f'{timestamp}/concat_image')
                os.makedirs(concat_image_save_path, exist_ok=True)
                
                # 获取图片文件名列表（假设三个文件夹中的图片名称相同）
                image_files = sorted(os.listdir(teacher_image_save_path))
                
                # 每组5张图片
                for i in range(0, len(image_files), 5):
                    group_files = image_files[i:i+5]
                    if len(group_files) < 5:  # 如果最后一组不足5个，则跳过
                        continue
                    
                    # 获取第一张图片以确定尺寸
                    sample_img = Image.open(os.path.join(teacher_image_save_path, group_files[0]))
                    img_width, img_height = sample_img.size
                    
                    # 创建大图 (3列×5行)
                    concat_img = Image.new('RGB', (img_width * 3, img_height * 5))
                    
                    # 填充大图
                    for row, img_file in enumerate(group_files):
                        # 每行放置三个文件夹中的同名图片
                        teacher_img = Image.open(os.path.join(teacher_image_save_path, img_file))
                        student_img = Image.open(os.path.join(student_image_save_path, img_file))
                        infer_img = Image.open(os.path.join(infer_image_save_path, img_file))
                        
                        # 粘贴到大图的对应位置
                        concat_img.paste(teacher_img, (0, row * img_height))
                        concat_img.paste(student_img, (img_width, row * img_height))
                        concat_img.paste(infer_img, (img_width * 2, row * img_height))
                    
                    # 保存合成图片
                    group_index = i // 5
                    concat_img_path = os.path.join(concat_image_save_path, f'group_{group_index}.png')
                    concat_img.save(concat_img_path)
                            

                
