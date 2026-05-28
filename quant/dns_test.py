import os
import torch
import numpy as np 
# import matplotlib.pyplot as plt
# from scipy import stats
import gc

from .dns_func import dns_diff_calcu_loop, aqe_dns_loop, tea_ddim_cfg_loop, dns_slope_calcu, convert_to_image, compute_and_set_diffusion_variables
from diffusers.training_utils import set_seed

def ori_test(ddim_T, ddim_steps, scale, nbit_w, nbit_a, timesteps_range, 
             timestamp, testdata_name, net_name, base_dir, cali_dir, device, noise_scheduler, latent_shape, 
             model_forward_tea_stu, get_noisy_images, model_vae_decode,
             get_img_label, get_label_cfg, get_encoded_label, get_test_label, 
             correct_batch_size, is_var = False, uni_weight = 1.0, torch_dtype = torch.float32):
    print("ori_test")
    cfg = scale   
    knc_title= f'dns_cfg{cfg}_ddim{ddim_steps}_uw{uni_weight}'
    if is_var:
        compute_and_set_diffusion_variables(noise_scheduler, timesteps_range)
    # init_generator = torch.Generator(device=device)
    # init_generator.manual_seed(14)
    all_class_indices = get_test_label('infer')
    correct_batch = len(all_class_indices)
    n_correct_batch = int(correct_batch/correct_batch_size)
    assert correct_batch_size*n_correct_batch == correct_batch
    cali_save_path = os.path.join(cali_dir, timestamp)
    # 创建保存图表的文件夹
    statistics_save_path = os.path.join(cali_save_path, f'{knc_title}_statistics_{net_name}_w{nbit_w}a{nbit_a}')   
    os.makedirs(statistics_save_path, exist_ok=True)
# region #? ******************************ori_test*****************************************#
    # try:    
    if True:   
        print("******************************ori_test*****************************************")
        modes = ['teacher', 'student']
        # modes = ['teacher']
        if True:
            for mode in modes:
                print('mode:', mode)
                set_seed(7)
                init_generator = torch.Generator(device=device)
                init_generator.manual_seed(seed)
                if True:
                    test_image_save_path = os.path.join(base_dir, f'{timestamp}/{mode}_image')
                    os.makedirs(test_image_save_path, exist_ok=True)
                    # Generate samples in batches
                    all_class_indices = get_test_label('infer')
                    correct_batch = len(all_class_indices)
                    n_correct_batch = int(correct_batch/correct_batch_size)
                    assert correct_batch_size*n_correct_batch == correct_batch
                    # all_images = []
                    test_save_img_num = 0
                    idx = 0
                    for batch_idx in range(n_correct_batch):
                        print(f"Generating batch {batch_idx+1}/{n_correct_batch}")
                        xc_ = all_class_indices[batch_idx * correct_batch_size: (batch_idx+1) * correct_batch_size]
                        # xc_ = torch.tensor(batch_class_indices, device=device)
                        xc = get_label_cfg(xc_) 
                        c = get_encoded_label(xc)
                        sample_images = torch.randn(correct_batch_size, *latent_shape, device=device, dtype=torch_dtype, generator=init_generator)
                        # Run diffusion process
                        for i, timestep in enumerate(timesteps_range):
                            # Get model output
                            model_output = model_forward_tea_stu(sample_images, timestep, mode=mode, is_var=is_var, **{'c': c})
                            # Update samples
                            prev_timestep = timesteps_range[i + 1] if i < len(timesteps_range) - 1 else timestep-50
                            sample_images = tea_ddim_cfg_loop(timestep=timestep, model_output_tea=model_output, prev_timestep=prev_timestep,
                                sample_images=sample_images, cond_kwargs={'c': c}, noise_scheduler=noise_scheduler, device=device,
                                cfg=cfg, is_var=is_var, step_index = i, 
                            )
                        
                        
                        for i in range(0, correct_batch_size, 100):
                            sample_images_sub = sample_images[i:i + 100]
                            sample_images_sub = model_vae_decode(sample_images_sub)
                            sample_images_sub = torch.clamp((sample_images_sub + 1)/2, min=0.0, max=1.0)

                            if test_save_img_num<100:
                                test_save_img_num=test_save_img_num+25
                                nrow = 5
                                from torchvision.utils import save_image
                                save_image(sample_images_sub[:25,:,:,:].cpu(), os.path.join(statistics_save_path, f'{mode}_test_image_{test_save_img_num}.png'), nrow=nrow)

                            sample_images_sub = sample_images_sub.permute(0, 2, 3, 1).cpu().numpy()
                            num_images = sample_images_sub.shape[0]
                            # 保存每张图像
                            for i in range(num_images):
                                img = convert_to_image(sample_images_sub[i])
                                img.save(os.path.join(test_image_save_path, f"img_{idx:05d}.png"))
                                idx = idx+1
                            # sample_images_sub = (sample_images_sub * 255).astype(np.uint8)
                            # all_images.append(sample_images_sub)
                        # 批次结束后清理
                        torch.cuda.empty_cache()
                        gc.collect()
                        # 连接所有批次的图像
                    # images_array = np.concatenate(all_images, axis=0)
                    
                    # # 保存为npz文件
                    # # output_npz_path = os.path.join(test_image_save_path, f"img_cls_compressed.npz")
                    # np.savez_compressed(output_npz_path, arr_0=images_array, classes=xc_.cpu().numpy())
                    # print(f"已保存NPZ文件到 {output_npz_path}, 形状: {images_array.shape}")
                    # del all_images, images_array

