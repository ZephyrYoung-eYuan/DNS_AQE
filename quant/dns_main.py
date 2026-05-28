import os
import torch
import numpy as np 
# import matplotlib.pyplot as plt
# from scipy import stats
import gc

from .dns_func import dns_diff_calcu_loop, aqe_dns_loop, tea_ddim_cfg_loop, dns_slope_calcu, convert_to_image, compute_and_set_diffusion_variables
from diffusers.training_utils import set_seed

def print_inputs(func):
    def wrapper(*args, **kwargs):
        print(f"函数参数: args={args}, kwargs={kwargs}")
        return func(*args, **kwargs)
    return wrapper

# 使用装饰器
@print_inputs
def dns_cali(ddim_T, ddim_steps, scale, nbit_w, nbit_a, timesteps_range, 
             timestamp, testdata_name, net_name, base_dir, cali_dir, device, noise_scheduler, latent_shape, 
             model_forward_tea_stu, get_noisy_images, model_vae_decode,
             get_img_label, get_label_cfg, get_encoded_label, get_test_label, 
             correct_batch_size, is_var = False, is_flux = False, uni_weight = 1.0, torch_dtype = torch.float32, dcnc_method = 'uniform'):
    # locals().update(config)
    print("dns_cali")
    cfg = scale
    knc_title= f'dns_cfg{cfg}_ddim{ddim_steps}_uw{uni_weight}'
    if dcnc_method is not 'uniform':
        knc_title+=f'_{dcnc_method}'
    if is_var:
        compute_and_set_diffusion_variables(noise_scheduler, timesteps_range)
    timings = {}
    correct_batch_record = None
    timesteps_len = len(timesteps_range)

# region #? *****************************生成图片用于计算diff********************************#     
    block_start = time.perf_counter()
    if True:
        print("*****************************生成图片用于计算diff********************************")
        gen_img_npy_save_path = os.path.join(base_dir, f'{testdata_name}samples_and_xc.pt')
        os.makedirs(os.path.dirname(gen_img_npy_save_path), exist_ok=True)
        # 检测文件是否存在
        if os.path.exists(gen_img_npy_save_path):
            print(f"file {gen_img_npy_save_path} exists")
        else:
            print(f"file {gen_img_npy_save_path} not exists")
            cfg = scale
            # Prepare storage for all generated samples
            all_generated_samples = []
            all_class_labels = []
            # Generate samples in batches
            all_class_indices = get_test_label(mode='cali')
            correct_batch = len(all_class_indices)
            correct_batch_record = correct_batch
            n_correct_batch = int(correct_batch/correct_batch_size)
            assert correct_batch_size*n_correct_batch == correct_batch

            for batch_idx in range(n_correct_batch):
                print(f"Generating batch {batch_idx+1}/{n_correct_batch}")
                xc_ = all_class_indices[batch_idx * correct_batch_size: (batch_idx+1) * correct_batch_size]
                # xc_ = torch.tensor(batch_class_indices, device=device)
                xc = get_label_cfg(xc_) 
                c = get_encoded_label(xc)
                sample_images = torch.randn(correct_batch_size, *latent_shape, device=device, dtype=torch_dtype)
                
                # Run diffusion process
                for i, timestep in enumerate(timesteps_range):
                    # Get model output
                    model_output = model_forward_tea_stu(sample_images, timestep, mode='teacher', is_var=is_var, **{'c': c})
                    # Update samples
                    prev_timestep = timesteps_range[i + 1] if i < len(timesteps_range) - 1 else timestep-50
                    sample_images = tea_ddim_cfg_loop(timestep=timestep, model_output_tea=model_output, prev_timestep=prev_timestep,
                        sample_images=sample_images, cond_kwargs={'c': c}, noise_scheduler=noise_scheduler, device=device,
                        cfg=cfg, is_var=is_var, step_index = i, 
                    )

                    # Print progress
                    print('timestep:',timestep)

                # Store generated samples and their class labels
                all_generated_samples.append(sample_images.cpu())
                if isinstance(xc_, torch.Tensor): all_class_labels.append(xc_)                  
                elif isinstance(xc_, list): all_class_labels.extend(xc_)
                else: raise TypeError

            # Concatenate all generated samples and class labels
            all_generated_samples = torch.cat(all_generated_samples, dim=0)
            if isinstance(xc_, torch.Tensor): all_class_labels = torch.cat(all_class_labels, dim=0).cpu()   
            
            # Save results
            print(f"Saving {len(all_generated_samples)} samples to {gen_img_npy_save_path}")
            torch.save((all_generated_samples, all_class_labels), gen_img_npy_save_path)
            print("Done!")  
    timings["generate_images"] = time.perf_counter() - block_start
# region #? **********************计算不同timesteps下量化前后的diff***************************#     
    block_start = time.perf_counter()
    if True: # TODO w/o CFG
        print("**********************计算不同timesteps下量化前后的diff***************************")
        diff_result_dir = base_dir
        os.makedirs(diff_result_dir, exist_ok=True)
        diff_result1_save_path = os.path.join(diff_result_dir, f'{testdata_name}samples_results_{net_name}_w{nbit_w}a{nbit_a}_{int(timesteps_range[-1])}.npy')
        # 检测文件是否存在
        if os.path.exists(diff_result1_save_path):
            print(f"file {diff_result1_save_path} exists")
        else:
            print(f"file {diff_result1_save_path} not exists")

            file_path = os.path.join(base_dir, f'{testdata_name}samples_and_xc.pt')
            
            batch = torch.load(file_path)
            images, xc = get_img_label(batch)
            del batch
            # Generate samples in batches
            correct_batch = images.shape[0]
            correct_batch_record = correct_batch
            # correct_batch_size = 300
            n_correct_batch = int(correct_batch/correct_batch_size)
            assert correct_batch_size*n_correct_batch == correct_batch
            for i, timestep in enumerate(timesteps_range):
                print('timestep:',timestep)
                # 用于保存当前时间步的批次结果
                all_model_output_tea = []
                all_model_output_stu = []
                for batch_idx in range(n_correct_batch):
                    print(f"Generating batch {batch_idx+1}/{n_correct_batch}")
                    batch_images = images[batch_idx * correct_batch_size: (batch_idx+1) * correct_batch_size].to(device)   #python 多个维度时只对第一个维度切片
                    batch_xc = xc[batch_idx * correct_batch_size: (batch_idx+1) * correct_batch_size]   #可能是文字，不在这移device，在get_encoded_label里移动device
                    batch_encoded_images = batch_images
                    batch_c_save = get_encoded_label(get_label_cfg(batch_xc)) 
                    del batch_images, batch_xc
                    batch_result = dns_diff_calcu_loop(timestep, batch_encoded_images, batch_encoded_images, ddim_T, device, model_forward_tea_stu, get_noisy_images, **{'c':batch_c_save})
                    # 收集输出结果
                    all_model_output_tea.append(batch_result['model_output_tea'])
                    all_model_output_stu.append(batch_result['model_output_stu'])
                # 合并当前时间步的所有批次结果
                if n_correct_batch == 1:
                    combined_result = batch_result
                else:
                    combined_result = {
                        'timestep': timestep,
                        'model_output_tea': np.concatenate(all_model_output_tea),
                        'model_output_stu': np.concatenate(all_model_output_stu)
                    }
                # results.append(combined_result)
                diff_result_path = os.path.join(diff_result_dir, f'{testdata_name}samples_results_{net_name}_w{nbit_w}a{nbit_a}_{int(timestep)}.npy')
                # 保存所有结果
                np.save(diff_result_path, combined_result)
                print(f'Results saved to {diff_result_path}')
    timings["compute_diff"] = time.perf_counter() - block_start
# region #? **********************用量化前后的diff计算slope和int***************************#
    block_start = time.perf_counter()
    if True :
        print("**********************用量化前后的diff计算slope和int***************************")
        cali_save_path = os.path.join(cali_dir, timestamp)
        diff_result_path_str = [f'{base_dir}/{testdata_name}samples_results_{net_name}_w{nbit_w}a{nbit_a}_','.npy']
        # 创建保存图表的文件夹
        statistics_save_path = os.path.join(cali_save_path, f'{knc_title}_statistics_{net_name}_w{nbit_w}a{nbit_a}')   
        os.makedirs(statistics_save_path, exist_ok=True)
        # diff_result_path =f'{base_dir}/diff_results_{net_name}_w{nbit_w}a8_lastq_q8.npy'
        dns_slope_calcu(diff_result_path_str, statistics_save_path, timesteps_range = timesteps_range, cfg = scale, uni_weight = uni_weight, dcnc_method = dcnc_method)
    timings["compute_slope"] = time.perf_counter() - block_start

    os.makedirs(cali_dir, exist_ok=True)
    log_path = os.path.join(cali_dir, f'cali_time_{timestamp}.txt')
    latent_shape_info = tuple(latent_shape) if not isinstance(latent_shape, torch.Size) else tuple(latent_shape)
    correct_batch_info = correct_batch_record if correct_batch_record is not None else 'N/A'
    log_lines = [
        "Timing summary:",
        f"generate_images: {timings.get('generate_images', 0):.4f}s",
        f"compute_diff: {timings.get('compute_diff', 0):.4f}s",
        f"compute_slope: {timings.get('compute_slope', 0):.4f}s",
        "",
        f"latent_shape: {latent_shape_info}",
        f"correct_batch: {correct_batch_info}",
        f"timesteps_range_len: {timesteps_len}",
    ]
    print("\n".join(log_lines))
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"Timing log saved to {log_path}")


save_npz = True
@print_inputs
def dns_infer(ddim_T, ddim_steps, scale, nbit_w, nbit_a, timesteps_range, 
             timestamp, net_name, testdata_name, base_dir, cali_dir, device, noise_scheduler, latent_shape, 
             model_forward_tea_stu, get_noisy_images, model_vae_decode,
             get_img_label, get_label_cfg, get_encoded_label, get_test_label, 
             correct_batch_size, is_var = False, is_flux = False, uni_weight = 1.0, torch_dtype = torch.float32, seed = 14, dcnc_method = 'uniform'):
    print("dns_infer")
    cfg = scale
    knc_title= f'dns_cfg{cfg}_ddim{ddim_steps}_uw{uni_weight}'
    if dcnc_method is not 'uniform':
        knc_title+=f'_{dcnc_method}'
    if is_var:
        compute_and_set_diffusion_variables(noise_scheduler, timesteps_range)
    init_generator = torch.Generator(device=device)
    init_generator.manual_seed(seed)
# region #? ******************************DNS*******************************************#
    if True:
        print("******************************DNS*******************************************")
        set_seed(7)
        infer_image_save_path = os.path.join(base_dir, timestamp)
        os.makedirs(infer_image_save_path, exist_ok=True)
        cali_save_path = os.path.join(cali_dir, timestamp)
        os.makedirs(cali_save_path, exist_ok=True)
        os.makedirs(os.path.join(infer_image_save_path, f'{knc_title}_statistics_{net_name}_w{nbit_w}a{nbit_a}'), exist_ok=True)
        infer_image_save_path = os.path.join(infer_image_save_path, 'dns_image')
        os.makedirs(infer_image_save_path, exist_ok=True)
        statistics_save_timestamp = timestamp
        # statistics_save_timestamp = '20250402_075338'
        statistics_save_path = os.path.join(cali_dir, f'{statistics_save_timestamp}/{knc_title}_statistics_{net_name}_w{nbit_w}a{nbit_a}/correlation_results_by_timestep.npy')
        results = np.load(statistics_save_path, allow_pickle=True)
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
            sample_images_stu = torch.randn(correct_batch_size, *latent_shape, device=device, dtype=torch_dtype, generator=init_generator)
            # sample_images_stu = sample_images_stu.to(device)
            for i, timestep in enumerate(timesteps_range):
                prev_timestep = timesteps_range[i + 1] if i < len(timesteps_range) - 1 else timestep-50
                result_t = results[i]
                assert timestep == result_t['timestep'], f"{timestep} is not {result_t['timestep']}"
                diff_slope = result_t['slope_k']
                diff_intercept = result_t['intercept']
                epsilon_delta_t_sq=result_t['residual_std']**2
                uniform_scale = result_t['uniform_scale']
                sample_images_stu = aqe_dns_loop(timestep=timestep, diff_slope=diff_slope, diff_intercept=diff_intercept, epsilon_delta_t_sq=epsilon_delta_t_sq, prev_timestep=prev_timestep,
                    sample_images=sample_images_stu, cond_kwargs={'c':c}, model_forward_tea_stu=model_forward_tea_stu, noise_scheduler=noise_scheduler, device=device, 
                    uniform_scale = uniform_scale, cfg = cfg, is_var=is_var, step_index = i, ddim_steps=ddim_steps,
                )
                if is_var:
                    sample_images_stu, var_noise = sample_images_stu  

            for i in range(0, correct_batch_size, 100):
                sample_images_sub = sample_images_stu[i:i + 100]
                sample_images_sub = model_vae_decode(sample_images_sub)
                sample_images_sub = torch.clamp((sample_images_sub + 1)/2, min=0.0, max=1.0)

                if test_save_img_num<100:
                    test_save_img_num=test_save_img_num+25
                    nrow = 5
                    from torchvision.utils import save_image
                    save_image(sample_images_sub[:25,:,:,:].cpu(), os.path.join(cali_dir, f'{timestamp}/{knc_title}_statistics_{net_name}_w{nbit_w}a{nbit_a}/dns_infer_image_{test_save_img_num}.png'), nrow=nrow)

                sample_images_sub = sample_images_sub.permute(0, 2, 3, 1).cpu().numpy()
                num_images = sample_images_sub.shape[0]
                # 保存每张图像
                for j in range(num_images):
                    img = convert_to_image(sample_images_sub[j])
                    img.save(os.path.join(infer_image_save_path, f"img_{idx:05d}.png"))
                    idx = idx+1
                # sample_images_sub = (sample_images_sub * 255).astype(np.uint8)
                # all_images.append(sample_images_sub)
            
            # 批次结束后清理
            torch.cuda.empty_cache()
            gc.collect()
            # 连接所有批次的图像
        # images_array = np.concatenate(all_images, axis=0)
        
        # # 保存为npz文件
        # output_npz_path = os.path.join(infer_image_save_path, f"img_cls_compressed.npz")
        # np.savez_compressed(output_npz_path, arr_0=images_array)
        # print(f"已保存NPZ文件到 {output_npz_path}, 形状: {images_array.shape}")

