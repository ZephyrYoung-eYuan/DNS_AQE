import os
import numpy as np
import torch    
from scipy import stats
import matplotlib.pyplot as plt
from PIL import Image

def convert_to_image(img_array):
    # 确保图像范围在0到1之间，多余的截断
    img = img_array.clip(0, 1)
    # 从[0, 1]范围转换到[0, 255]
    img = (img * 255.0).astype(np.uint8)
    # 将通道维度从第一维移到最后一维 (C,H,W) -> (H,W,C)，如果需要的话
    if len(img_array.shape) == 3 and img_array.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    return Image.fromarray(img)
def compute_and_set_diffusion_variables(noise_scheduler, timesteps_range):
    # 确保所有输入都是float64
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(torch.float64)
    
    # 创建时间步映射字典
    timestep_map = {}
    for i in range(len(timesteps_range)):
        current_t = timesteps_range[i]
        if i < len(timesteps_range) - 1:
            prev_t = timesteps_range[i+1]
        else:
            prev_t = -1
        timestep_map[current_t] = prev_t
    
    # 为每个时间步创建alphas_cumprod_prev
    alphas_cumprod_prev = torch.zeros_like(alphas_cumprod)
    
    # 对于每个时间步
    for t in range(len(alphas_cumprod)):
        if t in timestep_map:
            prev_t = timestep_map[t]
            if prev_t >= 0:
                alphas_cumprod_prev[t] = alphas_cumprod[prev_t]
            else:
                alphas_cumprod_prev[t] = torch.tensor(1.0, device=alphas_cumprod.device, dtype=torch.float64)
        else:
            if t > 0:
                alphas_cumprod_prev[t] = alphas_cumprod[t-1]
            else:
                alphas_cumprod_prev[t] = torch.tensor(1.0, device=alphas_cumprod.device, dtype=torch.float64)
    
    # 使用新公式计算beta
    new_betas = torch.zeros_like(alphas_cumprod)
    for t in range(len(alphas_cumprod)):
        new_betas[t] = 1.0 - alphas_cumprod[t] / alphas_cumprod_prev[t]
    
    # 重新计算alphas
    new_alphas = 1.0 - new_betas
    
    # 计算其余变量
    sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)
    
    # 使用新的beta值进行后验计算
    posterior_variance = new_betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    # 使用一个合适的默认值来避免log(0)
    safe_posterior_variance = torch.clamp(posterior_variance, min=1e-20)
    posterior_log_variance_clipped = torch.log(safe_posterior_variance)
    posterior_log_variance_clipped[timesteps_range[-1]]=posterior_log_variance_clipped[timesteps_range[-2]]
    
    posterior_mean_coef1 = new_betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(new_alphas) / (1.0 - alphas_cumprod)
    
    # 设置计算的变量到noise_scheduler中
    noise_scheduler.alphas_cumprod_prev = alphas_cumprod_prev
    noise_scheduler.betas = new_betas  # 存储新计算的beta值
    noise_scheduler.new_alphas = new_alphas  # 存储对应的alpha值
    noise_scheduler.posterior_log_variance_clipped = posterior_log_variance_clipped
    noise_scheduler.posterior_mean_coef1 = posterior_mean_coef1
    noise_scheduler.sqrt_recip_alphas_cumprod = sqrt_recip_alphas_cumprod
    noise_scheduler.posterior_mean_coef2 = posterior_mean_coef2
    noise_scheduler.sqrt_recipm1_alphas_cumprod = sqrt_recipm1_alphas_cumprod

            # noise_scheduler.alphas_cumprod_prev = alphas_cumprod_prev

def dns_diff_calcu(encoded_images, ddim_T, device, model_forward_tea_stu, get_noisy_images, diff_result_path, **cond_kwargs):
    results = []
    for i in range(ddim_T):
        return_t = dns_diff_calcu_loop(i, encoded_images, encoded_images, ddim_T, device, model_forward_tea_stu, get_noisy_images, diff_result_path, **cond_kwargs)
        results.append(return_t)

    # 保存所有结果
    np.save(diff_result_path, results)
    print(f'Results saved to {diff_result_path}')

def dns_diff_calcu_loop(i, encoded_images_tea, encoded_images_stu, ddim_T, device, model_forward_tea_stu, get_noisy_images, **cond_kwargs):
    timestep = i
    noise = torch.randn_like(encoded_images_stu)

    noisy_images_tea = get_noisy_images(encoded_images_tea, noise, timestep, **cond_kwargs)
    noisy_images_stu = get_noisy_images(encoded_images_stu, noise, timestep, **cond_kwargs)
    del noise

    model_output_tea = model_forward_tea_stu(noisy_images_tea, timestep, mode = 'teacher', **cond_kwargs)
    model_output_stu = model_forward_tea_stu(noisy_images_stu, timestep, mode = 'student', **cond_kwargs)
    
    # 计算两个网络输出的差值，并进行统计分析
    diff = model_output_stu - model_output_tea
    # 计算统计量
    diff_mean = diff.mean().item()
    diff_std = diff.std().item()
    diff_min = diff.min().item()
    diff_max = diff.max().item()
    # 存储结果
    diff_np = diff.detach().cpu().float().numpy()
    model_output_tea_np = model_output_tea.detach().cpu().float().numpy()
    model_output_stu_np = model_output_stu.detach().cpu().float().numpy()
    return {
            'timestep': i,
            'diff': diff_np,
            'model_output_tea': model_output_tea_np,
            'model_output_stu': model_output_stu_np,
            'diff_mean': diff_mean,
            'diff_std': diff_std,
            'diff_min': diff_min,
            'diff_max': diff_max
        }

def dns_slope_calcu(diff_result_path_str, statistics_save_path, timesteps_range, cfg = None, uni_weight = 1.0, dcnc_method = 'uniform'):
    results = []
    save_kurtosis = []
    save_residual_std = []
    save_slope_k = []
    save_residual_skew = []
    save_residual_m5 = []
    save_residual_m6 = []
    # 按时间步顺序处理数据
    for i, timestep in enumerate(timesteps_range):
        print("dns_slope_calcu_loop, timestep:",timestep)
        diff_result_path = diff_result_path_str[0]+str(int(timestep))+diff_result_path_str[1]

        result_t = np.load(diff_result_path, allow_pickle=True).item()
        result_t = dns_slope_calcu_loop(timestep, result_t, statistics_save_path, cfg, uni_weight, dcnc_method=dcnc_method)
        results.append(result_t)
        # 创建新字典，包含所有原始内容
        filtered_dict = result_t.copy()
        filtered_dict['slope_k'] = filtered_dict['slope_k'].shape
        filtered_dict['intercept'] = filtered_dict['intercept'].shape
        print(filtered_dict)
        save_kurtosis.append(result_t['kurtosis_res'])
        save_residual_std.append(result_t['residual_std'])
        save_slope_k.append(result_t['mean_slope_k'])
        save_residual_skew.append(result_t['residual_skew'])
        save_residual_m5.append(result_t['residual_m5'])
        save_residual_m6.append(result_t['residual_m6'])

    npy_file = os.path.join(statistics_save_path, 'correlation_results_by_timestep.npy')
    np.save(npy_file, results)
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    for ax, data, title in zip(axs, 
            [save_residual_std, save_slope_k, save_kurtosis],
            ['Residual STD', 'Slope', 'Kurtosis']):
        ax.plot(timesteps_range, data)
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
    plt.tight_layout()
    plt.savefig(os.path.join(statistics_save_path, "statistics_plot.png"), dpi=300, bbox_inches='tight')

    fig2, axs2 = plt.subplots(3, 1, figsize=(10, 12))
    titles = [
        'Residual Skewness',
        'Residual 5th Central Moment',
        'Residual 6th Central Moment'
    ]
    datas = [save_residual_skew, save_residual_m5, save_residual_m6]
    ylims = [(-0.5, 4.0), (-0.5, 4.0), None]  # 前两个固定范围，六阶不限制

    for ax, data, title, ylim in zip(axs2, datas, titles, ylims):
        ax.plot(timesteps_range, data)
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        if ylim is not None:
            ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.savefig(
        os.path.join(statistics_save_path, "statistics_moments_plot.png"),
        dpi=300,
        bbox_inches='tight'
    )

def estimate_std_by_iqr(noise):
    # IQR也对异常值较稳健
    q75, q25 = np.percentile(noise, [75, 25])
    iqr = q75 - q25
    # 对于高斯分布，标准差≈IQR/1.349
    variance = (iqr/1.349)
    return variance
def estimate_std(noise, method):
    if method == 'iqr':
        return estimate_std_by_iqr(noise)
    elif method == 'std':
        return np.std(noise)
    else:
        raise ValueError("不支持的方法")
    
def calculate_metrics(x, y):
    slope, intercept, r_value, p_value_reg, std_err = stats.linregress(x.flatten(), y.flatten())
    # correlation, p_value = stats.pearsonr(x.flatten(), y.flatten())
    return {
        'slope': slope,
        'intercept': intercept,
    }

def calculate_residuals(tea_outputs, diff, method='channal'):
    batch_size, channels, height, width = diff.shape
    residuals = []
    slopes = []
    intercepts = []
    
    for c in range(channels):
        metrics = calculate_metrics(
            tea_outputs[:, c, :, :],
            diff[:, c, :, :]
        )
        slopes.append(metrics['slope'])
        intercepts.append(metrics['intercept'])
        channel_residuals = diff[:, c, :, :].flatten() - (
            metrics['slope'] * tea_outputs[:, c, :, :].flatten() + metrics['intercept']
        )
        residuals.extend(channel_residuals)

    residuals = np.array(residuals)
    # 创建形状为[channels]的数组，可以广播到[batch, channels, height, width]
    diff_slope = np.array(slopes).reshape(1, channels, 1, 1)
    diff_intercept = np.array(intercepts).reshape(1, channels, 1, 1)
    
    stats_dict = {
        'diff_slope': diff_slope,
        'diff_intercept': diff_intercept
    }
        
    return residuals, stats_dict

import pickle
def dns_slope_calcu_loop(timestep, entry, output_folder, cfg = None, uni_weight = 1.0, std_method = 'iqr', dcnc_method = 'uniform'):
    # 只收集当前时间步的数据
    model_output_stu = entry['model_output_stu']
    model_output_tea = entry['model_output_tea']

    if not isinstance(model_output_stu, np.ndarray) or not isinstance(model_output_tea, np.ndarray):
        raise TypeError

    if cfg is None:
        diff = model_output_stu - model_output_tea
        model_output_stu_cfg = model_output_stu
        model_output_tea_cfg = model_output_tea
    else:
        noise_cond, noise_uncond = np.array_split(model_output_stu, 2, axis=0)
        model_output_stu_cfg = noise_uncond + cfg * (noise_cond - noise_uncond)

        noise_cond, noise_uncond = np.array_split(model_output_tea, 2, axis=0)
        model_output_tea_cfg = noise_uncond + cfg * (noise_cond - noise_uncond)
        diff = model_output_stu_cfg - model_output_tea_cfg
    residuals, stats_dict = calculate_residuals(model_output_tea_cfg, diff)    
    slope, intercept = stats_dict['diff_slope'], stats_dict['diff_intercept']

    mean_slope = np.mean(slope)
    kurtosis_res_old_before_clip = stats.kurtosis(residuals, fisher=True)
    residual_std_old = np.std(residuals)
    residual_std_old_save = residual_std_old
    residuals = np.clip(residuals, -6*1.29*residual_std_old, 6*1.29*residual_std_old) # 异常值裁剪，sqrt(峰度/3)：这种情况下，峰度5的修正因子约为1.29
    residual_std_old = estimate_std(residuals, std_method) 
    del diff
    kurtosis_res_old = stats.kurtosis(residuals, fisher=True)
    residual_skew_old = stats.skew(residuals, bias=False)
    centered = residuals - residuals.mean()
    residual_m5_old = np.mean(centered**5)
    residual_m6_old = np.mean(centered**6)
    residuals_before_dcnc = residuals.copy()

    uniform_scale = 0
    method = dcnc_method
    if True and kurtosis_res_old>0:
        # 默认值：不加噪声，仅重估 std
        dcnc_noise = 0.0
        is_math = False

        if method in ('none', 'off', 'no'):
            # 显式关闭 DCNC：不上噪声，下面直接用估计 std
            pass
        else:
            # 4 种 DCNC 噪声分布的超峰度（excess kurtosis）
            # 1) U(-a,a)： κ = -6/5
            # 2) Triangular(-a,0,a)： κ = -3/5
            # 3) Bimodal Gaussian: 0.5 N(-m,s^2) + 0.5 N(m,s^2)，取 r = m^2/s^2 = 1 => κ = -0.5
            # 4) Generalized Gaussian (exponential power), beta_gg = 4:
            #    理论标准峰度 K ≈ 2.1886 ⇒ κ ≈ -0.8114
            if method == 'uniform':
                kappa_noise = -6.0 / 5.0
            elif method == 'triangular':
                kappa_noise = -3.0 / 5.0
            elif method == 'bimodalgs':
                kappa_noise = -0.5
            elif method == 'generalizedgs':
                kappa_noise = -0.8114
            else:
                raise ValueError(f'Unknown dcnc_method: {dcnc_method}')

            # 针对不同噪声分布的解析目标方差（对应 uni_weight = 1）：
            # Var(noise) = σ0^2 * sqrt(-κ0 / κ_noise)
            uniform_var = residual_std_old**2 * np.sqrt(-kurtosis_res_old / kappa_noise)
            is_math = True

            # 根据不同分布，配置噪声参数并采样
            if method == 'uniform':
                # U(-a, a), Var = a^2 / 3
                uniform_scale = (uniform_var*3)**0.5*uni_weight
                dcnc_noise = np.random.uniform(
                    low=-uniform_scale,
                    high=uniform_scale,
                    size=residuals.shape
                )

            elif method == 'triangular':
                # Triangular(-a, 0, a), Var = a^2 / 6
                uniform_scale = (6.0 * uniform_var) ** 0.5 * uni_weight
                dcnc_noise = np.random.triangular(
                    left=-uniform_scale,
                    mode=0.0,
                    right=uniform_scale,
                    size=residuals.shape
                )

            elif method == 'bimodalgs':
                # 双峰高斯：0.5 N(-m, s^2) + 0.5 N(m, s^2)
                # 取 r = m^2 / s^2 = 1，此时 κ = -0.5
                # 总方差：Var = m^2 + s^2 = (1 + r) s^2 = 2 s^2
                # 令 Var(noise) = uniform_var * uni_weight^2
                r = 1.0
                target_var = uniform_var * (uni_weight ** 2)
                component_var = target_var / (1.0 + r)  # s^2
                s = component_var ** 0.5
                m = (r * component_var) ** 0.5
                signs = np.random.choice([-1.0, 1.0], size=residuals.shape)
                dcnc_noise = np.random.normal(
                    loc=signs * m,
                    scale=s,
                    size=residuals.shape
                )
                # 返回一个代表尺度的量（这里取组件标准差）
                uniform_scale = s

            elif method == 'generalizedgs':
                # 广义高斯（指数幂分布），beta_gg = 4
                # SciPy 中的 gennorm: Var = C_var * scale^2, 其中
                # C_var = Γ(3/4) / Γ(1/4) ≈ 0.337989
                beta_gg = 4.0
                GG_VAR_COEF = 0.337989  # 预计算的常数
                target_var = uniform_var * (uni_weight ** 2)
                alpha = (target_var / GG_VAR_COEF) ** 0.5  # scale 参数
                dcnc_noise = stats.gennorm.rvs(
                    beta_gg,
                    loc=0.0,
                    scale=alpha,
                    size=residuals.shape
                )
                uniform_scale = alpha

        # 加上补偿噪声（若 method 为 none/off/no，则 dcnc_noise 为 0）
        residuals = residuals + dcnc_noise
        if is_math:
            # 统一用解析方差：Var_new = Var_old + uniform_var（对应 uni_weight = 1）
            residual_std = (residual_std_old**2 + uniform_var) ** 0.5
        else:
            residual_std = estimate_std(residuals, std_method)
        residuals = np.clip(residuals, -6*residual_std, 6*residual_std)
        kurtosis_res = stats.kurtosis(residuals, fisher=True)
    else:
        residual_std = residual_std_old
        kurtosis_res = kurtosis_res_old

    # ====== 计算最终残差分布的偏度、五阶和六阶中心矩 ======
    residual_skew = stats.skew(residuals, bias=False)
    centered = residuals - residuals.mean()
    residual_m5 = np.mean(centered**5)
    residual_m6 = np.mean(centered**6)

    # ====== 画 DCNC 前后残差直方图（同一张图）并保存 ======
    os.makedirs(output_folder, exist_ok=True)

    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14
    })

    # 第一步：绘制 DCNC 前残差
    plt.figure(figsize=(8, 5))
    fig1 = plt.gcf()
    ax1 = plt.gca()

    bin_edges1 = np.linspace(-6.1 * residual_std_old, 6.1 * residual_std_old, 101)
    plt.hist(
        residuals_before_dcnc,
        bins=bin_edges1,
        alpha=0.6,
        density=True,
        color='blue',
        label='before DCNC'
    )

    # 第二步：绘制 DCNC 后残差
    plt.figure(fig1.number)
    bin_edges2 = np.linspace(-6.1 * residual_std, 6.1 * residual_std, 101)
    plt.hist(
        residuals,
        bins=bin_edges2,
        alpha=0.6,
        density=True,
        color='red',
        label='after DCNC'
    )

    plt.xlim(left=-6.1 * residual_std, right=6.1 * residual_std)
    plt.xlabel('residual (noise ε)')
    plt.ylabel('freq')

    plt.title(
        f'T:{timestep}, '
        f'k:{kurtosis_res_old_before_clip:.2f}, {kurtosis_res_old:.2f}, {kurtosis_res:.2f}'
    )
    plt.legend()

    residuals_file = os.path.join(output_folder, f'timestep_{timestep}_residuals.png')
    plt.savefig(residuals_file, dpi=300)

    pickle_path = os.path.join(output_folder, f'timestep_{timestep}_residuals.pkl')
    with open(pickle_path, 'wb') as file:
        pickle.dump(fig1, file)

    plt.close(fig1)

    # ====== 返回结果 ======
    return {
        'timestep': timestep,
        'slope_k': slope,
        'intercept': intercept,
        'mean_slope_k': mean_slope,
        'uniform_scale': uniform_scale,   # 各方法对应的尺度参数
        'residual_std': residual_std,
        'kurtosis_res': kurtosis_res,

        # 新增的三个统计量（最终残差分布的）
        'residual_std_old_save': residual_std_old_save,
        'residual_skew': residual_skew,
        'residual_m5': residual_m5,
        'residual_m6': residual_m6,
        'residual_skew_old': residual_skew_old,
        'residual_m5_old': residual_m5_old,
        'residual_m6_old': residual_m6_old,
    }
    
import numpy as np
from scipy.optimize import minimize_scalar, root

def solve_sigma_pai(sigma, sigma_tau, epsilon_delta_t_sq, eta):
    def equation(sigma_next):
        """计算残差方程"""
        dt = sigma_next-sigma
        C2=(1-sigma_next)+(sigma_next**2+(dt**2*epsilon_delta_t_sq))**0.5
        residual = (C2 + sigma_next-1)/C2 - sigma_tau
        return residual
    
    # 使用minimize_scalar寻找解，限制在alpha_tau和1之间
    def objective(x):
        return equation(x)**2  # 最小化残差的平方
    
    result = minimize_scalar(objective, bounds=(sigma_tau, 1), method='bounded')
    
    # 检查解的质量
    if objective(result.x) > eta**2:
        # 如果精度不够，尝试用root方法从alpha_tau附近开始求解
        best_x = result.x
        min_error = objective(result.x)
        
        # 从alpha_tau附近开始，逐渐向1靠近
        for guess in np.linspace(sigma_tau, min(sigma_tau*1.1, 0.99), 50):
            try:
                res = root(equation, guess)
                if res.success and sigma_tau <= res.x[0] <= 1:
                    error = objective(res.x[0])
                    # 优先选择接近alpha_tau的解
                    if error < min_error :
                        min_error = error
                        best_x = res.x[0]
                        if objective(result.x) < eta**2:
                            break
            except:
                continue
        return best_x
    return result.x

def solve_alpha_prod_pai(alpha_prod_t, alpha_prod_tau, epsilon_delta_t_sq, eta):
    """
    求解alpha_prod_t_prev的数值解
    
    参数:
    alpha_prod_tau: float, 已知参数
    alpha_prod_t: float, 已知参数
    epsilon_delta_t_sq: float, 已知参数
    
    返回:
    alpha_prod_t_prev: float, 求解得到的值，范围在alpha_tau到1之间，通常接近alpha_tau
    """
    
    def equation(alpha_prod_t_prev):
        """计算残差方程"""
        # 计算C1
        C1 = np.sqrt(1 - alpha_prod_t_prev) - np.sqrt((alpha_prod_t_prev * (1 - alpha_prod_t)) / alpha_prod_t)
        ws = alpha_prod_t_prev ** (0.5) / alpha_prod_t ** (0.5)

        # 计算C2_sq
        C2_sq = 1 + (C1 ** 2) * epsilon_delta_t_sq
        
        # 根据方程alpha_tau = alpha_prod_t_prev / C2_sq计算残差
        residual = alpha_prod_tau * C2_sq - alpha_prod_t_prev
        
        return residual
    
    # 使用minimize_scalar寻找解，限制在alpha_tau和1之间
    def objective(x):
        return equation(x)**2  # 最小化残差的平方
    
    result = minimize_scalar(objective, bounds=(alpha_prod_tau, 1), method='bounded')
    
    # 检查解的质量
    if objective(result.x) > eta**2:
        # 如果精度不够，尝试用root方法从alpha_tau附近开始求解
        best_x = result.x
        min_error = objective(result.x)
        
        # 从alpha_tau附近开始，逐渐向1靠近
        for guess in np.linspace(alpha_prod_tau, min(alpha_prod_tau*1.1, 0.99), 50):
            try:
                res = root(equation, guess)
                if res.success and alpha_prod_tau <= res.x[0] <= 1:
                    error = objective(res.x[0])
                    # 优先选择接近alpha_tau的解
                    if error < min_error :
                        min_error = error
                        best_x = res.x[0]
                        if objective(result.x) < eta**2:
                            break
            except:
                continue
        return best_x
    
    return result.x


def aqe_dns_loop(timestep, diff_slope, diff_intercept, epsilon_delta_t_sq, prev_timestep,
        sample_images, cond_kwargs, model_forward_tea_stu, noise_scheduler, device, 
        uniform_scale, cfg, is_var=False,
        step_index = 0, 
        ddim_steps = 20, 
    ):
    noise = None
    alphas_cumprod = noise_scheduler.alphas_cumprod.to('cpu')  # 已经是一个 numpy 数组
    prev_timestep = prev_timestep
    t_d=timestep-prev_timestep
    alpha_prod_t = alphas_cumprod[timestep].to(torch.float64)
    diff_slope = torch.tensor(diff_slope, device='cpu', dtype = torch.float64)
    diff_intercept = torch.tensor(diff_intercept, device='cpu', dtype = torch.float64)
    epsilon_delta_t_sq = torch.tensor(epsilon_delta_t_sq, device='cpu', dtype = torch.float64)

    epsilon_delta_t_sq = 1/(1+diff_slope)**2*epsilon_delta_t_sq     # diff = k*et+b 后的补偿
    epsilon_delta_t_sq = torch.mean(epsilon_delta_t_sq)             #TODO 这里如果这么估算斜率会导致这部分要取均值

    if prev_timestep>=t_d:
        alpha_prod_t_prev_tmp = alphas_cumprod[prev_timestep].to(torch.float64) if prev_timestep >= 0 else noise_scheduler.final_alpha_cumprod.to(torch.float64)
        alpha_prod_t_prev = solve_alpha_prod_pai(alpha_prod_tau=alpha_prod_t_prev_tmp.item(), alpha_prod_t=alpha_prod_t.item(), epsilon_delta_t_sq=epsilon_delta_t_sq.item(), eta = 1e-20)
        alpha_prod_t_prev = torch.tensor(alpha_prod_t_prev, device='cpu', dtype = torch.float64)
    else:
        alpha_prod_t_prev = alphas_cumprod[prev_timestep].to(torch.float64) if prev_timestep >= 0 else noise_scheduler.final_alpha_cumprod.to(torch.float64)
    # 1.2 计算去噪过程的权重
    ws = alpha_prod_t_prev ** (0.5) / alpha_prod_t ** (0.5)
    C1 = (1 - alpha_prod_t_prev) ** (0.5) - ((alpha_prod_t_prev * (1 - alpha_prod_t)) / alpha_prod_t) ** (0.5) #通过还原x_hat并带入得到的系数
    # C1 = -alpha_prod_t_prev** (0.5) * beta /(alpha_prod_t*(1-alpha_prod_t))**0.5
    we = (1 - alpha_prod_t_prev) ** (0.5) - ((alpha_prod_t_prev * (1 - alpha_prod_t)) / alpha_prod_t) ** (0.5) #通过还原x_hat并带入得到的系数
    if step_index>=ddim_steps-1:
        uniform_scale = 0   # 最后一步不用加噪补偿
    # 2.去噪    
    mode_in = "student"
    model_output = model_forward_tea_stu(sample_images, timestep, 
                                         mode = mode_in, uniform_scale = uniform_scale, is_var=is_var, **cond_kwargs)
    if is_var:
        # 解包model_output_tea
        model_output, model_log_variance = model_output
        model_log_variance,_ = torch.chunk(model_log_variance, 2, dim=0)
    model_output = get_model_output_cfg(model_output, cfg)

    model_output = (model_output-diff_intercept.to(device = model_output.device, dtype = model_output.dtype))/(1+diff_slope.to(device = model_output.device, dtype = model_output.dtype))   # diff = k*et+b
    
    if is_var:
        C1 = -noise_scheduler.posterior_mean_coef1[timestep] * \
            noise_scheduler.sqrt_recipm1_alphas_cumprod[timestep]
        we = C1
        # 计算确定性部分（均值）
        mean = ws * sample_images + we * model_output[:,:4,:,:]
        
        # 添加随机噪声
        noise = torch.randn_like(sample_images).to(sample_images.device)
        t=torch.tensor(timestep, device=sample_images.device)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(sample_images.shape) - 1))).to(sample_images.device)
        
        # 使用model_log_variance计算最终结果
        sample_images = mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    else:
        sample_images = ws * sample_images + we * model_output[:,:4,:,:]
    # 3. 根据给定公式依次计算
    # 3.1
    C2_sq = 1 + (C1 ** 2) * (epsilon_delta_t_sq)
    alpha_tau_solve = alpha_prod_t_prev / C2_sq

    # 3.2 在 alphas_cumprod 数列中找到值最接近 alpha_tau_solve 的序号
    differences = alphas_cumprod - alpha_tau_solve
    indices = np.where(differences >= 0)[0]
    if indices.size == 0:
        if is_var:
            return sample_images, noise
        else:
            return sample_images
    tau_timestep = indices[-1]
    timestep=tau_timestep
    
    alpha_prod_tau = alphas_cumprod[tau_timestep].to(torch.float64) 
    C3_sq=alpha_prod_t_prev/alpha_prod_tau
    if step_index>=ddim_steps-1:
        C3_sq =1 
    sample_images = sample_images/(C3_sq)** (0.5)
    if is_var:
        return sample_images, noise
    else:
        return sample_images


def tea_ddim_cfg_loop(timestep, model_output_tea, prev_timestep,
        sample_images, cond_kwargs, noise_scheduler, device, 
        cfg=3,
        is_var = False,
        ddim_eta = 0,
        noise = None,
        step_index = None, 
    ):
    alphas_cumprod = noise_scheduler.alphas_cumprod.to('cpu')
    alpha_prod_t = alphas_cumprod[timestep].to(torch.float64)
    if alpha_prod_t==0: alpha_prod_t=2.4e-9
    alpha_prod_t_prev = alphas_cumprod[prev_timestep].to(torch.float64) if prev_timestep >= 0 else noise_scheduler.final_alpha_cumprod.to(torch.float64)
    # 1.2 计算去噪过程的权重
    ws = alpha_prod_t_prev ** (0.5) / alpha_prod_t ** (0.5)
    C1 = (1 - alpha_prod_t_prev) ** (0.5) - ((alpha_prod_t_prev * (1 - alpha_prod_t)) / alpha_prod_t) ** (0.5) #通过还原x_hat并带入得到的系数
    we = C1
    
    if is_var:
        # 解包model_output_tea
        model_output_tea, model_log_variance = model_output_tea
        model_log_variance,_ = torch.chunk(model_log_variance, 2, dim=0)
    model_output = get_model_output_cfg(model_output_tea, cfg)
    if is_var:
        C1 = -noise_scheduler.posterior_mean_coef1[timestep] * \
            noise_scheduler.sqrt_recipm1_alphas_cumprod[timestep]
        we = C1
        # 计算确定性部分（均值）
        mean = ws * sample_images + we * model_output[:,:4,:,:]
        
        # 添加随机噪声
        if noise is None:
            noise = torch.randn_like(sample_images).to(sample_images.device)
        t=torch.tensor(timestep, device=sample_images.device)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(sample_images.shape) - 1))).to(sample_images.device)
        # nonzero_mask = (t != 0).float().view(-1, *([1] * (len(sample_images.shape) - 1)))
        
        # 使用model_log_variance计算最终结果
        sample_images = mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    else:
        temperature = 1.0
        sigma_t = ddim_eta *torch.sqrt((1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev))
        noise = sigma_t * torch.randn_like(sample_images) * temperature 
        we = (1 - alpha_prod_t_prev - sigma_t**2) ** (0.5) - ((alpha_prod_t_prev * (1 - alpha_prod_t)) / alpha_prod_t) ** (0.5)
        sample_images = ws * sample_images + we * model_output[:,:4,:,:] + noise
    
    return sample_images


def aqe_dns_loop_flux(timestep, diff_slope, diff_intercept, epsilon_delta_t_sq, prev_timestep,
        sample_images, cond_kwargs, model_forward_tea_stu, noise_scheduler, device, 
        uniform_scale, cfg, is_var=False,
        step_index = None, 
    ):
    sigma = noise_scheduler.sigmas[step_index]
    sigma_next = noise_scheduler.sigmas[step_index+1]
    diff_slope = torch.tensor(diff_slope, device='cpu', dtype = torch.float64)
    diff_intercept = torch.tensor(diff_intercept, device='cpu', dtype = torch.float64)
    # diff_intercept = torch.tensor(diff_intercept, device='cpu', dtype = torch.float64).view(1,3,1,1)
    epsilon_delta_t_sq = torch.tensor(epsilon_delta_t_sq, device='cpu', dtype = torch.float64)
    snc_var = torch.tensor(snc_var**2, device='cpu', dtype = torch.float64)
    epsilon_delta_t_sq = 1/(1+diff_slope)**2*epsilon_delta_t_sq     # diff = k*et+b 后的补偿
    epsilon_delta_t_sq = torch.mean(epsilon_delta_t_sq)             #TODO 这里如果这么估算斜率会导致这部分要取均值
    if sigma_next!=0:
        sigma_tau = sigma_next
        sigma_next_tmp = sigma_next
        sigma_next = solve_sigma_pai(sigma_tau=sigma_next_tmp.item(), sigma=sigma.item(), epsilon_delta_t_sq=epsilon_delta_t_sq.item(), snc_var = snc_var.item(), eta = 1e-20)
    # 1.2 计算去噪过程的权重
    ws = 1
    dt = sigma_next - sigma
    C1 = dt
    we = C1

    # 2.去噪    
    mode_in = 'student'
    model_output = model_forward_tea_stu(sample_images, timestep, 
                                         mode = mode_in, is_cfg = (cfg is not None), uniform_scale = uniform_scale, is_var=is_var, **cond_kwargs)
    
    model_output = (model_output-diff_intercept.to(device = model_output.device, dtype = model_output.dtype))/(1+diff_slope.to(device = model_output.device, dtype = model_output.dtype))   # diff = k*et+b
    model_output_dtype = sample_images.dtype
    sample = sample_images.to(torch.float32)
    sample_images = sample + dt * model_output
    # 3. 根据给定公式依次计算
    # 3.1
    C2=(1-sigma_next)+(sigma_next**2+(dt**2*epsilon_delta_t_sq))**0.5
    sample_images = sample_images/C2
    sample_images = sample_images.to(model_output_dtype)
    return sample_images

def tea_ddim_cfg_loop_flux(timestep, model_output_tea, prev_timestep,
        sample_images, cond_kwargs, noise_scheduler, device, 
        cfg=3, is_var = False,
        ddim_eta = 0,
        step_index = None,
    ):
    model_output = model_output_tea
    sample = sample_images
    # Upcast to avoid precision issues when computing prev_sample
    sample = sample.to(torch.float32)
    sigma = noise_scheduler.sigmas[step_index]
    sigma_next = noise_scheduler.sigmas[step_index+1]
    dt = sigma_next - sigma
    prev_sample = sample + dt * model_output
    prev_sample = prev_sample.to(model_output.dtype)
    return prev_sample

def get_model_output_cfg(model_output, cfg):
    if cfg is not None:
        guidance_scale_reshaped = cfg
        noise_cond, noise_uncond = torch.chunk(model_output, 2, dim=0)
        model_output = noise_uncond + guidance_scale_reshaped * (noise_cond - noise_uncond)
    return model_output
