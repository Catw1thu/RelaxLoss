# 文件名: output/plot_local_results.py
# 请将此脚本放在您的 RelaxLoss/output/ 目录下

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import yaml  # 用于加载 params.yml

# --- 0. 设置项目路径，确保能导入source下的自定义模块 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SOURCE_DIR = os.path.join(PROJECT_ROOT, "source")

if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# --- 1. 定义或导入修改后的 plot_hist_normalized 函数 ---
# (此处假设 plot_hist_normalized 函数已按之前讨论的方式定义在此或已正确导入)
# from utils.misc import plot_hist_normalized # 如果在 source/utils/misc.py 中
# 新的绘图函数定义
def plot_loss_distribution_final(
    values,  # 预期是一个列表，例如 [train_losses_np, test_losses_np]
    # 其中 train_losses_np 等已经是 NumPy 数组
    names,  # 图例标签列表，例如 ["Train (Member)", "Test (Non-Member)"]
    save_file,  # 完整的保存文件路径，例如 "output/local_plots/vanilla/loss_dist_density.png"
    plot_type="density",  # Y轴类型: 'density' (概率密度) 或 'frequency' (归一化频数/占比) 或 'counts' (原始频数)
    x_lim_params=None,  # 字典，例如 {'xlim': (0, 8), 'range_for_bins': (0, 8)}
    title=None,  # 图表标题
    num_bins=50,  # 直方图的bin数量
):
    plt.figure(figsize=(8, 6))  # 设置图片大小

    # 准备用于确定统一分箱范围的数据
    # 假设 'values' 中的元素已经是扁平化的 NumPy 数组
    all_flat_values_for_range = []
    if values:
        for v_array in values:
            if (
                v_array is not None
                and isinstance(v_array, np.ndarray)
                and v_array.size > 0
            ):
                all_flat_values_for_range.append(v_array)  # 直接添加numpy数组
            # 如果意外传入了Tensor，可以尝试转换，但最好确保传入的就是numpy数组
            elif (
                v_array is not None
                and hasattr(v_array, "cpu")
                and hasattr(v_array, "detach")
                and hasattr(v_array, "numpy")
            ):
                all_flat_values_for_range.append(
                    v_array.cpu().detach().numpy().flatten()
                )
                print(
                    f"  Warning: plot_loss_distribution_final received a Tensor for range calculation, converted to NumPy."
                )

    if not all_flat_values_for_range:
        print(
            f"  Warning: No valid data arrays to plot for {title if title else save_file}. Skipping plot."
        )
        plt.close()
        return

    # 将所有数据合并以确定全局的bin范围
    combined_data_for_bins = np.hstack(all_flat_values_for_range)
    if combined_data_for_bins.size == 0:
        print(
            f"  Warning: Combined data for bins is empty for {title if title else save_file}. Skipping plot."
        )
        plt.close()
        return

    # 确定分箱的实际范围
    if x_lim_params and x_lim_params.get("range_for_bins"):
        # 如果参数指定了用于计算bins的范围
        bin_calc_range = x_lim_params["range_for_bins"]
    else:
        # 否则根据所有合并数据的实际最小最大值确定
        v_min_data = combined_data_for_bins.min()
        v_max_data = combined_data_for_bins.max()
        # 添加一点padding避免数据点正好在边界上
        padding = (
            (v_max_data - v_min_data) * 0.01 if (v_max_data - v_min_data) > 0 else 0.1
        )
        bin_calc_range = (v_min_data - padding, v_max_data + padding)
        # 确保范围有效
        if bin_calc_range[0] >= bin_calc_range[1]:
            bin_calc_range = (bin_calc_range[0] - 0.5, bin_calc_range[1] + 0.5)

    # 创建分箱边界
    bins_for_hist = np.linspace(bin_calc_range[0], bin_calc_range[1], num_bins + 1)

    # 循环绘制每个数据集的直方图
    for val_numpy_array, name in zip(
        values, names
    ):  # 再次假设values中的元素是NumPy数组
        if (
            val_numpy_array is None
            or not isinstance(val_numpy_array, np.ndarray)
            or val_numpy_array.size == 0
        ):
            print(
                f"  Warning: No data or not a NumPy array for legend entry '{name}' in plot '{title if title else save_file}'. Skipping this series."
            )
            continue

        # 确保数据是一维的
        data_to_plot = val_numpy_array.flatten()

        weights_for_hist = None
        y_label_text = "Frequency"  # 默认Y轴标签

        if plot_type == "frequency":  # 归一化频数 (占比)
            if len(data_to_plot) > 0:  # 避免除以零
                weights_for_hist = np.ones_like(data_to_plot) / len(data_to_plot)
            y_label_text = "Normalized Frequency (Proportion)"
            current_density = (
                False  # plt.hist的density参数应为False当使用weights进行占比归一化时
            )
        elif plot_type == "density":  # 概率密度
            y_label_text = "Normalized Frequency (Density)"
            current_density = True  # plt.hist的density参数为True
        else:  # 原始频数 (plot_type == 'counts' 或其他)
            y_label_text = "Frequency (Counts)"
            current_density = False

        plt.hist(
            data_to_plot,
            bins=bins_for_hist,
            alpha=0.5,
            label=name,
            density=current_density,
            weights=weights_for_hist,
        )

    plt.legend(loc="best")
    plt.ylabel(y_label_text)
    plt.xlabel("Loss")
    if title:
        plt.title(title)

    # 设置X轴的显示范围 (如果通过参数传入)
    if x_lim_params and x_lim_params.get("xlim"):
        plt.xlim(x_lim_params["xlim"])

    # 设置Y轴的显示范围 (可选)
    current_ylim = plt.ylim()
    if (
        plot_type == "frequency"
    ):  # 如果是占比，Y轴上限通常不超过1.0 (除非bin极少且数据极端集中)
        plt.ylim(0, min(1.05, max(0.1, current_ylim[1] * 1.1)))  # 稍微给顶部留一点空间
    else:  # 密度或原始频数，Y轴可以超过1
        plt.ylim(0, max(0.1, current_ylim[1] * 1.1))  # 确保从0开始，并给顶部留空间

    plt.tight_layout()

    # 确保保存图片的目录存在
    save_dir_for_plot = os.path.dirname(save_file)
    if save_dir_for_plot and not os.path.exists(
        save_dir_for_plot
    ):  # 检查 save_dir_for_plot 是否为空字符串
        os.makedirs(save_dir_for_plot, exist_ok=True)

    plt.savefig(save_file, dpi=150, format="png", bbox_inches="tight")
    print(f"Plot ({plot_type}) saved to: {save_file}")
    # plt.show() # 在脚本中运行时，通常不需要交互式显示，可以注释掉
    plt.close()  # 保存后关闭图形，好习惯


# --- 2. 导入必要的类 (确保这些是您修改过的版本) ---
from utils.base import BaseTrainer
from cifar.defense.base import CIFARTrainer

# from nonimage.defense.base import NonImageTrainer # 如果需要处理非图像
import models

# --- 3. 定义实验结果文件所在的【基础目录】 ---
# LOCAL_RESULTS_BASE_DIR 指向 .../RelaxLoss/output/
LOCAL_RESULTS_BASE_DIR = SCRIPT_DIR

# 定义要重新绘图的实验配置
# 'path_from_output_dir': 相对于 LOCAL_RESULTS_BASE_DIR (即 output/ 目录) 的路径
# 'model_arch_override': 有时 params.yml 中的 'model' 可能是通用名，这里可以覆盖以匹配 models.__dict__
experiments_to_replot_config = {
    # "Vanilla_CIFAR10_ResNet20": {
    #     "path_from_output_dir": "cifar10/resnet20/vanilla",  # <--- 替换为您的实际Vanilla实验子目录名
    #     "label_for_plot": "Vanilla (CIFAR10, ResNet20)",
    #     "plot_display_params": {
    #         "xlim": (0, 8),
    #         "range_for_bins": (0, 8),
    #     },  # 绘图X轴显示和计算bins的范围
    #     # "model_arch_override": "resnet20" # 如果params.yml中的model名不直接对应models.__dict__key
    # },
    "RelaxLoss_a1_CIFAR10_ResNet20": {
        "path_from_output_dir": "cifar10/resnet20/relaxloss_a=1_epoches=300",  # <--- 替换为您的实际RelaxLoss实验子目录名
        "label_for_plot": "RelaxLoss α=1.0 (CIFAR10, ResNet20)",
        "plot_display_params": {"xlim": (0, 4), "range_for_bins": (0, 4)},
        # "model_arch_override": "resnet20"
    },
    # 您可以添加更多实验配置
}

# --- 4. 循环处理每个实验 ---
for exp_key, exp_config in experiments_to_replot_config.items():
    exp_relative_path = exp_config["path_from_output_dir"]
    exp_plot_label = exp_config["label_for_plot"]
    exp_plot_display_params = exp_config.get("plot_display_params", None)

    exp_dir_full_path = os.path.join(LOCAL_RESULTS_BASE_DIR, exp_relative_path)
    print(f"\nProcessing experiment: {exp_plot_label} from {exp_dir_full_path}")

    # a. 加载该实验的参数 (params.yml)
    params_path = os.path.join(exp_dir_full_path, "params.yml")
    if not os.path.exists(params_path):
        print(f"  ERROR: params.yml not found in {exp_dir_full_path}")
        continue

    with open(params_path, "r") as f:
        args_dict_from_yaml = yaml.safe_load(f)

    # 将YAML加载的字典转换为Namespace对象，以便通过args.attribute访问
    args = argparse.Namespace(**args_dict_from_yaml)

    # b. **为本地重新绘图覆盖或补充关键参数** #    (因为params.yml中的某些设置可能特定于原始训练环境)
    args.no_cuda = False  # 假设本地绘图时，如果GPU可用就用GPU计算损失
    args.num_workers = 0  # 重新获取损失时，0个worker通常更简单、更稳定

    # **重要：本地数据路径**
    # 您需要确保这个路径指向您本地存放CIFAR10（或其他）数据集的位置
    # 或者一个可写目录，让torchvision在需要时可以下载数据
    LOCAL_DATA_DIR_FOR_SCRIPT = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(LOCAL_DATA_DIR_FOR_SCRIPT, exist_ok=True)
    args.data_dir = LOCAL_DATA_DIR_FOR_SCRIPT  # 覆盖 params.yml 中的 data_dir

    # 确保 'partition' 和 'if_data_augmentation' 存在，用于 DataLoader 设置
    # 这些通常应该在 params.yml 中，但提供默认值以防万一
    args.partition = getattr(args, "partition", "target")
    args.if_data_augmentation = getattr(
        args, "if_data_augmentation", False
    )  # 获取损失时通常禁用数据增强

    # 确保 'seed' 或 'random_seed' 存在，用于 BaseTrainer 的种子和生成器
    # BaseTrainer 会优先使用 args.seed
    if not hasattr(args, "seed") and hasattr(args, "random_seed"):
        args.seed = args.random_seed
    elif not hasattr(args, "seed") and not hasattr(args, "random_seed"):
        args.seed = 42  # 最终备用种子

    # 如果 params.yml 中的 'model' 名称与 models.__dict__ 中的键名不完全匹配，
    # 可以使用 exp_config 中的 'model_arch_override'
    model_arch_to_load = exp_config.get("model_arch_override", args.model)

    print(
        f"  Using effective args for {exp_plot_label}: relevant subset: dataset={args.dataset}, seed={args.seed}, model_to_load='{model_arch_to_load}'"
    )

    # c. 初始化 Trainer
    trainer_save_dir_for_plotting = (
        exp_dir_full_path  # 可以用实验目录作为save_dir，尽管这里不保存新模型
    )

    current_trainer = None
    if args.dataset in ["CIFAR10", "CIFAR100"]:
        current_trainer = CIFARTrainer(args, trainer_save_dir_for_plotting)
    # elif args.dataset in ['Texas', 'Purchase']:
    #     current_trainer = NonImageTrainer(args, trainer_save_dir_for_plotting)
    else:
        print(
            f"  ERROR: Dataset '{args.dataset}' (from params.yml) not supported by this plotting script's Trainer selection."
        )
        continue

    # d. 加载模型 (模型加载逻辑与之前回复类似，但使用 model_arch_to_load)
    model_file_to_try = "model.pt"
    model_path = os.path.join(exp_dir_full_path, model_file_to_try)

    loaded_model = None
    # --- 开始模型加载逻辑 (复制并适配自之前回复) ---
    if os.path.exists(model_path):
        try:
            loaded_model_obj = torch.load(
                model_path, map_location=current_trainer.device, weights_only=False
            )
            if isinstance(loaded_model_obj, dict):
                state_dict_key = None
                if "model_state_dict" in loaded_model_obj:
                    state_dict_key = "model_state_dict"
                elif "state_dict" in loaded_model_obj:
                    state_dict_key = "state_dict"
                if state_dict_key:
                    model_instance = models.__dict__[model_arch_to_load](
                        num_classes=current_trainer.num_classes
                    )
                    from collections import OrderedDict

                    new_state_dict = OrderedDict()
                    for k, v in loaded_model_obj[state_dict_key].items():
                        name = k[7:] if k.startswith("module.") else k
                        new_state_dict[name] = v
                    model_instance.load_state_dict(new_state_dict)
                    loaded_model = model_instance
                else:
                    if isinstance(loaded_model_obj, torch.nn.Module):
                        loaded_model = loaded_model_obj
            elif isinstance(loaded_model_obj, torch.nn.Module):
                loaded_model = loaded_model_obj
            else:
                print(
                    f"  ERROR: {model_file_to_try} is not a nn.Module or a recognized checkpoint dict in {exp_dir_full_path}."
                )
        except Exception as e_pt:
            print(
                f"  Failed to load {model_file_to_try} ({e_pt}), trying checkpoint.pkl..."
            )
            loaded_model = None

    if loaded_model is None:
        model_path_pkl = os.path.join(exp_dir_full_path, "checkpoint.pkl")
        if os.path.exists(model_path_pkl):
            try:
                checkpoint = torch.load(
                    model_path_pkl, map_location=current_trainer.device
                )
                model_instance = models.__dict__[model_arch_to_load](
                    num_classes=current_trainer.num_classes
                )
                from collections import OrderedDict

                new_state_dict = OrderedDict()
                for k, v in checkpoint["model_state_dict"].items():
                    name = k[7:] if k.startswith("module.") else k
                    new_state_dict[name] = v
                model_instance.load_state_dict(new_state_dict)
                loaded_model = model_instance
            except Exception as e_pkl:
                print(
                    f"  ERROR: Failed to load model from checkpoint.pkl in {exp_dir_full_path}: {e_pkl}"
                )
        else:
            print(
                f"  ERROR: Neither model.pt nor checkpoint.pkl found or loadable in {exp_dir_full_path}"
            )

    if loaded_model is None:
        continue  # 跳过此实验的后续处理

    if isinstance(loaded_model, torch.nn.DataParallel):
        loaded_model = loaded_model.module
    loaded_model.to(current_trainer.device)
    print(
        f"  Model for {exp_plot_label} loaded and on device: {current_trainer.device}"
    )
    loaded_model.eval()
    # --- 结束模型加载逻辑 ---

    # e. 获取损失分布
    print(f"  Getting loss distributions for {exp_plot_label}...")
    train_losses, test_losses = current_trainer.get_loss_distributions(loaded_model)
    # f. 计算并打印均值和方差
    # train_losses 和 test_losses 此时是 NumPy 数组
    if train_losses is not None and len(train_losses) > 0:
        print(
            f"  {exp_plot_label} Train (Member) Losses: Mean={np.mean(train_losses):.4f}, Var={np.var(train_losses):.4f}, Min={np.min(train_losses):.4f}, Max={np.max(train_losses):.4f}"
        )
    else:
        print(f"  {exp_plot_label} Train (Member) Losses: No data or error.")
    if test_losses is not None and len(test_losses) > 0:
        print(
            f"  {exp_plot_label} Test (Non-Member) Losses: Mean={np.mean(test_losses):.4f}, Var={np.var(test_losses):.4f}, Min={np.min(test_losses):.4f}, Max={np.max(test_losses):.4f}"
        )
    else:
        print(f"  {exp_plot_label} Test (Non-Member) Losses: No data or error.")

    # g. 使用新的绘图函数重新绘图并保存
    local_plot_subdir = os.path.join(
        LOCAL_RESULTS_BASE_DIR, "local_normalized_plots", exp_key
    )  # 使用exp_key作为子目录名
    new_plot_filename = "loss_dist_normalized.png"
    plot_save_path = os.path.join(local_plot_subdir, new_plot_filename)

    # ---- 绘制概率密度图 (Y轴总面积为1，高度可大于1) ----
    density_plot_filename = "loss_dist_density.png"
    density_plot_save_path = os.path.join(local_plot_subdir, density_plot_filename)
    if (
        train_losses is not None
        and test_losses is not None
        and train_losses.size > 0
        and test_losses.size > 0
    ):  # 检查数据有效性
        plot_loss_distribution_final(
            [train_losses, test_losses],
            ["Train (Member)", "Test (Non-Member)"],
            density_plot_save_path,
            plot_type="density",  # 指定Y轴类型
            x_lim_params=exp_plot_display_params,  # 使用配置中定义的绘图参数
            title=f"{exp_plot_label} Loss Distribution (Density)",
        )
    else:
        print(
            f"  Skipping density plot for {exp_plot_label} due to missing/empty loss data."
        )

    # ---- 绘制归一化频数/占比图 (Y轴各项和为1，高度不大于1) ----
    frequency_plot_filename = "loss_dist_frequency.png"
    frequency_plot_save_path = os.path.join(local_plot_subdir, frequency_plot_filename)
    if (
        train_losses is not None
        and test_losses is not None
        and train_losses.size > 0
        and test_losses.size > 0
    ):  # 再次检查数据有效性
        plot_loss_distribution_final(
            [train_losses, test_losses],
            ["Train (Member)", "Test (Non-Member)"],
            frequency_plot_save_path,
            plot_type="frequency",  # 指定Y轴类型
            x_lim_params=exp_plot_display_params,
            title=f"{exp_plot_label} Loss Distribution (Proportion)",
        )
    else:
        print(
            f"  Skipping frequency plot for {exp_plot_label} due to missing/empty loss data."
        )

print("\nAll specified experiments replotted.")
