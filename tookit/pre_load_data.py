import os
from datasets import load_dataset, get_dataset_config_names

# === 配置区域 ===
# 所有的缓存将存放在这个目录下
SAVE_DIR = "/mnt/innovator/data/wangcong/.cache" 
HF_TOKEN=os.environ["HF_TOKEN"]

os.environ["HF_DATASETS_CACHE"] = SAVE_DIR
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_TOKEN"] = HF_TOKEN
# export HF_ENDPOINT="https://hf-mirror.com"
# export HF_DATASETS_CACHE="/mnt/innovator/data/wangcong/.cache"
print(os.environ["HF_DATASETS_CACHE"])
# 定义任务映射 (lm-eval 任务名 -> HF 数据集 ID 和配置)
# 注意：有些任务（如 MMLU/CMMLU/CEval）包含几十个子任务，脚本会自动处理
dataset_map = {
    # "hellaswag": {"path": "Rowan/hellaswag"},
    # "winogrande": {"path": "winogrande", "name": "winogrande_xl"},
    # "arc_easy": {"path": "allenai/ai2_arc", "name": "ARC-Easy"},
    # "arc_challenge": {"path": "allenai/ai2_arc", "name": "ARC-Challenge"},
    # "ai2_arc": {"path": "ai2_arc"},
    # "gsm8k": {"path": "gsm8k", "name": "main"},
    # "triviaqa": {"path": "trivia_qa", "name": "rc.nocontext"},
    # "nq_open": {"path": "nq_open"},
    # "humaneval": {"path": "openai/openai_humaneval"},
    # "bbh": {"path": "SaylorTwift/bbh"}, # BBH 通常在多个仓库有，这是常用版本之一
    # "mmlu": {"path": "cais/mmlu"},   # 包含 57 个子任务
    # "ceval": {"path": "ceval/ceval-exam"}, # 包含 52 个子任务
    
    # "cmmlu": {"path": "lmlmcat/cmmlu"}, # 包含 67 个子任务
    # "agieval": {"path": "haonan-li/AGIEval"}, # 包含多个子任务
    # "aime24": {"path": "Maxwell-Jia/AIME_2024"}, # 包含多个子任务
    # "aime25": {"path": "math-ai/aime25"}, # 包含多个子任务
    "olympic": {"path": "GAIR/OlympicArena"}, # 包含多个子任务
}

def download():
    for task_label, info in dataset_map.items():
        path = info["path"]
        name = info.get("name")
        
        print(f"\n正在处理任务: {task_label} (HF Path: {path})...")
        
        try:
            if task_label in ["mmlu", "ceval", "cmmlu", "agieval", "bbh", "olympic"]:
                # 这些数据集有很多子配置，我们需要遍历下载
                print(f"检测到多子任务数据集，正在获取所有配置...")
                configs = get_dataset_config_names(path)
                for config in configs:
                    print(f"  -> 下载子配置: {config}")
                    load_dataset(path, config)
            else:
                # 普通单配置数据集
                load_dataset(path, name)
                print(f"完成 {task_label} 下载")
                
        except Exception as e:
            print(f"错误: 任务 {task_label} 下载失败: {e}")

if __name__ == "__main__":
    # if not os.path.exists(SAVE_DIR):
    #     os.makedirs(SAVE_DIR)
    download()
    print(f"\n所有数据集已预加载至: {os.environ['HF_DATASETS_CACHE']}")