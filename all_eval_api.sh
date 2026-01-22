#!/bin/bash
source /mnt/innovator/miniconda3/etc/profile.d/conda.sh
conda activate lm_evaluation
cd /mnt/innovator/code/wangcong/Evaluation/lm-evaluation-harness
export VLLM_USE_V1_ENGINE=0
# 环境变量设置
export HF_ENDPOINT="https://hf-mirror.com"
export HF_TOKEN=$HF_TOKEN
export HF_DATASETS_CACHE="/mnt/innovator/data/wangcong/.cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ALLOW_CODE_EVAL="1"

current_time=$(date "+%Y%m%d_%H%M%S")
model_path="/mnt/innovator/code/wenzichen/LLaVA-OneVision-1.5/TEST/LLaVA-OneVision-1.5-RL/experiments/checkpoints/root/0109_SFT_Stage1_44M_Mixed_AII_Sci_Data_V2_iter_0157285_sft_780K_hf-STAGE2-RL-GSPO/0109-trial1/default/epoch1epochstep1126globalstep6499"
model_name=$(basename $model_path)
output_path="/mnt/innovator/data/wangcong/data/eval/general"
mkdir -p $output_path

log_file="${output_path}/eval_${model_name}_${current_time}.log"

# 配置任务及其对应的 shot 数
# 格式保持不变，方便后面解析
tasks_config=(
    # "mmlu:5" # logprob
    # "cmmlu:5" # logprob
    # "ceval-valid:5" # logprob
    # "bbh:3"
    "gsm8k_cot:8"
    "humaneval:0"
    # "leaderboard_gpqa_diamond:5" # logprob
    # "winogrande:5" # logprob
    # "triviaqa:5"
    "nq_open:3"
    # "arc_challenge:25" # logprob
    # "arc_easy:25" # logprob
    # "hellaswag:10"
    # "agieval:0" # logprob
    "aime24:0"
    "aime25:0"
)
# tasks_config=(

#     # "hendrycks_math:4"
# )
# --- 关键修改：解析数组并拼接成逗号分隔的字符串 ---
task_names=()
shot_nums=()

for item in "${tasks_config[@]}"; do
    task_names+=("${item%%:*}")
    shot_nums+=("${item##*:}")
done

# 使用逗号连接数组
tasks_arg=$(IFS=,; echo "${task_names[*]}")
shots_arg=$(IFS=,; echo "${shot_nums[*]}")
# ----------------------------------------------

tensor_parallel_size=8
data_parallel_size=1

# 初始化日志
echo "==========================================================" >> $log_file
echo "开始全量评测时间: $(date)" >> $log_file
echo "模型路径: $model_path" >> $log_file
echo "任务列表: $tasks_arg" >> $log_file
echo "对应 Shots: $shots_arg" >> $log_file
echo "==========================================================" >> $log_file

export OPENAI_BASE_URL="http://61.175.246.233:8002/v1/chat/completions"
export model_name="stage_2_instruct_Mixed_40M_1111_shuffled_v1_iter_0135000_hf"
export OPENAI_API_KEY="EMPTY"

start_total=$(date +%s.%N)


python3 -m lm_eval run \
    --model openai-chat-completions \
    --model_args "model=${model_name},base_url=http://61.175.246.233:8002/v1/chat/completions,num_concurrent=128,eos_string=<|im_end|>,logprobs=True,top_logprobs=5" \
    --gen_kwargs "temperature=0.7,top_p=0.8,top_k=20,presence_penalty=1.5,max_tokens=8192,max_length=32768" \
    --tasks "$tasks_arg" \
    --num_fewshot "$shots_arg" \
    --apply_chat_template \
    --confirm_run_unsafe_code \
    2>&1 | tee -a $log_file

end_total=$(date +%s.%N)
total_runtime=$(echo "$end_total - $start_total" | bc)

echo -e "\n\n==========================================================" >> $log_file
echo "全部评测结束!" >> $log_file
echo "总耗时: $total_runtime 秒" >> $log_file
echo "==========================================================" >> $log_file

# 后处理提取数据
python ./tookit/extract_log.py \
    --input_log $log_file \
    --output_excel "/mnt/innovator/data/wangcong/data/eval/general/results/general_task.xlsx"

# test
# 111 test gitdoc
# test ai summary commit
