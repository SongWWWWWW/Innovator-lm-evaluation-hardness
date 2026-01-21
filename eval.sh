# /bin/bash
source /mnt/innovator/miniconda3/etc/profile.d/conda.sh
conda activate lm_evaluation
cd /mnt/innovator/code/wangcong/Evaluation/lm-evaluation-harness
export HF_ENDPOINT="https://hf-mirror.com"
export HF_TOKEN=$HF_TOKEN
export HF_DATASETS_CACHE="/mnt/innovator/data/wangcong/.cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ALLOW_CODE_EVAL="1"


start=$(date +%s.%N)

# tasks="bbh,triviaqa,nq_open,winogrande,mmlu,arc_easy,hellaswag,humaneval,arc_challenge,gsm8k_cot,ceval-valid,cmmlu,agieval,leaderboard_gpqa_diamond"
tasks="bbh,triviaqa,nq_open,winogrande,mmlu,arc_easy,hellaswag,humaneval,arc_challenge,gsm8k_cot,ceval-valid,cmmlu,leaderboard_gpqa_diamond"
# cmmlu agieval
extra_tasks="cmath,cluewsc2020"
model_path="/mnt/innovator/data/wangcong/model/Qwen3-30B-A3B-Base"
tensor_parallel_size=8
data_parallel_size=1
output_path="/mnt/innovator/data/wangcong/data/eval"
gen_kwargs="temperature=0.7,top_p=0.8,top_k=20,presence_penalty=1.5"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 lm_eval --model vllm \
    --model_args pretrained=$model_path,tensor_parallel_size=$tensor_parallel_size,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=$data_parallel_size\
    --gen_kwargs temperature=0.7,top_p=0.8,top_k=20,presence_penalty=1.5 \
    --tasks hellaswag,mmlu\
    --num_fewshot 0,1\
    --batch_size auto \
    --output_path $output_path \
    --confirm_run_unsafe_code \

end=$(date +%s.%N)
# 使用 bc 计算差值
runtime=$(echo "$end - $start" | bc)
echo "Total evaluation time: $runtime seconds"

# test ai commit
# test auto commit aaa
# 
