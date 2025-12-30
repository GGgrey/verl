#!/usr/bin/env bash
set -e

export VLLM_WORKER_MULTIPROC_METHOD=spawn

date=`date '+%Y-%m-%d-%H-%M-%S'`

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"

NUM_GPU_PER_NODE=6
NUM_NODES=1
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

echo "NUM_GPUS: ${NUM_GPUS}"
echo "NUM_NODES: ${NUM_NODES}"
echo "NUM_GPU_PER_NODE: ${NUM_GPU_PER_NODE}"

start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Start time: $start_time"

# MODEL=/workspace/verl_exp/ckpts/Qwen2.5-1.5B-GRPO-Math-1Epoch/
# MODEL=/models/Qwen/Qwen2.5-1.5B
MODEL=/models/Qwen/Qwen2.5-3B
# MODEL=/models/Qwen/Qwen2.5-7B-Instruct/
# MODEL=/workspace/verl_exp/ckpts/Qwen2.5-1.5B-RFPO-Math-1Epoch/
# MODEL=/workspace/verl_exp/ckpts/Qwen2.5-1.5B-CGPO-Math-1Epoch/
# MODEL=/workspace/verl_exp/ckpts/Qwen2.5-1.5B-DAPO-Math-1Epoch/
# MODEL=/workspace/verl_exp/ckpts/Qwen2.5-1.5B-GSPO-Math-1Epoch/
# MODEL=/workspace/verl_exp/ckpts/Qwen2.5-3B-GRPO-Math-1Epoch
# MODEL=/workspace/verl_exp/ckpts/Qwen2.5-1.5B-CGPO-Math-1Epoch-1

# MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=5120,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95,seed:1234}"
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=5120,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:4096,temperature:0.0,seed:1234}"

OUTPUT_DIR=/workspace/verl/recipe/rfpo
CUSTOM_TASKS_PATH=/workspace/verl/recipe/rfpo/evaluate.py

# lighteval vllm \
#     $MODEL_ARGS \
#     "math_500_custom" \
#     --custom-tasks "$CUSTOM_TASKS_PATH" \
#     --output-dir "$OUTPUT_DIR" \
#     --save-details

# lighteval vllm \
#     $MODEL_ARGS \
#     "gsm8k_custom" \
#     --custom-tasks "$CUSTOM_TASKS_PATH" \
#     --output-dir "$OUTPUT_DIR" \
#     --save-details

# lighteval vllm \
#     $MODEL_ARGS \
#     "aime25_custom" \
#     --custom-tasks "$CUSTOM_TASKS_PATH" \
#     --output-dir "$OUTPUT_DIR" \
#     --save-details

# lighteval vllm \
#     $MODEL_ARGS \
#     "aime24_custom" \
#     --custom-tasks "$CUSTOM_TASKS_PATH" \
#     --output-dir "$OUTPUT_DIR" \
#     --save-details

# lighteval vllm \
#     $MODEL_ARGS \
#     "aime90_custom" \
#     --custom-tasks "$CUSTOM_TASKS_PATH" \
#     --output-dir "$OUTPUT_DIR" \
#     --save-details

# lighteval vllm \
#     $MODEL_ARGS \
#     "gpqa_diamond_custom" \
#     --custom-tasks "$CUSTOM_TASKS_PATH" \
#     --output-dir "$OUTPUT_DIR" \
#     --save-details

# lighteval vllm \
#     $MODEL_ARGS \
#     "minerva_custom" \
#     --custom-tasks "$CUSTOM_TASKS_PATH" \
#     --output-dir "$OUTPUT_DIR" \
#     --save-details

# lighteval vllm \
#     $MODEL_ARGS \
#     "amc23_custom" \
#     --custom-tasks "$CUSTOM_TASKS_PATH" \
#     --output-dir "$OUTPUT_DIR" \
#     --save-details

# lighteval vllm \
#     $MODEL_ARGS \
#     "olympiadbench_custom" \
#     --custom-tasks "$CUSTOM_TASKS_PATH" \
#     --output-dir "$OUTPUT_DIR" \
#     --save-details

# lighteval vllm \
#     $MODEL_ARGS \
#     "gsm_plus_custom" \
#     --custom-tasks "$CUSTOM_TASKS_PATH" \
#     --output-dir "$OUTPUT_DIR" \
#     --save-details

# lighteval vllm \
#     $MODEL_ARGS \
#     "mmlu_pro_custom" \
#     --custom-tasks "$CUSTOM_TASKS_PATH" \
#     --output-dir "$OUTPUT_DIR" \
#     --save-details

end_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "End time: $end_time"
