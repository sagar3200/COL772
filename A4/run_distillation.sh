#!/bin/bash


TEACHER_MODEL="Qwen/Qwen2.5-7B-Instruct"
LLAMA_STUDENT="meta-llama/Llama-3.2-1B-Instruct"
QWEN_STUDENT="Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_DATA="data/train.jsonl"
VAL_DATA="data/val.jsonl"
TEST_DATA="data/test.jsonl"
SAMPLE_DATA="data/sampled.jsonl" 

LLAMA_OUTPUT="outputs/llama_distilled"
QWEN_OUTPUT="outputs/qwen_distilled"


echo "Starting teacher data generation..."
CUDA_VISIBLE_DEVICES=0,1 python dataset_generation.py \
    --teacher_model $TEACHER_MODEL \
    --num_samples 2250,2250,2250,2250,2000 \
    --output_file $TRAIN_DATA \
    --val_output_file $VAL_DATA \
    --test_output_file $TEST_DATA \
    --val_per_language 200 \
    --gpu_memory_utilization 0.6 \
    --tensor_parallel_size 2


echo "Starting Llama distillation training..."
CUDA_VISIBLE_DEVICES=0,1 python train_distill.py \
    --student_model $LLAMA_STUDENT \
    --teacher_model $TEACHER_MODEL \
    --train_data $TRAIN_DATA \
    --output_dir $LLAMA_OUTPUT


echo "Starting Qwen distillation training..."
CUDA_VISIBLE_DEVICES=0,1 python train_distill.py \
    --student_model $QWEN_STUDENT \
    --teacher_model $TEACHER_MODEL \
    --train_data $TRAIN_DATA \
    --output_dir $QWEN_OUTPUT


echo "Running Llama inference..."
CUDA_VISIBLE_DEVICES=0 python inference_eval.py \
    --base_model $LLAMA_STUDENT \
    --adapter_path $LLAMA_OUTPUT/best_checkpoint \
    --test_data $SAMPLE_DATA \
    --output_predictions outputs/predictions_llama.jsonl \
    --report_file outputs/metrics_llama.txt \
    --gpu_memory_utilization 0.4 \
    --tensor_parallel_size 1


echo "Running Qwen inference..."
CUDA_VISIBLE_DEVICES=0 python inference_eval.py \
    --base_model $QWEN_STUDENT \
    --adapter_path $QWEN_OUTPUT/best_checkpoint \
    --test_data $SAMPLE_DATA \
    --output_predictions outputs/predictions_qwen.jsonl \
    --report_file outputs/metrics_qwen.txt \
    --gpu_memory_utilization 0.4 \
    --tensor_parallel_size 1

echo "Pipeline complete!"