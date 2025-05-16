# NeurIPS 2025 Dataset Track Submission #23

The repository provides the code to evaluate multiple video MLLMs on the `VideoMathQA` benchmark. We provide a task implementation compatible with [lmms\_eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) to improve the reproducibility of our experiments. Further, the repository includes self-contained scripts for CoT post-processing and CoT step evaluation.

Below, we provide a step-by-step example to evaluate [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) on `VideoMathQA`.

## Setup

Please follow the steps below.

```shell
# Clone neurips_submission_23 and lmms_eval
git clone https://github.com/annonymousa378/neurips_submission_23.git
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git

# Copy the VideoMathQA implementation to lmms_eval
cp -r neurips_submission_23/videomathqa lmms-eval/lmms_eval/tasks/

cd lmms-eval
```

## Installation

Please follow the steps to create a conda environment and install the required packages.

```shell
# Create conda environment
conda create --name neurips_submission_23 python=3.12
conda activate neurips_submission_23

# Install lmms_eval
pip install -e .
```

## Run Evaluation

Please run the following command to start evaluation.

```python
accelerate launch --num_processes=8 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=Qwen/Qwen2.5-VL-3B-Instruct,max_pixels=151200,min_pixels=100352,use_flash_attention_2=True,device_map=auto \
    --tasks videomath_mbin \
    --batch_size 1 --log_samples --log_samples_suffix qwen_2_5_vl \
    --output_path output
```

This command starts evaluating the Qwen2.5-VL-3B model on `VideoMathQA` for multi-binary accuracy. The other available `VideoMathQA` tasks are:

1. videomath\_mcq
2. videomath\_mcq\_w\_subtitles
3. videomath\_mcq\_cot
4. videomath\_mcq\_cot\_w\_subtitles
5. videomath\_mbin
6. videomath\_mbin\_w\_subtitles
7. videomath\_mbin\_cot
8. videomath\_mbin\_cot\_w\_subtitles

`w_subtitles` tasks additionally use subtitles during evaluation. `cot` tasks prompt the model to think step-by-step before answering the question.

Please note that we use 8 A100-80GB GPUs for running all evaluations. For models with parameter scale ≤ 8B, we use data parallelism with 8 workers (e.g., `num_processes=8`).
For larger models, we use tensor parallelism with TP=8 (e.g., `num_processes=1`).

## CoT Step Evaluation

We provide a [VLLM](https://github.com/vllm-project/vllm)-based script to run CoT step evaluation after inference. The self-contained script is available at [cot\_step\_evaluation.py](cot_step_evaluation.py).

```shell
# Install VLLM
pip install vllm

# Run CoT step evaluation
python cot_step_evaluation.py --gt_file <path/to/the/annotation/parquet_file/from/annonymousa378/neurips_submission_23> --res_file <path/to/the/results/file/generated/after/running/inference/using/lmms_eval>
```
