# First-Order Steering: Translating Weight Adaptation into Activation Steering

This repository contains code for first-order steering and HeRD-Merging. 

## Installation

Tested with Python 3.11, PyTorch 2.6, and 4 NVIDIA A100 80 GB GPUs.

```bash
python -m pip install -r requirements.txt
python -m pip install -e ".[test]"
pytest -q
```

To build the supplied container from the parent directory:

```bash
docker build -t first-order-steering -f first-order-steering/Dockerfile first-order-steering
```

The test prompts used are provided in `data/test_prompts.jsonl`

## Results 

First-order steering is substantially more accurate than existing merging methods, and HeRD-Merging produces more accurate steering vectors than task arithmetic. 

| Steering Method | Analogy Use | Bulletpoint Use | Sophisticated Language | Average |
| :--- | :---: | :---: | :---: | :---: |
| **First-Order (HeRD-Merging, Ours)** | 19.8 | **65.9** | **53.6** | **46.4** |
| **First-Order (Task Arithmetic, Ours)** | **20.2** | 63.5 | 44.2 | 42.6 |
| ODESteer | 0.2 | 37.8 | 31.5 | 23.2 |
| Linear-AcT | 1.1 | 45.8 | 44.1 | 30.3 |
| RE-Control | 0.4 | 32.6 | 5.7 | 12.9 |
| CAA | 2.1 | 42.9 | 37.3 | 27.4 |
| RepE | 2.6 | 42.9 | 26.9 | 24.1 |

HeRD-Merging retains merging performance when compared to existing merging baselines. 

| Model Merging Method | Analogy Use | Bulletpoint Use | Sophisticated Language | Average |
| :--- | :---: | :---: | :---: | :---: |
| **HeRD-Merging (Ours)** | **74.8** | **94.4** | **70.6** | **79.9** |
| DARE task arithmetic | 73.5 | 93.3 | 68.7 | 78.5 |
| Task arithmetic | 74.2 | 94.3 | 66.1 | 78.2 |
| DARE-TIES | 70.0 | 94.3 | 69.1 | 77.8 |
| KnOTS-TIES | 51.2 | 90.5 | 52.2 | 64.6 |
| TIES | 51.4 | 76.1 | 55.1 | 60.9 |

## Reproducibility

The commands below can be used to reproduce the results of the paper. The seed used is $42$.

### 1. Train independent LoRA adapters

Train one conventional PEFT LoRA independently for each behavioral axis. These adapters are the common inputs to HeRD, task arithmetic, and the steering baselines.

~~~bash
python -m steered_finetuner.train_independent_lora \
  --base_model <MODEL_ID> \
  --axis_dataset analogy=<ANALOGY_DATASET> \
  --axis_dataset bulletpointer=<BULLETPOINTER_DATASET> \
  --axis_dataset sophisticated_language=<SOPHISTICATED_LANGUAGE_DATASET> \
  --dataset_split train \
  --output_dir artifacts/independent \
  --max_steps 100 \
  --batch_size 2 \
  --gradient_accumulation_steps 16 \
  --max_seq_len 512 \
  --lora_rank 8 \
  --lora_alpha 8 \
  --lora_dropout 0.0 \
  --learning_rate 3e-4 \
  --weight_decay 0.01 \
  --warmup_steps 50 \
  --save_every 100 \
  --log_every 10 \
  --num_workers 4 \
  --tokenization_num_proc 16 \
  --seed 42
~~~

Pass `--layers_to_transform <LAYER>` to restrict an adapter set to one decoder layer. Omitting it attaches LoRA adapters at every projection.

### 2. Calibrate steering layers

~~~bash
python -m analysis.caa_layer_calibration \
  --independent_root artifacts/independent \
  --axis_dataset analogy=<ANALOGY_DATASET> \
  --axis_dataset bulletpointer=<BULLETPOINTER_DATASET> \
  --axis_dataset sophisticated_language=<SOPHISTICATED_LANGUAGE_DATASET> \
  --prompts_file data/calibration_prompts.jsonl \
  --fit_prompts 100 \
  --validation_fraction 0.2 \
  --batch_size 8 \
  --max_prompt_tokens 128 \
  --device cuda:0 \
  --seed 42 \
  --output artifacts/evaluation/caa_layer_calibration.json
~~~

### 3. Fit HeRD adapters

The first stage fixes the independent LoRAs and fits the low-rank matrices $R_{S}$.

~~~bash
python -m analysis.herd_merging \
  --independent_root artifacts/independent \
  --output_dir artifacts/herd \
  --axes analogy,bulletpointer,sophisticated_language \
  --axis_dataset analogy=<ANALOGY_DATASET> \
  --axis_dataset bulletpointer=<BULLETPOINTER_DATASET> \
  --axis_dataset sophisticated_language=<SOPHISTICATED_LANGUAGE_DATASET> \
  --dataset_split train \
  --fit_prompts_per_axis 100 \
  --validation_prompts_per_axis 20 \
  --validation_prompt_jsonl data/test_prompts.jsonl \
  --teacher_batch_size 8 \
  --max_prompt_tokens 128 \
  --prefix_contexts 4 \
  --interaction_rank 8 \
  --steps 100 \
  --learning_rate 3e-3 \
  --weight_decay 1e-4 \
  --random_directions 4 \
  --validation_every 10 \
  --device cuda:0 \
  --seed 42 \
  --log_every 5
~~~

The second stage fixes the matrices $R_{S}$ and fits the parameters $\eta_S$ with the Levenberg-Marquardt algorithm.

~~~bash
python -m analysis.herd_gain_fit \
  --independent_root artifacts/independent \
  --interaction_dir artifacts/herd \
  --output_dir artifacts/herd-gains \
  --axes analogy,bulletpointer,sophisticated_language \
  --axis_dataset analogy=<ANALOGY_DATASET> \
  --axis_dataset bulletpointer=<BULLETPOINTER_DATASET> \
  --axis_dataset sophisticated_language=<SOPHISTICATED_LANGUAGE_DATASET> \
  --dataset_split train \
  --fit_pairs_per_axis 100 \
  --validation_pairs_per_axis 20 \
  --pair_batch_size 2 \
  --teacher_batch_size 2 \
  --max_seq_len 256 \
  --max_prompt_tokens 128 \
  --prefix_contexts 2 \
  --evaluation_prompts_file data/test_prompts.jsonl \
  --heldout_evaluation_prompts 100 \
  --steps 12 \
  --finite_difference 0.02 \
  --lm_damping 0.1 \
  --trust_radius 0.5 \
  --max_gain 1.0 \
  --interior_margin 0.1 \
  --curvature_prompts 4 \
  --kl_weight 1.0 \
  --marginal_weight 1.0 \
  --ss_weight 0.1 \
  --static_weight 0.1 \
  --interaction_norm_weight 0.05 \
  --kl_improvement_retention 0.8 \
  --device cuda:0 \
  --seed 42
~~~

### 5. Generate model-merging results

~~~bash
python -m analysis.compositional_generation weight \
  --independent_root artifacts/independent \
  --interaction_dir artifacts/herd-gains \
  --axes analogy,bulletpointer,sophisticated_language \
  --prompts_file data/test_prompts.jsonl \
  --num_prompts 500 \
  --prompt_offset 0 \
  --generation_batch_size 8 \
  --direction_batch_size 8 \
  --max_prompt_tokens 128 \
  --max_new_tokens 64 \
  --include_task_arithmetic \
  --merge_methods ties,dare_task_arithmetic,dare_ties,knots_ties \
  --device cuda:0 \
  --seed 42 \
  --output artifacts/generations/weight_merging.json
~~~

### 6. Generate first-order steering and steering baselines

~~~bash
python -m analysis.compositional_generation static \
  --independent_root artifacts/independent \
  --interaction_dir artifacts/herd-gains \
  --axes analogy,bulletpointer,sophisticated_language \
  --fit_prompts_file data/static_fit_prompts.jsonl \
  --fit_prompts 100 \
  --static_layers_by_axis artifacts/evaluation/caa_layers_by_axis.json \
  --prompts_file data/test_prompts.jsonl \
  --num_prompts 500 \
  --generation_batch_size 8 \
  --direction_batch_size 8 \
  --static_fit_batch_size 8 \
  --static_fit_tokens 1 \
  --max_prompt_tokens 128 \
  --max_new_tokens 64 \
  --steer-prefill \
  --static_vectors_output artifacts/steering/static_vectors.pt \
  --device cuda:0 \
  --seed 42 \
  --output artifacts/generations/herd_static.json

python -m analysis.independent_steering_comparison \
  --independent_root artifacts/independent \
  --axis_dataset analogy=<ANALOGY_DATASET> \
  --axis_dataset bulletpointer=<BULLETPOINTER_DATASET> \
  --axis_dataset sophisticated_language=<SOPHISTICATED_LANGUAGE_DATASET> \
  --caa_calibration artifacts/evaluation/caa_layer_calibration.json \
  --prompts_file data/test_prompts.jsonl \
  --methods caa,repe,linear_act,re_control,odesteer \
  --levels 0,0.25,0.5,0.75,1 \
  --fit_prompts 100 \
  --num_prompts 500 \
  --activation_batch_size 8 \
  --generation_batch_size 8 \
  --direction_batch_size 8 \
  --method_batch_size 5 \
  --max_prompt_tokens 128 \
  --max_new_tokens 64 \
  --device cuda:0 \
  --seed 42 \
  --output artifacts/generations/steering_baselines.json
~~~

### 7. Train classifiers, score generations, and report results

~~~bash
python -m analysis.behavior_classifier train \
  --axis_dataset analogy=<ANALOGY_DATASET> \
  --axis_dataset bulletpointer=<BULLETPOINTER_DATASET> \
  --axis_dataset sophisticated_language=<SOPHISTICATED_LANGUAGE_DATASET> \
  --validation_prompt_jsonl data/test_prompts.jsonl \
  --output_dir artifacts/classifiers \
  --model_name <CLASSIFIER_MODEL_ID> \
  --dataset_split train \
  --validation_fraction 0.1 \
  --test_fraction 0.1 \
  --max_length 256 \
  --epochs 2 \
  --batch_size 16 \
  --eval_batch_size 64 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --tokenization_batch_size 2048 \
  --bootstrap_repetitions 500 \
  --device cuda:0 \
  --seed 42 \
  --log_every 50

python -m analysis.behavior_classifier score-generations \
  --classifier_root artifacts/classifiers \
  --manifest artifacts/evaluation/generation_manifest.json \
  --output artifacts/evaluation/generation_scores.json \
  --scored_generations_jsonl artifacts/evaluation/generation_scores.scored.jsonl \
  --generation_json_loading eager \
  --batch_size 512 \
  --tokenization_batch_size 2048 \
  --inference_log_every 10 \
  --bootstrap_repetitions 200 \
  --device cuda:0 \
  --seed 42

python -m analysis.behavior_classifier_paper_report \
  --scores_json artifacts/evaluation/generation_scores.json \
  --scored_generations_jsonl artifacts/evaluation/generation_scores.scored.jsonl \
  --output_dir artifacts/evaluation/paper \
  --bootstrap_repetitions 2000 \
  --seed 42
~~~