"""Fit CAA on matched behavioral response pairs and evaluate on held-out prompts."""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import logging
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from analysis.activation import default_axis_directions, generate_static_layer_vectors
from analysis.common import chunked, encode_messages, json_dump, read_prompts
from analysis.steering_comparison import _messages, _last_token
from analysis.supervised_baselines.run import normalize
from model import _decoder_layers, _load_base_model

LOG = logging.getLogger('contrastive_caa')
AXES = ('analogy', 'bulletpointer', 'sophisticated_language')
MODELS = {
    'qwen': ('Qwen/Qwen2.5-VL-3B-Instruct', 15, '/qwen-classifiers'),
    'mistral': ('mistralai/Mistral-7B-Instruct-v0.3', 11, '/general-classifiers'),
    'gemma': ('google/gemma-4-E2B-it', 34, '/general-classifiers'),
}


def select_pairs(rows, excluded, count, seed):
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    selected, seen = [], set()
    for idx in order:
        row = rows[idx]
        if str(row.get('_status', '')).lower() == 'failed' or str(row.get('generation_failed', '')).lower() in ('true', '1', 'yes'):
            continue
        pos, neg = _messages(row.get('messages')), _messages(row.get('original_messages'))
        if not pos or not neg:
            continue
        candidates = {normalize(str(row.get('prompt', '')))}
        candidates.update(normalize(m['content']) for m in pos + neg if m['role'] in ('user', 'system'))
        if candidates & excluded:
            continue
        turn = next((i for i, (a, b) in enumerate(zip(pos, neg)) if a['role'] == b['role'] == 'assistant'
                     and a['content'] != b['content'] and pos[:i] == neg[:i]), None)
        if turn is None:
            continue
        key = json.dumps(pos[:turn], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        selected.append((idx, pos[:turn + 1], neg[:turn + 1]))
        if len(selected) == count:
            return selected
    raise ValueError(f'Only {len(selected)} valid non-overlapping distinct-context pairs; need {count}')


def capture(model, layer, encoded):
    outputs = []
    def hook(_module, _inputs, out):
        outputs.append((out[0] if isinstance(out, tuple) else out).detach())
    handle = layer.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(**encoded, use_cache=False)
    finally:
        handle.remove()
    if len(outputs) != 1:
        raise RuntimeError(f'Expected one layer output, got {len(outputs)}')
    return _last_token(outputs[0], encoded['attention_mask']).cpu()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--axis-dataset', action='append', required=True, metavar='AXIS=DATASET')
    p.add_argument('--classifier-root', type=Path)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--prompts-file', type=Path, default=Path('/data/test_prompts.jsonl'))
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    datasets = dict(item.split('=', 1) for item in args.axis_dataset)
    if set(datasets) != set(AXES) or any(not value for value in datasets.values()):
        p.error('Provide exactly one nonempty --axis-dataset AXIS=DATASET for each behavior')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    root = args.output_root / (args.model + ('_smoke' if args.smoke else ''))
    root.mkdir(parents=True, exist_ok=False)
    model_name, layer_index, classifier_root = MODELS[args.model]
    classifier_root = str(args.classifier_root) if args.classifier_root else classifier_root
    prompts = read_prompts(args.prompts_file, seed=42)
    evaluation = prompts[100:101 if args.smoke else 600]
    if len(evaluation) != (1 if args.smoke else 500):
        raise ValueError('Insufficient held-out prompts')
    excluded = {normalize(p['prompt']) for p in prompts}
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = _load_base_model(model_name, torch.bfloat16, 'cuda:0').eval()
    layers = _decoder_layers(model)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    ctx = SimpleNamespace(axes=AXES, tokenizer=tokenizer, device=torch.device('cuda:0'),
                          dtype=next(model.parameters()).dtype, model=SimpleNamespace(llama=model), layers=layers)
    vectors, metadata = {}, {}
    for axis_idx, axis in enumerate(AXES):
        pairs = select_pairs(load_dataset(datasets[axis], split='train'), excluded, 3 if args.smoke else 100, 42 + axis_idx)
        differences = []
        for batch in chunked(pairs, 4):
            conversations = [x[1] for x in batch] + [x[2] for x in batch]
            encoded = encode_messages(tokenizer, conversations, ctx.device, 512, add_generation_prompt=False)
            values = capture(model, layers[layer_index], encoded)
            differences.append(values[:len(batch)] - values[len(batch):])
        differences = torch.cat(differences)
        nfit = 2 if args.smoke else 80
        vector = differences[:nfit].mean(0)
        if not torch.isfinite(vector).all() or vector.norm() == 0:
            raise ValueError(f'Invalid CAA vector for {axis}')
        vectors[axis] = {layer_index: vector.to(device=ctx.device, dtype=ctx.dtype)}
        metadata[axis] = {'dataset': datasets[axis], 'fit_pair_indices': [x[0] for x in pairs[:nfit]],
                          'validation_pair_indices': [x[0] for x in pairs[nfit:]],
                          'vector_l2': float(vector.norm()),
                          'validation_mean_pair_difference_l2': float(differences[nfit:].mean(0).norm())}
        LOG.info('%s: CAA fit %d pairs, held out %d, norm %.4f', axis, nfit, len(pairs)-nfit, vector.norm())
    torch.save({axis: vectors[axis][layer_index].float().cpu() for axis in AXES}, root / 'vectors.pt')
    json_dump(root / 'calibration.json', {'model': model_name, 'layer': layer_index, 'axes': metadata,
              'definition': 'Base-model final-token activation difference for paired positive and negative assistant responses; mean over fit pairs.',
              'exclusion': 'All test prompt texts, NFKC casefold and whitespace normalized', 'seed': 42})
    directions = default_axis_directions(AXES)
    results = {name: [] for name in directions}
    batches = list(chunked(evaluation, 4))
    for batch_index, batch in enumerate(batches, 1):
        LOG.info('%s CAA generation batch %d/%d', args.model, batch_index, len(batches))
        generated = generate_static_layer_vectors(ctx, [x['prompt'] for x in batch], list(directions.items()),
            vectors, 128, 4 if args.smoke else 64, anchored_neutral=False)
        for name in directions:
            results[name].extend(generated[name])
    payload = {'analysis': {'axes': list(AXES), 'layer': layer_index, 'directions': {name: {
        'direction': direction, 'examples': [{'prompt_index': i, 'source_index': item['source_index'],
          'prompt': item['prompt'], 'contrastive_caa_generation': results[name][i]}
          for i, item in enumerate(evaluation)]} for name, direction in directions.items()}}}
    json_dump(root / 'generations.json', payload)
    if args.smoke:
        LOG.info('Smoke test complete')
        return
    manifest = {'sources': [{'path': str(root / 'generations.json'), 'benchmark': 'steering',
       'family': 'activation_steering', 'schema': 'directions', 'binary_corners_only': True,
       'include_methods': ['contrastive_caa_generation'],
       'method_aliases': {'contrastive_caa_generation': 'contrastive_caa'}}]}
    json_dump(root / 'manifest.json', manifest)
    del ctx, model, layers, vectors
    gc.collect()
    torch.cuda.empty_cache()
    subprocess.run([sys.executable, '-m', 'analysis.behavior_classifier', 'score-generations',
      '--classifier_root', classifier_root, '--manifest', str(root / 'manifest.json'),
      '--output', str(root / 'generation_scores.json'), '--scored_generations_jsonl', str(root / 'generation_scores.scored.jsonl'),
      '--generation_json_loading', 'eager', '--batch_size', '256', '--tokenization_batch_size', '2048',
      '--bootstrap_repetitions', '200', '--seed', '42', '--device', 'cuda:0'], check=True)
    subprocess.run([sys.executable, '-m', 'analysis.behavior_classifier_paper_report',
      '--scores_json', str(root / 'generation_scores.json'), '--scored_generations_jsonl', str(root / 'generation_scores.scored.jsonl'),
      '--output_dir', str(root / 'paper'), '--bootstrap_repetitions', '2000', '--seed', '42'], check=True)
    LOG.info('Results for %s: %s', args.model, root / 'paper/RESULTS.md')


if __name__ == '__main__':
    main()
