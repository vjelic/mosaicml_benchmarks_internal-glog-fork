from src.mosaic_gpt import ComposerMosaicGPT
from src.data_c4 import build_c4_dataloader
from omegaconf import OmegaConf as om
from composer.utils import reproducibility, dist
import os
import torch

def main():
    reproducibility.seed_all(17)
    model_args = {'name': 'mosaic_gpt',
 'device': 'cpu',
 'tokenizer_name': 'gpt2',
 'd_model': 768,
 'n_heads': 12,
 'n_layers': 12,
 'mlp_ratio': 4,
 'max_seq_len': 2048,
 'vocab_size': 50257,
 'init_std': 0.02,
 'attn_pdrop': 0.0,
 'resid_pdrop': 0.0,
 'emb_pdrop': 0.0,
 'attn_impl': 'flash'}
    dataloader_args = {'name': 'c4',
 'dataset': {'remote': '/workdisk/danielking/github/benchmarks/llm/my-copy-c4',
  'local': '/workdisk/danielking/github/benchmarks/llm/my-copy-c4',
  'split': 'val',
  'shuffle': True,
  'prefetch': 1000000,
  'tokenizer_name': 'gpt2',
  'max_seq_len': 2048,
  'group_method': 'concat'},
 'drop_last': True,
 'num_workers': 8,
 'pin_memory': True,
 'prefetch_factor': 2,
 'persistent_workers': True,
 'timeout': 0}
    model_cfg = om.create(model_args)
    dataloader_cfg = om.create(dataloader_args)

    model = ComposerMosaicGPT(model_cfg)
    model.to('cuda:0')
    dataloader = build_c4_dataloader(dataloader_cfg, 4)

    first_batch = next(iter(dataloader)).to('cuda:0')
    first_item = {
        'input_ids': first_batch['input_ids'][0, :].unsqueeze(dim=0),
        'attention_mask': first_batch['attention_mask'][0, :].unsqueeze(dim=0),
        'labels': first_batch['labels'][0, :].unsqueeze(dim=0)
    }
    second_item = {
        'input_ids': first_batch['input_ids'][1, :].unsqueeze(dim=0),
        'attention_mask': first_batch['attention_mask'][1, :].unsqueeze(dim=0),
        'labels': first_batch['labels'][1, :].unsqueeze(dim=0)
    }
    third_item = {
        'input_ids': first_batch['input_ids'][2, :].unsqueeze(dim=0),
        'attention_mask': first_batch['attention_mask'][2, :].unsqueeze(dim=0),
        'labels': first_batch['labels'][2, :].unsqueeze(dim=0)
    }
    fourth_item = {
        'input_ids': first_batch['input_ids'][3, :].unsqueeze(dim=0),
        'attention_mask': first_batch['attention_mask'][3, :].unsqueeze(dim=0),
        'labels': first_batch['labels'][3, :].unsqueeze(dim=0)
    }
    first_half = {
        'input_ids': first_batch['input_ids'][0:2, :],
        'attention_mask': first_batch['attention_mask'][0:2, :],
        'labels': first_batch['labels'][0:2, :]
    }
    second_half = {
        'input_ids': first_batch['input_ids'][2:4, :],
        'attention_mask': first_batch['attention_mask'][2:4, :],
        'labels': first_batch['labels'][2:4, :]
    }

    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        full_output = model.forward(first_batch)
        first_output = model.forward(first_item)
        second_output = model.forward(second_item)
        third_output = model.forward(third_item)
        fourth_output = model.forward(fourth_item)
        first_half_output = model.forward(first_half)
        second_half_output = model.forward(second_half)

    full_vs_half_diff = (full_output[0, :] - first_half_output[0, :]).sum()
    full_vs_first_diff = (full_output[0, :] - first_output[0, :]).sum()
    print(full_vs_first_diff)
    print(full_vs_half_diff)

if __name__ == '__main__':
    main()