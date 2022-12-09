from src.mosaic_gpt import ComposerMosaicGPT, MosaicGPT
from src.hf_causal_lm import ComposerHFCausalLM
from src.data_c4 import build_c4_dataloader
from omegaconf import OmegaConf as om
from composer.utils import reproducibility, dist
import os
import torch
import functools

def hf_init_fn(device):
    reproducibility.seed_all(17)
    hf_model_args = {
        'name': 'hf_causal_lm',
        'hf_config_name_or_path': 'gpt2',
    }
    config_overrides = {'attn_pdrop': 0.0, 'resid_pdrop': 0.0, 'embd_pdrop': 0.0}
    hf_model_cfg = om.create(hf_model_args)

    model = ComposerHFCausalLM(hf_model_cfg, config_overrides)
    model.to(device)
    return model

def mosaic_init_fn(attn_impl, device):
    reproducibility.seed_all(17)
    mosaic_model_args = {
        'name': 'mosaic_gpt',
        'device': 'cpu',
        'tokenizer_name': 'gpt2',
        'd_model': 768,
        'n_heads': 12,
        'n_layers': 12,
        'mlp_ratio': 4,
        'max_seq_len': 512,
        'vocab_size': 50257,
        'init_std': 0.02,
        'attn_pdrop': 0.0,
        'resid_pdrop': 0.0,
        'emb_pdrop': 0.0,
        'attn_impl': attn_impl
    }
    mosaic_model_cfg = om.create(mosaic_model_args)
    model = ComposerMosaicGPT(mosaic_model_cfg)
    model.to(device)
    return model

def create_random_input(repeats):
    batches = []
    reproducibility.seed_all(17)
    ids = torch.randint(low=0, high=100, size=(1, 512), dtype=torch.int32)
    mask = torch.ones((1, 512))
    for repeat in repeats:
        batch = {
            'input_ids': ids.repeat(repeat, 1),
            'attention_mask': mask.repeat(repeat, 1),
            'labels': ids.repeat(repeat, 1)
        }
        batches.append(batch)

    for batch in batches:
        assert all(torch.equal(row, ids[0, :]) for row in batch['input_ids'])
        assert all(torch.equal(row, mask[0, :]) for row in batch['attention_mask'])
        assert all(torch.equal(row, ids[0, :]) for row in batch['labels'])
    return batches

def get_outputs(model_init_fn, inputs, autocast_enabled, autocast_dtype, device):
    outputs = []
    for input in inputs:
        reproducibility.seed_all(17)
        model = model_init_fn(device=device)
        with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=autocast_dtype):
            with torch.inference_mode():
                for key, value in input.items():
                    input[key] = value.to(device)


                output = model.forward(input)
        outputs.append(output)
    return outputs

def print_results(outputs):
    one_vs_one_diff = (outputs[0][0, :] - outputs[1][0, :]).abs()
    one_vs_two_diff = (outputs[0][0, :] - outputs[2][0, :]).abs()
    two_vs_four_diff = (outputs[2][0, :] - outputs[3][0, :]).abs()
    four_vs_eight_diff = (outputs[3][0, :] - outputs[4][0, :]).abs()
    six_vs_eight_diff = (outputs[4][0, :] - outputs[5][0, :]).abs()
    one_vs_eight_diff = (outputs[0][0, :] - outputs[4][0, :]).abs()

    two_vs_two_within_diff = (outputs[2][0, :] - outputs[2][1, :]).abs()
    print("one vs one diff: (sum, max)", (one_vs_one_diff.sum(), one_vs_one_diff.max()))
    print("one vs two diff: (sum, max)", (one_vs_two_diff.sum(), one_vs_two_diff.max()))
    print("one vs eight diff: (sum, max)", (one_vs_eight_diff.sum(), one_vs_eight_diff.max()))
    print("two vs four diff: (sum, max)", (two_vs_four_diff.sum(), two_vs_four_diff.max()))
    print("four vs eight diff: (sum, max)", (four_vs_eight_diff.sum(), four_vs_eight_diff.max()))
    print("six vs eight diff: (sum, max)", (six_vs_eight_diff.sum(), six_vs_eight_diff.max()))
    print("two (first item) vs two (second item) diff: (sum, max)", (two_vs_two_within_diff.sum(), two_vs_two_within_diff.max()))

def main():
    reproducibility.seed_all(17)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # See https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    # and https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    repeats = [1, 1, 2, 4, 8, 6]
    inputs = create_random_input(repeats)
    
    mosaic_outputs_flash = get_outputs(functools.partial(mosaic_init_fn, attn_impl='flash'), inputs, True, torch.bfloat16, 'cuda:0')
    mosaic_outputs_torch = get_outputs(functools.partial(mosaic_init_fn, attn_impl='torch'), inputs, True, torch.bfloat16, 'cuda:0')
    mosaic_outputs_torch_fp16 = get_outputs(functools.partial(mosaic_init_fn, attn_impl='torch'), inputs, True, torch.float16, 'cuda:0')
    hf_outputs = get_outputs(hf_init_fn, inputs, True, torch.bfloat16, 'cuda:0')
    hf_outputs_no_autocast = get_outputs(hf_init_fn, inputs, False, None, 'cuda:0')
    hf_outputs_cpu = get_outputs(hf_init_fn, inputs, False, None, 'cpu')


    print("HF results (bfloat16)")
    print_results(hf_outputs)

    print()

    print("Mosaic results (flash, bfloat16)")
    print_results(mosaic_outputs_flash)

    print()

    print("Mosaic results (not flash, bfloat16)")
    print_results(mosaic_outputs_torch)
    
    print()

    print("Mosaic results (not flash, float16)")
    print_results(mosaic_outputs_torch_fp16)

    print()

    print("HF results (no autocast)")
    print_results(hf_outputs_no_autocast)

    print()

    print("HF results (cpu)")
    print_results(hf_outputs_cpu)


if __name__ == '__main__':
    main()