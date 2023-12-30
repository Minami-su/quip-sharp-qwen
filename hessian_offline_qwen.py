import os
import datetime
import random
import argparse
from copy import deepcopy
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import numpy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast
from datasets import load_dataset
from model.modeling_qwen import QWenLMHeadModel
import torch.multiprocessing as mp

# import data_utils
from lib import utils

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--devset_size', default=256, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--base_model', default='meta-llama/Llama-2-70b-hf', type=str)
parser.add_argument('--save_path', default='hessians/llama2_70b', type=str)
parser.add_argument('--scratch_path', default=None, type=str)
parser.add_argument('--chunk_size', default=256, type=int)
parser.add_argument('--async_copy_speed', default=-1, type=int)
parser.add_argument('--act_save_rate', default=4, type=int)
parser.add_argument('--save_activations', action='store_true')
parser.add_argument('--sample_proc', default=4, type=int)


def move_fn(in_q, async_copy_speed):
    # async copy to avoid slow disk
    while True:
        item = in_q.get()
        if item is None:
            return
        src, tgt = item
        if async_copy_speed > 0:
            os.system(f'rsync --bwlimit={async_copy_speed} {src} {tgt}')
        else:
            os.system(f'rsync {src} {tgt}')
        os.system(f'rm {src}')
        print(f'moved {src} to {tgt}')
import math

def forward_layer(layer, rotary_pos_emb_list,registered_causal_mask, attention_mask,max_positions, bs, device, in_q, out_q):
    torch.set_grad_enabled(False)
    layer = layer.to(device)
    #position_ids = position_ids.to(device)
    #attention_mask = attention_mask.to(device)
    done_c_attn = utils.register_H_hook(layer.attn.c_attn, device)
    done_c_proj = utils.register_H_hook(layer.attn.c_proj, device)
    done_w1 = utils.register_H_hook(layer.mlp.w1, device)
    #done_w2 = utils.register_H_hook(layer.mlp.w2, device)
    done_w3 = utils.register_H_hook(layer.mlp.c_proj, device)

    while True:
        dev_emb = in_q.get()
        if dev_emb is None:
            layer = layer.cpu()
            #position_ids = position_ids.cpu()
            #attention_mask = attention_mask.cpu()
            out_q.put({'c_attn': done_c_attn(), 'c_proj': done_c_proj(), 'w1': done_w1(), 'w3': done_w3()})
            return

        assert len(dev_emb) % bs == 0
        for i in range(len(dev_emb) // bs):
            dev_emb[i * bs:(i + 1) * bs] = layer(hidden_states=dev_emb[i * bs:(i + 1) * bs].to(device),
                                                # position_ids=position_ids,
                                                 attention_mask=attention_mask,
                                                 rotary_pos_emb_list=rotary_pos_emb_list,
                                                 #registered_causal_mask=registered_causal_mask,
                                                 use_cache=False,
                                                 output_attentions=False)[0].cpu()


def accumulate(in_q, move_q, ngpus, args, transformer_layer_index):
    Hs = {}
    mus = {}
    cts = {}

    for i in range(ngpus):
        out = in_q.get()
        if i == 0:
            for key in out:
                Hs[key] = torch.zeros(out[key][0].shape, dtype=out[key][0].dtype)
                mus[key] = torch.zeros(out[key][1].shape, dtype=out[key][1].dtype)
                cts[key] = 0
        for key in out:
            Hs[key].add_(out[key][0])
            mus[key].add_(out[key][1])
            cts[key] += out[key][2]

    keys = list(Hs.keys())

    for key in Hs:
        mus[key].div_(cts[key])
        Hs[key].div_(cts[key])
        Hs[key].addmm_(-mus[key].unsqueeze(-1), mus[key].unsqueeze(0))
        save_path = f"{args.scratch_path}/{transformer_layer_index}_{key}.pt" if args.scratch_path is not None else f"{args.save_path}/{transformer_layer_index}_{key}.pt"
        torch.save(
            {
                'flatH': utils.sym_to_flat(Hs[key].to(torch.float32)),
                'mu': mus[key].to(torch.float32),
                'n': Hs[key].shape[0],
                'ct': cts[key]
            }, save_path)
        if args.scratch_path is not None:
            move_q.put((f"{args.scratch_path}/{transformer_layer_index}_{key}.pt",
                        f"{args.save_path}/{transformer_layer_index}_{key}.pt"))

    del Hs, mus, cts, out

def get_ntk_alpha(true_seq_len,seq_length):
    context_value = math.log(true_seq_len / seq_length, 2) + 1
    ntk_alpha = 2 ** math.ceil(context_value) - 1
    ntk_alpha = max(ntk_alpha, 1)
    return ntk_alpha

def main(args):
    print("loading model...")
    model = QWenLMHeadModel.from_pretrained(args.base_model,
                                                 fp32=True,
                                                 low_cpu_mem_usage=True)
    #model = QWenLMHeadModel.from_pretrained(args.base_model, low_cpu_mem_usage=True, trust_remote_code=True)

    # Set the data type to float32
    #model = model.to(dtype=torch.float32)
    print("loaded model!")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True,trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eod_id
    max_positions = model.config.max_position_embeddings
    if os.path.isfile(f"{args.save_path}/dev_activations.pt"):
        print("loading cached dataset...")
        loaded_dev_activations = torch.load(f"{args.save_path}/dev_activations.pt")
        after_layer = loaded_dev_activations['after_layer']
        dev_emb = loaded_dev_activations['dev_emb']
        print(f"loaded cached dataset from {loaded_dev_activations['timestamp']}")
    else:
        print("loading dataset...")
        dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train",cache_dir="jama")
        devset = utils.sample_devset(dataset,
                                     tokenizer,
                                     args.devset_size,
                                     args.ctx_size,
                                     nproc=args.sample_proc)
        dev_emb = model.transformer.wte(devset)
        micro_batch_size, seq_length = devset.size()
        att_mask_batch = micro_batch_size
        after_layer = -1
        print("loaded dataset!")

    print(f"dev_emb dtype: {dev_emb.dtype}")
    dev_emb.share_memory_()
    device = 0
    attention_mask = torch.tril(
        torch.ones((args.batch_size, seq_length, seq_length), device=device)
    ).view(args.batch_size, 1, seq_length, seq_length)
    orig_emb = model.transformer.wte(devset)
    quant_emb = orig_emb.clone()
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    max_positions = model.config.max_position_embeddings
    registered_causal_mask=torch.tril(
                    torch.ones((max_positions, max_positions), dtype=torch.bool)
                ).view(1, 1, max_positions, max_positions).to(device)
    hidden_states = orig_emb
    kv_seq_len = hidden_states.size()[1]
    ntk_alpha_list = []
    if attention_mask is not None and kv_seq_len > seq_length:
        true_seq_lens = attention_mask.squeeze(1).squeeze(1).eq(0).sum(dim=-1, dtype=torch.int32)
        for i in range(hidden_states.size()[0]):
            true_seq_len = true_seq_lens[i].item()
            ntk_alpha = get_ntk_alpha(true_seq_len,seq_length)
            ntk_alpha_list.append(ntk_alpha)
    else:
        ntk_alpha = get_ntk_alpha(kv_seq_len,seq_length)
        ntk_alpha_list.append(ntk_alpha)
    rotary_pos_emb_list = []
    for ntk_alpha in ntk_alpha_list:
        rotary_pos_emb = model.transformer.rotary_emb(kv_seq_len, ntk_alpha=ntk_alpha)
        rotary_pos_emb_list.append(rotary_pos_emb)
    #registered_causal_mask=model.transformer.registered_causal_mask
    if args.scratch_path is not None:
        move_q = mp.Queue()
        move_p = mp.Process(target=move_fn, args=(move_q, args.async_copy_speed))
        move_p.start()
    else:
        move_q = None

    for transformer_layer_index in range(len(model.transformer.h)):
        if (transformer_layer_index <= after_layer):
            print(
                f"skipping layer {transformer_layer_index} because it is before cached activations at layer {after_layer}"
            )
            continue

        transformer_layer = model.transformer.h[transformer_layer_index]
        linear_layers = [m for m in transformer_layer.modules() if isinstance(m, torch.nn.Linear)]
        print(f"第 {transformer_layer_index} 层中的线性层数量：{len(linear_layers)}")

        # 检查是否有四个线性层，根据你的模型架构调整期望值
        #assert len(linear_layers) == 7

        chunk_size = min(args.chunk_size, len(dev_emb))
        ngpus = min(torch.cuda.device_count(), len(dev_emb) // chunk_size)

        manager = mp.get_context('spawn').Manager()
        in_q = manager.Queue()
        out_q = manager.Queue()

        accumulate_proc = mp.Process(target=accumulate,
                                     args=(out_q, move_q, ngpus, args, transformer_layer_index))
        accumulate_proc.start()

        forward_procs = []
        for i in range(ngpus):
            p = mp.Process(target=forward_layer,
                           args=(transformer_layer, rotary_pos_emb_list,registered_causal_mask, attention_mask,max_positions, args.batch_size,
                                 i, in_q, out_q))
            p.start()
            forward_procs.append(p)

        assert len(dev_emb) % args.batch_size == 0 and chunk_size % args.batch_size == 0
        i = 0
        while i < len(dev_emb):
            next = min(i + chunk_size, len(dev_emb))
            in_q.put(dev_emb[i:next])
            i = next

        for i in range(ngpus):
            in_q.put(None)

        for p in forward_procs:
            p.join()

        accumulate_proc.join()

        transformer_layer.cpu()
        model.transformer.h[transformer_layer_index] = None
        utils.clean()

        if args.save_activations and (
                transformer_layer_index % args.act_save_rate == 0 or \
                transformer_layer_index == len(model.model.layers) - 1):
            if args.scratch_path is not None:
                if os.path.exists(f'{args.scratch_path}/dev_activations.pt'):
                    print('not saving layer since disk is too slow')
                else:
                    torch.save(
                        {
                            'dev_emb': dev_emb,
                            'after_layer': transformer_layer_index,
                            'timestamp': str(datetime.datetime.now())
                        }, f'{args.scratch_path}/dev_activations.pt')
                    move_q.put((f'{args.scratch_path}/dev_activations.pt',
                                f'{args.save_path}/dev_activations.pt'))
            else:
                torch.save(
                    {
                        'dev_emb': dev_emb,
                        'after_layer': transformer_layer_index,
                        'timestamp': str(datetime.datetime.now())
                    }, f'{args.save_path}/dev_activations.pt')

        print(f"done processing layer {transformer_layer_index}")

    if args.scratch_path is not None:
        move_q.put(None)
        move_p.join()


if __name__ == "__main__":
    mp.set_start_method('spawn')
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    main(args)
