import argparse
import math
import datetime
import time
import os
import gc
from tqdm import tqdm
import copy

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import torch
import torch.multiprocessing as mp
from torch import nn, optim
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from lib import codebook, utils
from lib.algo import quip, preprocess, outlier_channel_split as ocs
from model.modeling_qwen import QWenLMHeadModel
import glog

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_cpu_threads', default=8, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--devset_size', default=64, type=int)
parser.add_argument('--ctx_size', default=2048, type=int)
parser.add_argument('--save_path', default='checkpoints/quantized-hada-70b', type=str)
parser.add_argument('--hessian_path', default='/share/desa/nfs01/quip_llama2/hessians', type=str)
parser.add_argument('--base_model', default='meta-llama/Llama-2-70b-hf', type=str)
parser.add_argument('--sigma_reg', default=1e-2, type=float)
parser.add_argument('--sigma_reg2', default=1e-2, type=float)
parser.add_argument('--incoh_mode', default='had', type=str, choices=['had', 'kron'])
parser.add_argument('--lora_rank', default=0, type=int, help='if <=0 then turned off')
parser.add_argument('--scale_override', default=-1, type=float)
parser.add_argument('--codebook', default='D4', type=str)
parser.add_argument('--quip_tune_iters', default=10, type=int)
parser.add_argument('--remove_mean', action='store_true')
parser.add_argument('--outlier_channel_split', action='store_true')
parser.add_argument('--ocs_down_size', default=2**15, type=int)
parser.add_argument('--use_fp64', action='store_true')
parser.add_argument('--full_svd', action='store_true')
parser.add_argument('--no_use_buffered', action='store_true')
parser.add_argument('--q_buffer_size', default=2, type=int)
parser.add_argument('--rescale_WH', action='store_true')
parser.add_argument('--sample_proc', default=1, type=int)


def quantize_c_attn(layer, idx, cb, args, device='cpu', check_only=False):
    dtype_ = torch.float64 if args.use_fp64 else torch.float32
    hatw_path = f'{args.save_path}/{idx}_c_attn.pt'
    W_q = layer.attn.c_attn.weight
    W_q_bias_scale=layer.attn.c_attn.bias
    W_q_scale = W_q.to(dtype_).square().mean().sqrt().to(dtype_)
    if os.path.exists(hatw_path):
        if check_only:
            return
        hatW = utils.load_quip(hatw_path, cb, args, device)
        glog.info(f'loaded saved hatW from {hatw_path}')
    else:
        H_data = torch.load(f'{args.hessian_path}/{idx}_c_attn.pt', map_location=torch.device('cpu'))
        H = utils.flat_to_sym(H_data['flatH'], H_data['n'])
        mu = H_data['mu']
        n = H_data['n']
        W_qr = (W_q.to(dtype_) / W_q_scale).to(dtype_)
       # W_q = torch.vstack(
       #     (W_q.to(dtype_) / W_q_scale, W_q.to(dtype_) / W_q_scale)).to(dtype_)
        H, mu = preprocess.basic_preprocess(H, mu, n, args)

        hatW, attr = quip.quantize(H, W_qr, args.lora_rank, cb, args, device)
        # W_q_bias_scale= nn.Parameter(
        #     (W_q.to(dtype_) @ mu - hatW* W_q_scale.to(dtype_) @ mu).half())
        attr.update({
            'W_q_scale': W_q_scale,
            'W_q_bias_scale': W_q_bias_scale,
        })
        torch.save(attr, hatw_path)
        utils.show_metrics(hatW, W_qr, H.to(dtype_), f'layer {idx} q')
        utils.clean()
    W_q_next = (hatW* W_q_scale).half()

    if args.remove_mean:
        layer.attn.c_attn.bias = nn.Parameter(
            (W_q.to(dtype_) @ mu - W_q_next.to(dtype_) @ mu).half())

    W_q.copy_(W_q_next)
    
def quantize_c_proj(layer, idx, cb, args, device='cpu', check_only=False):
    dtype_ = torch.float64 if args.use_fp64 else torch.float32
    hatw_path = f'{args.save_path}/{idx}_c_proj.pt'
    W_k = layer.attn.c_proj.weight
    W_k_scale = W_k.to(dtype_).square().mean().sqrt().to(dtype_)

    if os.path.exists(hatw_path):
        if check_only:
            return
        hatW = utils.load_quip(hatw_path, cb, args, device)
        glog.info(f'loaded saved hatW from {hatw_path}')
    else:
        H_data = torch.load(f'{args.hessian_path}/{idx}_c_proj.pt', map_location=torch.device('cpu'))
        H = utils.flat_to_sym(H_data['flatH'], H_data['n'])
        mu = H_data['mu']
        n = H_data['n']
        W_kr = W_k.to(dtype_) / W_k_scale
        # W_k = torch.vstack(
        #     (W_k.to(dtype_) / W_k_scale, W_k.to(dtype_) / W_k_scale)).to(dtype_)
        H, mu = preprocess.basic_preprocess(H, mu, n, args)

        hatW, attr = quip.quantize(H, W_kr, args.lora_rank, cb, args, device)
       # W_k_bias_scale= nn.Parameter(
         #   (W_k.to(dtype_) @ mu - hatW* W_k_scale.to(dtype_) @ mu).half())
        attr.update({
            'W_k_scale': W_k_scale,
            #'W_k_bias_scale': W_k_bias_scale,
        })
        torch.save(attr, hatw_path)
        utils.show_metrics(hatW, W_kr, H.to(dtype_), f'layer {idx} k')
        utils.clean()
    W_k_next = (hatW* W_k_scale).half()

    if args.remove_mean:
        layer.attn.c_proj.bias = nn.Parameter(
            (W_k.to(dtype_) @ mu - W_k_next.to(dtype_) @ mu).half())

    W_k.copy_(W_k_next)


def quantize_up(layer, idx, cb, args, device='cpu', check_only=False):
    dtype_ = torch.float64 if args.use_fp64 else torch.float32
    hatw_path = f'{args.save_path}/{idx}_w1.pt'
    W_up = layer.mlp.w1.weight
    W_gate = layer.mlp.w2.weight
    W_up_scale = W_up.to(dtype_).square().mean().sqrt().to(dtype_)
    W_gate_scale = W_gate.to(dtype_).square().mean().sqrt().to(dtype_)
    if os.path.exists(hatw_path):
        if check_only:
            return
        glog.info(f'loading saved hatW from {hatw_path}')
        hatW = utils.load_quip(hatw_path, cb, args, device)
    else:
        H_data = torch.load(f'{args.hessian_path}/{idx}_w1.pt', map_location=torch.device('cpu'))
        H = utils.flat_to_sym(H_data['flatH'], H_data['n'])
        mu = H_data['mu']
        n = H_data['n']
        W_upgate = torch.vstack(
            (W_up.to(dtype_) / W_up_scale, W_gate.to(dtype_) / W_gate_scale)).to(dtype_)
       # W_up = torch.vstack(
       #     (W_up.to(dtype_) / W_up_scale, W_up.to(dtype_) / W_up_scale)).to(dtype_)
        H, mu = preprocess.basic_preprocess(H, mu, n, args)

        hatW, attr = quip.quantize(H, W_upgate, args.lora_rank, cb, args, device)
        attr.update({
            'W_up_scale': W_up_scale,
            'W_gate_scale': W_gate_scale,
        })
        torch.save(attr, hatw_path)
        utils.show_metrics(hatW, W_upgate, H.to(dtype_), f'layer {idx} up')
        utils.clean()
    W_up_next = (hatW[0:(W_up.shape[0]), :] * W_up_scale).half()
    W_gate_next = (hatW[(W_up.shape[0]):(W_up.shape[0] + W_gate.shape[0]), :] * W_gate_scale).half()
    #W_up_next = (hatW * W_up_scale).half()
    if args.remove_mean:
        layer.mlp.w1.bias = nn.Parameter(
            (W_up.to(dtype_) @ mu - W_up_next.to(dtype_) @ mu).half())
        layer.mlp.w2.bias = nn.Parameter(
            (W_gate.to(dtype_) @ mu - W_gate_next.to(dtype_) @ mu).half())

    W_up.copy_(W_up_next)
    W_gate.copy_(W_gate_next)
    

def quantize_down(layer, idx, cb, args, device='cpu', check_only=False):
    dtype_ = torch.float64 if args.use_fp64 else torch.float32
    hatw_path = f'{args.save_path}/{idx}_w3.pt'

    W_down = layer.mlp.c_proj.weight
    W_down_scale = W_down.to(dtype_).square().mean().sqrt().to(dtype_)

    if os.path.exists(hatw_path):
        if check_only:
            return
        glog.info(f'loading saved hatW from {hatw_path}')
        hatW = utils.load_quip(hatw_path, cb, args, device)
        if args.outlier_channel_split:
            extra_inds = torch.load(hatw_path)['ocs_extra_inds']
    else:
        H_data = torch.load(f'{args.hessian_path}/{idx}_w3.pt', map_location=torch.device('cpu'))
        #print(H_data)
        H = utils.flat_to_sym(H_data['flatH'], H_data['n'])
        mu = H_data['mu']
        n = H_data['n']
        if args.outlier_channel_split:
            # outlier channel split to next power of two
            glog.info(f'outlier channel splitting to {args.ocs_down_size}')
            W_down, H, mu, extra_inds, dupe_inds = ocs.outlier_channel_split(
                W_down, H, mu, args.ocs_down_size)
            n = args.ocs_down_size
            utils.clean()
        W_orig = W_down.to(dtype_) / W_down_scale
        #print(W_orig.size())
        H, mu = preprocess.basic_preprocess(H, mu, n, args)
        hatW, attr = quip.quantize(H, W_orig, args.lora_rank, cb, args, device)
        attr.update({'W_down_scale': W_down_scale})
        if args.outlier_channel_split:
            attr['ocs_extra_inds'] = extra_inds
            attr['ocs_dupe_inds'] = dupe_inds
        torch.save(attr, hatw_path)
        utils.show_metrics(hatW, W_orig, H.to(dtype_), f'layer {idx} w3')
        utils.clean()
    W_down_next = (hatW * W_down_scale).half()

    if args.remove_mean:
        layer.mlp.c_proj.bias = nn.Parameter(
            (W_down.to(dtype_) @ mu - W_down_next.to(dtype_) @ mu).half())

    if args.outlier_channel_split:
        # fuse back outlier channel split
        W_down_next = ocs.fuse_W(W_down_next, extra_inds)

    W_down.copy_(W_down_next)


def quantize_layer(layer, idx, cb, args, device='cpu', return_layer=False):

    # check_only=not return_layer -> If we are not returning the layer just check
    # if it has been quantized already. Otherwise, load it for returning.
    torch.manual_seed(idx)
    torch.set_grad_enabled(False)
    quantize_c_attn(layer, idx, cb, args, device, check_only=not return_layer)
    utils.clean()
    quantize_c_proj(layer, idx, cb, args, device, check_only=not return_layer)
    utils.clean()
    quantize_up(layer, idx, cb, args, device, check_only=not return_layer)
    utils.clean()
    quantize_down(layer, idx, cb, args, device, check_only=not return_layer)
    utils.clean()

    glog.info(f'finished layer {idx}')
    if return_layer:
        return layer


def quantize_layer_queue(in_q, cb, args, device):
    while True:
        next_item = in_q.get()
        if next_item is None:
            return
        quantize_layer(*next_item, cb, args, device, False)
def get_ntk_alpha(true_seq_len,seq_length):
        context_value = math.log(true_seq_len / seq_length, 2) + 1
        ntk_alpha = 2 ** math.ceil(context_value) - 1
        ntk_alpha = max(ntk_alpha, 1)
        return ntk_alpha

def main(args):
    dtype_ = torch.float64 if args.use_fp64 else torch.float32

    cb = codebook.get_codebook(args.codebook)

    # model = QWenLMHeadModel.from_pretrained(args.base_model,
    #                                              torch_dtype='auto',
    #                                              low_cpu_mem_usage=True)
    model = QWenLMHeadModel.from_pretrained(args.base_model,fp32=True,
                                                 low_cpu_mem_usage=True)
    # Set the data type to float32
    #model = model.to(dtype=torch.float32)
    # save configs
    all_config = {'quant_args': args, 'model_config': model.config}
    all_config['model_config'].update({
        'quip_params': {
            'outlier_channel_split': args.outlier_channel_split,
            'lora_rank': args.lora_rank,
            'rescale_WH': args.rescale_WH,
            'codebook': args.codebook,
            'codebook_version': cb.version,
            'codesz': cb.codesz,
            'idx_dtype': str(cb.idx_dtype),
            'fused': True,
            'packsz': cb.packsz,
        }
    })
    if args.outlier_channel_split:
        all_config['model_config'].quip_params['ocs_down_size'] = args.ocs_down_size
    torch.save(all_config, os.path.join(args.save_path, 'config.pt'))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model,trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eod_id
    glog.info('loaded model')

    dataset = load_dataset('togethercomputer/RedPajama-Data-1T-Sample', split='train',cache_dir="jama")
    devset = utils.sample_devset(dataset, tokenizer, args.devset_size, args.ctx_size,
                                 args.sample_proc)
    glog.info('loaded dataset and devset')
    #print(devset)
    micro_batch_size, seq_length = devset.size()
    att_mask_batch = micro_batch_size
    # Reduce cpu memory consumption at the expense of latency. Tune as needed
    nproc = torch.cuda.device_count()
    #nproc=1

    if nproc > 1:
        # If we only have one process run the serial version
        # and calculate activation errors too
        layer_q = mp.Queue(maxsize=args.q_buffer_size)

        quantize_procs = []
        for i in range(nproc):
            p = mp.Process(target=quantize_layer_queue, args=(layer_q, cb, args, i))
            p.start()
            quantize_procs.append(p)
        for _ in range(len(model.transformer.h)):
            layer_q.put((copy.deepcopy(model.transformer.h[_]), _))
        for p in quantize_procs:
            layer_q.put(None)
        for p in quantize_procs:
            p.join()

        glog.info('done quantizing')

    # do the rest of the stuff on gpu 0
    device = 0
    # load quantized layers from disk and calculate activation errors
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
            ntk_alpha = get_ntk_alpha(true_seq_len)
            ntk_alpha_list.append(ntk_alpha)
    else:
        ntk_alpha = get_ntk_alpha(kv_seq_len,seq_length)
        ntk_alpha_list.append(ntk_alpha)
    rotary_pos_emb_list = []
    for ntk_alpha in ntk_alpha_list:
        rotary_pos_emb = model.transformer.rotary_emb(kv_seq_len, ntk_alpha=ntk_alpha)
        rotary_pos_emb_list.append(rotary_pos_emb)
    for i in range(len(model.transformer.h)):
        model.transformer.h[i] = model.transformer.h[i].to(device)

        for j in range(args.devset_size // args.batch_size):
            orig_emb[args.batch_size * j : args.batch_size * (j + 1)] = \
                model.transformer.h[i](
                    orig_emb[args.batch_size * j : args.batch_size * (j + 1)].to(device),
                    #position_ids=position_ids,
                    rotary_pos_emb_list=rotary_pos_emb_list,
                    attention_mask=attention_mask,
                    #registered_causal_mask=registered_causal_mask,
                    use_cache=False,
                    output_attentions=False)[0].cpu()

        model.transformer.h[i] = model.transformer.h[i].cpu()

        model.transformer.h[i] = quantize_layer(model.transformer.h[i],
                                               i,
                                               cb,
                                               args,
                                               device=device,
                                               return_layer=True).to(device)

        for j in range(args.devset_size // args.batch_size):
            quant_emb[args.batch_size * j : args.batch_size * (j + 1)] = \
                model.transformer.h[i](
                    quant_emb[args.batch_size * j : args.batch_size * (j + 1)].to(device),
                    #position_ids=position_ids,
                    rotary_pos_emb_list=rotary_pos_emb_list,
                    attention_mask=attention_mask,
                    #registered_causal_mask=registered_causal_mask,
                    use_cache=False,
                    output_attentions=False)[0].cpu()

        model.transformer.h[i] = model.transformer.h[i].cpu()
        model.transformer.h[i] = None

        act_error = (quant_emb.to(dtype_) - orig_emb.to(dtype_)).square().sum() / \
            (orig_emb.to(dtype_) - orig_emb.to(dtype_).mean((0, 1))).square().sum()

        glog.info(f'layer {i} activation error {act_error}')

    glog.info('calculating perplexity on devset')

    lm_head = model.lm_head.to(dtype_)
    lm_head.to(device)

    norm = model.transformer.ln_f.to(dtype_)
    norm.to(device)

    acc = 0.0
    for i in tqdm(range(args.devset_size // args.batch_size), desc='original model perplexity'):
        shift_logits = lm_head(
            norm(orig_emb[args.batch_size * i:args.batch_size *
                          (i + 1)].to(device).to(dtype_)))[..., :-1, :].contiguous().view(
                              -1, model.config.vocab_size)
        shift_labels = devset[args.batch_size * i:args.batch_size * (i + 1),
                              1:].contiguous().view(-1).to(device)
        loss_fct = nn.CrossEntropyLoss().to(device)
        acc += loss_fct(shift_logits, shift_labels)
    perplexity = (acc / (args.devset_size // args.batch_size + 1)).exp()
    glog.info(f'original model perplexity: {perplexity}')

    acc = 0.0
    for i in tqdm(range(args.devset_size // args.batch_size), desc='quantized model perplexity'):
        shift_logits = lm_head(
            norm(quant_emb[args.batch_size * i:args.batch_size *
                           (i + 1)].to(device).to(dtype_)))[..., :-1, :].contiguous().view(
                               -1, model.config.vocab_size)
        shift_labels = devset[args.batch_size * i:args.batch_size * (i + 1),
                              1:].contiguous().view(-1).to(device)
        loss_fct = nn.CrossEntropyLoss().to(device)
        acc += loss_fct(shift_logits, shift_labels)
    perplexity = (acc / (args.devset_size // args.batch_size + 1)).exp()
    glog.info(f'quantized model perplexity: {perplexity}')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    mp.set_start_method('spawn')
    args = parser.parse_args()
    torch.set_num_threads(args.num_cpu_threads)
    torch.manual_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    main(args)
