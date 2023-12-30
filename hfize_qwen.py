import argparse
import os
import glog
import torch
from transformers import AutoTokenizer
from model.version import MODEL_VERSION
from model.llama import LlamaForCausalLM as llama_fuse
from model.llama_nofuse import LlamaForCausalLM as llama_nofuse
from model.mistral import MistralForCausalLM
from lib import codebook
from lib.utils.unsafe_import import model_from_hf_path
import time

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--quantized_path', type=str)
parser.add_argument('--hf_output_path', type=str)


def unpack_quip(module, saved_layer, codebook_id, codesz):
    (m, n) = saved_layer['Qidxs'].shape
    if codebook_id in codebook.cache_permute_set:
        module.Qidxs.copy_(saved_layer['Qidxs'].view(m, n // codesz,
                                                     codesz).permute(1, 0,
                                                                     2).reshape(m, n).contiguous())
    else:
        module.Qidxs.copy_(saved_layer['Qidxs'])

    if module.rank > 0:
        module.A.copy_(saved_layer['A'])
        module.B.copy_(saved_layer['B'])
    module.SU.copy_(saved_layer['SU'])
    module.SV.copy_(saved_layer['SV'])
    module.Wscale.copy_(saved_layer['Wscale'])
    if module.rescale_WH:
        module.scaleWH.copy_(saved_layer['scaleWH'])

    module.codebook_id.copy_(codebook_id)

from model.modeling_qwenq import QWenLMHeadModel
def main(args):
    assert os.path.exists(args.quantized_path)
    saved_config = torch.load(os.path.join(args.quantized_path, 'config.pt'))
    print(saved_config)
    model_config = saved_config['model_config']

    codebook_id = codebook.get_id(model_config.quip_params['codebook'])
    codesz = model_config.quip_params['codesz']

    tokenizer = AutoTokenizer.from_pretrained(model_config._name_or_path,trust_remote_code=True)

    model_type = model_config.model_type
    fused = model_config.quip_params.get('fused', True)

    model_config.quip_params['model_version'] = MODEL_VERSION

    if model_type == 'llama':
        model_cls = llama_fuse if fused else llama_nofuse
    elif model_type == 'mistral':
        model_cls = MistralForCausalLM
    elif model_type == 'qwen':
        model_cls = QWenLMHeadModel
    else:
        raise Exception

    model = model_cls.from_pretrained(model_config._name_or_path,
                                      torch_dtype='auto',
                                      low_cpu_mem_usage=True,
                                      config=model_config).half()

    for ii in range(len(model.transformer.h)):
        glog.info(f'updating layer {ii}')

        layer = model.transformer.h[ii]
        cpu = torch.device('cpu')
        print(fused)
        if fused:
            glog.info(f'loading layer {ii} qkv')
            saved_layer = torch.load(f'{args.quantized_path}/{ii}_c_attn.pt', map_location=cpu)
            layer.attn.c_attn_scale.copy_(saved_layer['W_q_scale'])
            #print(saved_layer['W_q_bias_scale'])
            layer.attn.c_attn_bias_scale.copy_(saved_layer['W_q_bias_scale'])
            unpack_quip(layer.attn.c_attn, saved_layer, codebook_id, codesz)


            saved_layer = torch.load(f'{args.quantized_path}/{ii}_c_proj.pt', map_location=cpu)
            layer.attn.c_proj_scale.copy_(saved_layer['W_k_scale'])
            unpack_quip(layer.attn.c_proj, saved_layer, codebook_id, codesz)

            saved_layer = torch.load(f'{args.quantized_path}/{ii}_w1.pt', map_location=cpu)
            layer.mlp.w1_scale.copy_(saved_layer['W_up_scale'])
            layer.mlp.w2_scale.copy_(saved_layer['W_gate_scale'])
            unpack_quip(layer.mlp.w1w2, saved_layer, codebook_id, codesz)

            saved_layer = torch.load(f'{args.quantized_path}/{ii}_w3.pt', map_location=cpu)
            layer.mlp.w3_scale.copy_(saved_layer['W_down_scale'])
            if model_config.quip_params['outlier_channel_split']:
                layer.mlp.c_proj.ocs_dupe_inds.copy_(torch.tensor(saved_layer['ocs_dupe_inds']))
            unpack_quip(layer.mlp.c_proj, saved_layer, codebook_id, codesz)

        else:
            saved_layer = torch.load(f'{args.quantized_path}/{ii}_c_attn.pt', map_location=cpu)
            layer.attn.c_attn_scale.copy_(saved_layer['W_q_scale'])
            if model_config.quip_params['outlier_channel_split']:
                layer.attn.c_attn.ocs_dupe_inds.copy_(
                    torch.tensor(saved_layer['ocs_dupe_inds']))
            unpack_quip(layer.attn.c_attn, saved_layer, codebook_id, codesz)


            saved_layer = torch.load(f'{args.quantized_path}/{ii}_c_proj.pt', map_location=cpu)
            layer.attn.c_proj_scale.copy_(saved_layer['W_k_scale'])
            if model_config.quip_params['outlier_channel_split']:
                layer.attn.c_proj.ocs_dupe_inds.copy_(
                    torch.tensor(saved_layer['ocs_dupe_inds']))
            unpack_quip(layer.attn.c_proj, saved_layer, codebook_id, codesz)

            saved_layer = torch.load(f'{args.quantized_path}/{ii}_w1.pt', map_location=cpu)
            layer.mlp.w1_scale.copy_(saved_layer['W_up_scale'])
            if model_config.quip_params['outlier_channel_split']:
                layer.mlp.w1.ocs_dupe_inds.copy_(torch.tensor(saved_layer['ocs_dupe_inds']))
            unpack_quip(layer.mlp.w1w2, saved_layer, codebook_id, codesz)

            saved_layer = torch.load(f'{args.quantized_path}/{ii}_w2.pt', map_location=cpu)
            layer.mlp.w2_scale.copy_(saved_layer['W_gate_scale'])
            if model_config.quip_params['outlier_channel_split']:
                layer.mlp.w2.ocs_dupe_inds.copy_(torch.tensor(saved_layer['ocs_dupe_inds']))
            unpack_quip(layer.mlp.w1w2, saved_layer, codebook_id, codesz)

            saved_layer = torch.load(f'{args.quantized_path}/{ii}_w3.pt', map_location=cpu)
            layer.mlp.w3_scale.copy_(saved_layer['W_down_scale'])
            if model_config.quip_params['outlier_channel_split']:
                layer.mlp.c_proj.ocs_dupe_inds.copy_(torch.tensor(saved_layer['ocs_dupe_inds']))
            unpack_quip(layer.mlp.c_proj, saved_layer, codebook_id, codesz)

    glog.info(f'saving model...')
    model.save_pretrained(args.hf_output_path, safe_serialization=True)

    del model

    model, _ = model_from_hf_path(args.hf_output_path, use_cuda_graph=False, use_flash_attn=False)

    glog.info('successfully loaded hfized model')

    glog.info('generating some text...')

    start = time.time()
    prompt = 'It is a truth universally acknowledged that'
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(input_ids=inputs['input_ids'].cuda(),
                             attention_mask=inputs['attention_mask'].cuda(),
                             max_new_tokens=64,
                             return_dict_in_generate=True)
    token = outputs.sequences[0, :]
    output_str = tokenizer.decode(token)
    glog.info(output_str)
    glog.info(f'elapsed: {time.time() - start}')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)
