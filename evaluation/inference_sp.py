import os
print("=== 环境变量检查 ===")
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "未设置"))

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
print("设置后:", os.environ.get("CUDA_VISIBLE_DEVICES"))

import torch
print("Torch可见GPU数:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
import sys
import os
import argparse
import torch
torch.cuda.set_device(0)
sys.path.insert(0, os.path.abspath("/data/zn"))

from transformers import AutoTokenizer
from eval import run_eval, reorg_answer_file

import sp.decoding as sp_decoding
import sp.modeling_llama as sp_modeling

def ssd_forward(inputs, model, tokenizer, max_new_tokens, temperature=0.0, do_sample=False):
    """
    带详细调试输出的 forward 函数
    """
    try:
        if isinstance(inputs, torch.Tensor):
            input_ids = inputs
        elif isinstance(inputs, dict):
            # 从字典中获取 input_ids
            input_ids = inputs.get('input_ids', inputs.get('model_inputs', {}).get('input_ids'))
            if input_ids is None:
                # 如果是其他格式，尝试直接使用第一个键的值
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor) and len(value.shape) == 2:
                        input_ids = value
                        break
        else:
            print(f"❌ ERROR: Unknown input type: {type(inputs)}")
            return input_ids[:, :1], 1, 1, [1]
        
        # 🔍 打印输入信息
        print("\n" + "="*80)
        print("🔍 DEBUG: Starting generation")
        print(f"Input type: {type(inputs)}")
        print(f"Input keys: {inputs.keys() if isinstance(inputs, dict) else 'Tensor'}")
        print(f"Input shape: {input_ids.shape}")
        
        # 解码前100个字符用于调试
        try:
            input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)[:100]
            print(f"Input text: {input_text}...")
        except:
            print("Cannot decode input text")
        
        print("="*80)
        
        # 调用 self_speculative_sample 生成
        res = sp_decoding.clasp_generate(
            model,
            tokenizer,
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            early_stop=True,
            max_step_draft=10, 
            #th_stop_draft=0.3,
            # th_random_draft=1.0,
            # auto_th_stop_draft=True,
        )
        
        # 提取结果
        generate_ids = res["generate_ids"]
        print(f"DEBUG: input_ids length: {input_ids.shape[1]}")
        print(f"DEBUG: generate_ids length: {generate_ids.shape[1]}")
        print(f"DEBUG: difference: {generate_ids.shape[1] - input_ids.shape[1]}")
        
        # if generate_ids.shape[1] < input_ids.shape[1]:
        #     # generate_ids 只包含新token，需要拼接
        #     print("⚠️  generate_ids appears to be truncated, concatenating with input")
        #     generate_ids = torch.cat([input_ids, generate_ids], dim=1)
            
        new_tokens = generate_ids.size(1)- input_ids.size(1)
        matchness = res.get("matchness", 0)
        num_drafted = res.get("num_drafted_tokens", 0)
        raw_accept_counts = res.get("accept_counts", [1])
        if not raw_accept_counts:
            raw_accept_counts = [1]
            
        print("\n" + "="*80)
        print("📊 DEBUG: Generation Results")
        print(f"New tokens generated: {new_tokens}")
        print(f"Matchness (acceptance rate): {matchness:.3f}")
        print(f"Total drafted tokens: {num_drafted}")
        if num_drafted > 0:
            print(f"Average accepted per draft: {matchness * num_drafted / max(num_drafted, 1):.2f}")
        
        try:
            generated_text = tokenizer.decode(generate_ids[0], skip_special_tokens=True)[:200]
            print(f"Generated text: {generated_text}...")
        except:
            print("Cannot decode generated text")
        
        print("="*80 + "\n")
        
        accept_lengths = raw_accept_counts
        return generate_ids, new_tokens, new_tokens, accept_lengths
        
    except Exception as e:
        print(f"❌ ERROR in ssd_forward: {e}")
        import traceback
        traceback.print_exc()
        if 'input_ids' in locals():
            return input_ids[:, :1], 1, 1, [1]
        else:
            return torch.zeros((1, 1), dtype=torch.long, device=model.device), 1, 1, [1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--bench-name", default="mt_bench")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--question-begin", type=int, default=0)
    parser.add_argument("--question-end", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    
    model = sp_modeling.LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        local_files_only=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, 
        local_files_only=True,
        use_fast=False
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if tokenizer.chat_template is None:
        print("⚠️  No chat template found, setting Llama-3 template...")
        tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
    
    print(f"Model loaded. Training state: {model.training}")
    model.eval()
    
    # 🔍 打印 skip layers 配置
    attn_skip, mlp_skip = model.get_skip_layers()
    print(f"\n📋 Initial skip layer configuration:")
    print(f"   Attention skip layers: {attn_skip}")
    print(f"   MLP skip layers: {mlp_skip}")
    print(f"   Total layers: {model.config.num_hidden_layers}")
    print(f"   Skipping {len(attn_skip)} attention layers, {len(mlp_skip)} MLP layers\n")
    
    do_sample = args.temperature > 0.0
    question_end = args.question_end if args.question_end else None
    
    print(f"Running evaluation on {args.bench_name}")
    print(f"Temperature: {args.temperature}, Max new tokens: {args.max_new_tokens}")
    
    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=ssd_forward,
        model_id=args.model_id,
        question_file=f"data/{args.bench_name}/question.jsonl",
        answer_file=f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl",
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        question_begin=args.question_begin,
        question_end=question_end,
        num_choices=1,
        num_gpus_per_model=1,
        num_gpus_total=1,
    )

    reorg_answer_file(f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl")
    print("Evaluation completed!")

if __name__ == "__main__":
    main()