"""
SWIFT inference for Spec-Bench (MT-Bench style evaluation)

Usage:
python inference_swift_specbench.py \
    --model-path /path/to/model \
    --model-id llama-3-8b \
    --bench-name mt_bench \
    --max-new-tokens 512
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # 根据需要修改

import argparse
import sys
import torch

# 添加项目路径
sys.path.insert(0, os.path.abspath("/data/zn/SWIFT"))  # 修改为实际路径

from transformers import AutoTokenizer
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import numpy as np

# 导入 SWIFT 相关模块
from model.swift.utils import *
from model.swift.modeling_llama import LlamaForCausalLM
from model.swift.kv_cache import initialize_past_key_values

# 导入 Spec-Bench 评测框架
from eval import run_eval, reorg_answer_file


def swift_forward(inputs, model, tokenizer, max_new_tokens, 
                  statistics=None, optimizer=None, utility=None,
                  logits_processor=None, max_steps=512):
    """
    SWIFT forward function adapted for Spec-Bench evaluation
    
    Args:
        inputs: dict with 'input_ids' and 'attention_mask' (Spec-Bench format)
        model: SWIFT model
        tokenizer: tokenizer
        max_new_tokens: maximum new tokens to generate
        statistics: SWIFT optimization statistics
        optimizer: Bayesian optimizer
        utility: utility function for Bayesian optimization
        logits_processor: logits processor for sampling
        max_steps: maximum decoding steps
    
    Returns:
        output_ids: generated token ids (including input)
        new_token_num: number of new tokens generated
        step: number of decoding steps
        accept_length_list: list of accepted token lengths per step
    """
    try:
        # 1. 提取 input_ids (兼容不同输入格式)
        if isinstance(inputs, torch.Tensor):
            input_ids = inputs
        elif isinstance(inputs, dict):
            input_ids = inputs.get('input_ids', inputs.get('model_inputs', {}).get('input_ids'))
            if input_ids is None:
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor) and len(value.shape) == 2:
                        input_ids = value
                        break
        else:
            raise ValueError(f"Unknown input type: {type(inputs)}")
        
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        input_ids = input_ids.clone()
        
        # 2. 初始化 SWIFT 所需的 KV cache
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(model.model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data
        
        input_len = input_ids.shape[1]
        cur_length = input_len
        
        # 3. SWIFT 初始化
        reset_swift_mode(model)
        swift_logits, sample_token, top1_prob = initialize_swift(
            input_ids, model, max_new_tokens,
            past_key_values, past_key_values_data,
            current_length_data, logits_processor=logits_processor
        )
        
        # 4. 保存初始 KV cache 用于优化
        input_past_key_values_data = []
        for i in range(len(past_key_values_data)):
            input_past_key_values_data.append(past_key_values_data[i].clone())
        input_current_length_data = current_length_data.clone()
        
        # 5. SWIFT 主循环
        new_token_num = 0
        draft_token_num = 0
        total_acc_num = 0
        accept_length_list = []
        
        for idx in range(max_steps):
            draft_token_num += len(top1_prob)
            
            # 生成 SWIFT choices 和 buffers
            swift_choices = eval(f"{get_choices_list(top1_prob, logits_processor=logits_processor)}")
            swift_buffers = generate_swift_buffers(
                swift_choices, 
                device=model.model.layers[-1].self_attn.q_proj.weight.device
            )
            model.swift_buffers = swift_buffers
            model.swift_choices = swift_choices
            model.model.swift_mask = swift_buffers["swift_attn_mask"]
            
            # 生成候选 tokens
            candidates, cart_candidates_prob, tree_candidates = generate_candidates(
                swift_logits,
                swift_buffers["tree_indices"],
                swift_buffers["retrieve_indices"],
                sample_token,
                logits_processor
            )
            
            # Tree decoding
            logits, outputs = tree_decoding(
                model,
                tree_candidates,
                past_key_values,
                swift_buffers["swift_position_ids"],
                input_ids,
                swift_buffers["retrieve_indices"],
            )
            
            # 评估后验概率
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor, cart_candidates_prob, 
                swift_logits[2], swift_buffers["p_indices"], tree_candidates, 
                swift_buffers["b_indices"]
            )
            
            # 更新推理输入
            input_ids, new_token_num, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                swift_buffers["retrieve_indices"],
                logits_processor,
                new_token_num,
                past_key_values_data,
                current_length_data,
                sample_p
            )
            
            # Layer set 优化 (可选)
            if (statistics is not None and statistics["optimization"] and 
                new_token_num > (statistics["context_window"] + 1) and 
                idx % statistics["opt_interval"] == 0):
                swift_optimization(
                    model,
                    input_ids[:, input_len:],
                    input_past_key_values_data,
                    input_current_length_data,
                    new_token_num,
                    statistics,
                    optimizer=optimizer,
                    utility=utility
                )
            
            # SWIFT drafting
            swift_logits, top1_prob = swift_draft(
                model,
                input_ids=sample_token,
                new_token_num=new_token_num,
                past_key_values_data=past_key_values_data,
                current_length_data=current_length_data,
                max_new_tokens=max_new_tokens,
                logits_processor=logits_processor,
            )
            
            # 记录接受长度
            accept_length_tree = input_ids.shape[1] - cur_length
            cur_length = accept_length_tree + cur_length
            accept_length_list.append(accept_length_tree)
            total_acc_num += accept_length_tree - 1
            
            # 停止条件
            if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token_num >= max_new_tokens:
                break
        
        # 6. 返回结果 (Spec-Bench 格式)
        return input_ids, new_token_num, idx + 1, accept_length_list
        
    except Exception as e:
        print(f"❌ ERROR in swift_forward: {e}")
        import traceback
        traceback.print_exc()
        # 返回最小有效结果
        if 'input_ids' in locals():
            return input_ids[:, :1], 1, 1, [1]
        else:
            return torch.zeros((1, 1), dtype=torch.long, device=model.device), 1, 1, [1]


def main():
    parser = argparse.ArgumentParser()
    
    # 基础参数
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the model")
    parser.add_argument("--model-id", type=str, required=True,
                        help="Model identifier")
    parser.add_argument("--bench-name", type=str, default="mt_bench",
                        help="Benchmark name (mt_bench, etc.)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Maximum number of new tokens")
    parser.add_argument("--question-begin", type=int, default=0)
    parser.add_argument("--question-end", type=int, default=None)
    
    # SWIFT 参数
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.85,
                        help="Top-p sampling")
    parser.add_argument("--skip-ratio", type=float, default=0.45,
                        help="Skipped layer ratio")
    parser.add_argument("--context-window", type=int, default=32,
                        help="Context window for SWIFT")
    
    # 优化参数
    parser.add_argument("--optimization", action="store_true", default=False,
                        help="Enable layer set optimization")
    parser.add_argument("--bayes", action="store_true", default=False,
                        help="Enable Bayesian optimization")
    parser.add_argument("--opt-interval", type=int, default=1,
                        help="Optimization interval")
    parser.add_argument("--bayes-interval", type=int, default=25)
    parser.add_argument("--max-opt-iter", type=int, default=1000)
    parser.add_argument("--max-tolerance-iter", type=int, default=300)
    parser.add_argument("--max-score", type=float, default=0.95)
    parser.add_argument("--cache-hit", action="store_true", default=False,
                        help="Use cached layer configuration")
    
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float32", "float16", "bfloat16"])
    
    args = parser.parse_args()
    
    print("="*80)
    print("Starting SWIFT evaluation on Spec-Bench")
    print("="*80)
    print(f"Model: {args.model_id}")
    print(f"Benchmark: {args.bench_name}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Optimization: {args.optimization}")
    print("="*80)
    
    # 1. 加载模型
    print(f"Loading SWIFT model from {args.model_path}...")
    
    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    
    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 设置 Llama-3 chat template (如果没有)
    if tokenizer.chat_template is None:
        print("⚠️  No chat template found, setting Llama-3 template...")
        tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}"
    
    print(f"Model loaded. Training state: {model.training}")
    model.eval()
    
    # 2. 配置 logits processor
    if args.temperature > 1e-5:
        logits_processor = prepare_logits_processor(
            temperature=args.temperature, 
            top_p=args.top_p
        )
    else:
        logits_processor = None
    
    # 3. 配置 skip layers
    if args.cache_hit:
        args.optimization, args.bayes = False, False
        _attn_skip_layer_id_set, _mlp_skip_layer_id_set = get_cache_configuration(
            model_name=args.model_id,
            task_name=args.bench_name
        )
    else:
        # 默认初始化：跳过奇数层
        _attn_skip_layer_id_set = np.arange(1, model.config.num_hidden_layers - 1, 2)
        _mlp_skip_layer_id_set = np.arange(1, model.config.num_hidden_layers - 1, 2)
    
    model.set_skip_layers(_attn_skip_layer_id_set, _mlp_skip_layer_id_set)
    
    print(f"\n📋 Skip layer configuration:")
    print(f"   Attention skip layers: {_attn_skip_layer_id_set}")
    print(f"   MLP skip layers: {_mlp_skip_layer_id_set}")
    print(f"   Total layers: {model.config.num_hidden_layers}")
    print(f"   Skip ratio: {len(_mlp_skip_layer_id_set) / (model.config.num_hidden_layers - 2):.2f}\n")
    
    # 4. 配置 Bayesian Optimization (如果启用)
    pbounds = {f"x{i}": (0, 1) for i in range((model.config.num_hidden_layers - 2) * 2)}
    optimizer = BayesianOptimization(
        f=None, 
        pbounds=pbounds, 
        random_state=1, 
        verbose=1, 
        allow_duplicate_points=True
    )
    optimizer.set_gp_params(alpha=1e-2)
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
    
    statistics = {
        "origin_score": 0, 
        "opt_iter": 0, 
        "tolerance_iter": 0,
        "skip_ratio": args.skip_ratio, 
        "acceptance_rate_list": [], 
        "opt_interval": args.opt_interval,
        "bayes_interval": args.bayes_interval, 
        "max_opt_iter": args.max_opt_iter,
        "max_tolerance_iter": args.max_tolerance_iter, 
        "max_score": args.max_score,
        "context_window": args.context_window, 
        "optimization": args.optimization, 
        "bayes": args.bayes
    }
    
    # 5. 运行评测
    question_file = f"data/{args.bench_name}/question.jsonl"
    answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}_swift.jsonl"
    
    print(f"\n🚀 Running evaluation...")
    print(f"Question file: {question_file}")
    print(f"Answer file: {answer_file}\n")
    
    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=swift_forward,
        model_id=args.model_id,
        question_file=question_file,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        question_begin=args.question_begin,
        question_end=args.question_end,
        num_choices=1,
        num_gpus_per_model=1,
        num_gpus_total=1,
        # SWIFT 额外参数
        optimizer=optimizer,
        utility=utility,
        statistics=statistics,
        logits_processor=logits_processor,
    )
    
    # 6. 重组答案文件
    reorg_answer_file(answer_file)
    
    print("\n" + "="*80)
    print("✅ Evaluation completed!")
    print(f"Results saved to: {answer_file}")
    print("="*80)


if __name__ == "__main__":
    main()