"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
# adapted from fastchat: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py

import json
import os
import time
import torch
import numpy as np
import shortuuid

from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def run_eval(
        model,
        tokenizer,
        forward_func,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_tokens,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        **kwargs,
):
    questions = load_questions(question_file, question_begin, question_end)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        import ray
        ray.init()
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model,
                tokenizer,
                forward_func,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_tokens,
                num_choices,
                **kwargs,
            )
        )

    if use_ray:
        ray.get(ans_handles)



@torch.inference_mode()
def get_model_answers(
    model,
    tokenizer,
    forward_func,
    model_id,
    questions,
    answer_file,
    max_new_tokens,
    num_choices,
    **kwargs,
):
    model.eval()
    print('Check model training state:', model.training)
    
    # 确保 tokenizer 设置了 padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"DEBUG: Using Chat Template: {tokenizer.chat_template is not None}")

    # ================= Warmup =================
    print("Starting Warmup...")
    for _ in range(3):
        torch.manual_seed(0)
        # 构建一个简单的符合 Llama-3 格式的 warmup 输入
        warmup_msgs = [{"role": "user", "content": "Hello"}]
        
        # 使用 apply_chat_template 或 fallback
        try:
            if tokenizer.chat_template:
                input_ids = tokenizer.apply_chat_template(
                    warmup_msgs, 
                    add_generation_prompt=True, 
                    return_tensors="pt", 
                ).to(model.device)
            else:
                # Fallback: 手动构建 Llama-3 格式
                text = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
        except Exception as e:
            print(f"Tokenization failed: {e}")
            # 简单的 fallback
            input_ids = torch.tensor([[1, 2, 3]], device=model.device)
        
        inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids)
        }
        
        try:
            torch.cuda.synchronize()
            # 调用 forward_func
            forward_func(
                inputs,  # 传入字典格式
                model,
                tokenizer,
                max_new_tokens=20,
                **kwargs,
            )
            torch.cuda.synchronize()
        except Exception as e:
            print(f"Warmup failed: {e}")
    print('Warmup done')

    # ================= Main Loop =================
    accept_lengths_tree = []
    
    for question in tqdm(questions):
        choices = []
        for i in range(num_choices):
            cur_accept_lengths_tree = []
            torch.manual_seed(i)
            
            messages = [] 
            turns = []
            steps = []
            new_tokens = []
            wall_time = []
            
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                
                # ✅ 每个 turn 都重新构建完整的对话历史
                messages = []
                # 添加之前的所有轮次
                for k in range(j):
                    messages.append({"role": "user", "content": question["turns"][k]})
                    messages.append({"role": "assistant", "content": turns[k]})
                # 添加当前问题
                messages.append({"role": "user", "content": qs})
                
                # 构建输入
                try:
                    if tokenizer.chat_template:
                        input_ids = tokenizer.apply_chat_template(
                            messages, 
                            add_generation_prompt=True, 
                            return_tensors="pt",
                        ).to(model.device)
                    else:
                        # Fallback: 手动构建对话格式
                        test_text = "test"
                        test_ids = tokenizer.encode(test_text, add_special_tokens=True)
                        has_auto_bos = test_ids[0] == tokenizer.bos_token_id if tokenizer.bos_token_id else False
                        
                        text = "" if has_auto_bos else "<|begin_of_text|>"
                        for msg in messages:
                            text += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
                        text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
                        input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=has_auto_bos).to(model.device)
                        
                    # 🔍 添加调试输出
                    if j == 0:  # 只在第一个turn打印
                        print(f"\n{'='*80}")
                        print(f"Question ID: {question['question_id']}")
                        print(f"Question: {qs[:100]}...")
                        decoded = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                        print(f"Tokenized prompt (first 200 chars): {decoded[:200]}...")
                        print(f"{'='*80}\n")
                        
                except Exception as e:
                    print(f"Tokenization failed for question {question['question_id']}: {e}")
                    # 简单的 fallback
                    input_ids = torch.tensor([[1, 2, 3]], device=model.device)
                
                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids)
                }
                
                input_ids_len = model_inputs['input_ids'].shape[1]
                
                output_str = "ERROR"
                step = 0
                new_token = 0
                accept_length_tree = []
                total_time = 0.0

                try:
                    torch.cuda.synchronize()
                    start_time = time.time()
                    
                    # 调用推理函数
                    output_ids, new_token, step, accept_length_tree = forward_func(
                        model_inputs,
                        model,
                        tokenizer,
                        max_new_tokens,
                        **kwargs,
                    )
                    
                    torch.cuda.synchronize()
                    total_time = time.time() - start_time
                    
                    # 处理输出
                    if isinstance(output_ids, torch.Tensor):
                        generated_ids = output_ids[0][input_ids_len:] 
                        output_str = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    else:
                        print(f"Warning: output_ids is not Tensor, type: {type(output_ids)}")
                        output_str = "ERROR"

                except RuntimeError as e:
                    print(f"ERROR question ID: {question['question_id']}, Error: {e}")
                    output_str = "ERROR"
                    import traceback
                    traceback.print_exc()
                
                turns.append(output_str)
                steps.append(int(step))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                cur_accept_lengths_tree.extend(accept_length_tree)

            choices.append({
                "index": i, 
                "turns": turns, 
                "decoding_steps": steps, 
                "new_tokens": new_tokens, 
                "wall_time": wall_time,
                "accept_lengths": cur_accept_lengths_tree
            })
            
            accept_lengths_tree.extend(cur_accept_lengths_tree)

        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "category": question["category"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")

    if len(accept_lengths_tree) > 0:
        print("#Mean accepted tokens: ", np.mean(accept_lengths_tree))
    else:
        print("#Mean accepted tokens: N/A")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])
