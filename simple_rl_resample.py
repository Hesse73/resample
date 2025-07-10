import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0" # make the scheduling deterministic

import time
import json
import torch
import datasets
import argparse
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from datasets import load_dataset
from collections import defaultdict
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer
from typing import List, Tuple
from copy import deepcopy

from math_utils import get_acc_list


def format_vllm_logp(logprobs, tokenizer):
    if type(logprobs) is list:
        return [{id_:(lp.logprob, tokenizer.decode([id_])) for id_,lp in lp_per_token.items()} for lp_per_token in logprobs]
    elif type(logprobs) is dict:
        return {id_:(lp.logprob, tokenizer.decode([id_])) for id_,lp in logprobs.items()}
    else:
        raise ValueError(f"Unsupported logprobs type: {type(logprobs)} for logprobs:\n{logprobs}\nExpected list or dict.")
    
def get_another_top_token(logprobs, skip_id):
    sorted_tokens = sorted(logprobs.items(), key=lambda x: x[1].logprob, reverse=True)
    for token, lp in sorted_tokens:
        if token != skip_id:
            return token
    return None

def generate_with_formatted_return(llm: LLM, prompts: List[str|List[int]], sampling_params: SamplingParams,
                                   return_format="processed"):
    if isinstance(prompts[0], list):
        prompts = [TokensPrompt(prompt_token_ids=p) for p in prompts]
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    if return_format == "processed":
        all_texts = []
        all_entropies = []
        all_logprobs = []
        all_generated_tokens = []
        for output in tqdm(outputs, desc="processing outputs"):
            for out in output.outputs:
                entropies = []
                for lp_per_token in out.logprobs:
                    with torch.no_grad():
                        logprobs = torch.tensor([lp.logprob for lp in lp_per_token.values()])
                        logprobs_norm = logprobs - torch.logsumexp(logprobs, dim=-1)  # Normalize logprobs
                        entropy = -torch.sum(logprobs_norm * torch.exp(logprobs_norm))  # Calculate entropy
                    entropies.append(entropy.item())
                all_texts.append(out.text)
                all_entropies.append(entropies)
                all_logprobs.append(out.logprobs)
                all_generated_tokens.append(out.token_ids)
        return all_entropies, all_logprobs, all_generated_tokens, all_texts
    else:
        return outputs

def iterative_rl_resample(args, base_model: LLM, rl_model: LLM, tokenizer: AutoTokenizer, prompts: List[str], gts: List[str],
                          continue_info=None):
    # flatten the prompts and repeat each n times
    initial_prompts = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    MAX_MODEL_LEN = base_model.llm_engine.model_config.max_model_len
    
    if continue_info is not None and args.continue_from > 0:
        print(f"--- Resuming from step {args.continue_from} ---")
        start_step, all_info = args.continue_from + 1, continue_info
        # load the previous step's info
        prev_info = continue_info[f'step_{args.continue_from}']
        # extract the final prefixs as the current prompts, and restore the entropy thresholds
        current_prompts = sum([p_info['final_prefixs'] for p_info in prev_info] ,[])
        entropy_thresholds = sum([p_info['entropy_thresholds'] for p_info in prev_info], [])
    else:
        start_step, all_info = 0, {}
        current_prompts = sum([[p]*args.n for p in initial_prompts], [])

    base_sampling_params = SamplingParams(
        temperature=args.temperature_base,
        top_p=args.top_p_base,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        logprobs=args.num_logprobs,
        skip_special_tokens=False
    )
    init_base_sampling_params = deepcopy(base_sampling_params)
    init_base_sampling_params.n = args.n  # initially, sample n times for each prompt
    rl_sampling_params = SamplingParams(
        temperature=args.temperature_rl,
        top_p=args.top_p_rl,
        max_tokens=args.rl_tokens,
        seed=args.seed,
        logprobs=args.num_logprobs,
        skip_special_tokens=True  # skip special tokens for RL model
    )

    for step in range(start_step, args.max_rl_resample + 1):
        entropy_thresholds = None if step == 0 or args.dyna_thresh else entropy_thresholds
        print(f"\n--- Iteration {step} ---\n")
        # Generate base model outputs
        if step == 0:
            base_outputs = generate_with_formatted_return(base_model, initial_prompts, init_base_sampling_params)
        else:
            base_outputs = generate_with_formatted_return(base_model, current_prompts, base_sampling_params)
        base_entropies, base_logprobs, base_generated_tokens, base_responses = base_outputs

        if entropy_thresholds is None:
            percentile = 100 * (1 - args.top_ent)
            # set entropy threshold on demand
            if args.thresh_level == "response":
                entropy_thresholds = [np.percentile(ent, percentile) for ent in base_entropies]
            elif args.thresh_level == "prompt":
                aggregated_entropies = [np.concatenate(base_entropies[i:i+args.n]) for i in range(0, len(base_entropies), args.n)]
                entropy_thresholds = [np.percentile(ent, percentile) for ent in aggregated_entropies]
                entropy_thresholds = sum([[x]*args.n for x in entropy_thresholds], [])  # repeat the threshold for each response
            elif args.thresh_level == "dataset":
                aggregated_entropies = np.concatenate(base_entropies)
                entropy_thresholds = [np.percentile(aggregated_entropies, percentile)] * len(base_entropies)  # use the same threshold for all responses
            else:
                raise ValueError(f"Unsupported threshold level: {args.thresh_level}. Choose from ['response', 'prompt', 'dataset'].")

        # high-entropy token selection
        high_entropy_idxs, prefixs_for_rl = [], []
        for idx, cur_prompt in enumerate(current_prompts):
            cur_entropies = np.array(base_entropies[idx])
            cur_generated_tokens = base_generated_tokens[idx]
            threshold = entropy_thresholds[idx]
            high_entropy_indices = np.where(cur_entropies > threshold)[0]
            # skip the prefix tokens at the first step
            if step == 0: high_entropy_indices = high_entropy_indices[high_entropy_indices >= args.num_prefix_keep]
            # if no high-entropy token over the threshold, use the highest entropy token
            if len(high_entropy_indices) == 0: high_entropy_indices = [np.argmax(cur_entropies)]
            high_entropy_idx = int(high_entropy_indices[0])  # Take the first high-entropy token index
            prefix_for_rl = cur_prompt + cur_generated_tokens[:high_entropy_idx]
            high_entropy_idxs.append(high_entropy_idx)
            # NOTE: there is no need for the prefix_for_rl (prompt + response[:idx]) to be truncated, 
            # since we have: len(prompt) + len(response) <= MAX_MODEL_LEN + 1
            # and for idx < len(response), we have len(prompt) + len(response[:idx]) <= MAX_MODEL_LEN
            prefixs_for_rl.append(prefix_for_rl)
            # log
            if idx == 0:
                print(f"Prefix for RL: {tokenizer.decode(prefix_for_rl)}")
                print(f"Selected token: [{tokenizer.decode([cur_generated_tokens[high_entropy_idx]])}], " \
                      f"entropy: {cur_entropies[high_entropy_idx]:.4f}, Threshold: {threshold:.4f}")
    
        # RL model resampling
        rl_outputs = generate_with_formatted_return(rl_model, prefixs_for_rl, rl_sampling_params)
        rl_entropies, rl_logprobs, rl_generated_tokens, rl_responses = rl_outputs
        rl_resample_end_idxs = []
        for idx, cur_rl_prompt in enumerate(prefixs_for_rl):
            # replace until entropy is lower than threshold or only one token is resampled
            end_idx = 1
            if args.rl_tokens_stop == "hard":
                while end_idx < len(rl_entropies[idx]) and rl_entropies[idx][end_idx] >= entropy_thresholds[idx]:
                    end_idx += 1
            elif args.rl_tokens_stop == "mean":
                mean_entropies = np.array([np.mean(rl_entropies[idx][:j+1]) for j in range(len(rl_entropies[idx]))])
                valid_idxs = np.where(mean_entropies >= entropy_thresholds[idx])[0]
                end_idx = int(valid_idxs[-1]) + 1 if len(valid_idxs) > 0 else 1  # at least replace one token
            replace_tokens = rl_generated_tokens[idx][:end_idx]
            if replace_tokens[-1] == tokenizer.eos_token_id:
                # Handling eot token, remove is preferred.
                if args.eot_replace == "keep":
                    pass # do nothing, keep the <|endoftext|> token
                elif args.eot_replace == "remove":
                    replace_tokens = replace_tokens[:-1]
                    end_idx -= 1  # since we removed the last token, we need to adjust the end index
                elif args.eot_replace == "replace":
                    last_logprobs = rl_logprobs[idx][:end_idx][-1]
                    another_top_token = get_another_top_token(last_logprobs, tokenizer.eos_token_id)
                    if another_top_token is not None:
                        replace_tokens[-1] = another_top_token  # replace <|endoftext|> with another top token
                    else:
                        print(f"Warning: No alternative token found to replace <|endoftext|> in RL resampling.")

            current_prompts[idx] = cur_rl_prompt + replace_tokens
            rl_resample_end_idxs.append(end_idx)  # record the end index for each prompt
            # handle longer than context length (lazy)
            # NOTE: base model's response to the max-length prompt will be a single token, which will be removed for RL resampling,
            # then any RL-resampled tokens will be truncated again in the following lines, so the current prompt will keep unchanged
            if len(current_prompts[idx]) > MAX_MODEL_LEN:
                print(f"Warning: Truncating prompt to fit model max length ({len(current_prompts[idx])} > {MAX_MODEL_LEN}).")
                current_prompts[idx] = current_prompts[idx][:MAX_MODEL_LEN]
            if tokenizer.decode(replace_tokens) != tokenizer.decode(replace_tokens, skip_special_tokens=True):
                print(f"Warning: The replacement tokens ({tokenizer.decode(replace_tokens)}) contain special tokens.")
            # log
            if idx == 0:
                print(f"RL resampled: [{rl_responses[idx]}], entropies: {rl_entropies[idx]},"\
                      f"selected for replace: [{tokenizer.decode(replace_tokens)}] (idx<{end_idx}).")

        # record info
        info_list, avg_accs = [], []
        for idx, gt in enumerate(gts):
            start, end = idx * args.n, (idx + 1) * args.n
            p_responses = base_responses[start:end]  # all responses for this prompt
            # full responses
            p_full_responses = [tokenizer.decode(current_prompts[i] + base_generated_tokens[i]) for i in range(start, end)]
            p_replace_infos = []  # record each response's replacement info
            for global_idx in range(start, end):
                replace_idx = high_entropy_idxs[global_idx]
                p_replace_infos.append({
                    "replace_idx": replace_idx,
                    "replace_token": tokenizer.decode([base_generated_tokens[global_idx][replace_idx]]),
                    "replace_entropy": base_entropies[global_idx][replace_idx],
                    "replace_logprobs": format_vllm_logp(base_logprobs[global_idx][replace_idx], tokenizer),
                    "resampled_end_idx": rl_resample_end_idxs[global_idx],
                    "resampled_response": rl_responses[global_idx],
                    "resampled_entropies": rl_entropies[global_idx],
                    "resampled_logprobs": format_vllm_logp(rl_logprobs[global_idx], tokenizer),
                })
            # calculate the accs
            acc_list, pred_answers = get_acc_list(p_full_responses, gt, True)
            info_list.append({
                "prompt": prompts[idx],
                "ground_truth": gt,
                "responses": p_responses,
                "full_responses": p_full_responses,
                "pred_answers": pred_answers,
                "accs": acc_list,
                "entropy_thresholds": entropy_thresholds[start:end],
                "replacements": p_replace_infos,
                "final_prefixs": current_prompts[start:end] if step == args.max_rl_resample else None,
            })
            avg_accs.append(np.mean(acc_list))
        print(f"** Step {step} Summary **\n")
        print(f"Average accuracy: {np.mean(avg_accs):.4f}")
        print(f"Average length kept: {np.mean(high_entropy_idxs):.2f}\n")
        all_info[f'step_{step}'] = info_list
    return all_info


def run_exp_aime(args:argparse.Namespace, base_model: LLM, rl_model: LLM, tokenizer:AutoTokenizer, 
                 ds:datasets.Dataset, prompt_key='problem', gt_key='answer', continue_info=None):
    # Prepare prompts
    suffix = "\nPlease reason step by step, and put your final answer within \\boxed{}."
    gen_prefix = "To approach this problem" if args.include_prefix else ""
    if args.use_chat:
        prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': q.strip() + suffix}], tokenize=False, 
                                                 add_generation_prompt=True) + gen_prefix for q in ds[prompt_key]]
    else:
        prompts = [q.strip() + suffix + '\n' + gen_prefix for q in ds[prompt_key]]
    print(f"Total prompts: {len(prompts)}.\nExample:\n{prompts[0]}")
    # Prepare ground truths and run iterative resampling
    gts = ds[gt_key]
    ret_info = iterative_rl_resample(args, base_model, rl_model, tokenizer, prompts, gts, continue_info=continue_info)

    # Save results
    with open(args.save_file, 'w') as f:
        json.dump(ret_info, f, indent=2)
        print(f"Results saved to {args.save_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterative RL Resampling with VLLM")
    # MODEL AND DATASET
    parser.add_argument("--base_model", type=str, default="hf_models/Qwen2.5-32B")
    parser.add_argument("--rl_model", type=str, default="hf_models/DAPO-Qwen-32B")
    parser.add_argument("--dataset", type=str, default="hf_datasets/aime24")
    # SAMPLING PARAMETERS
    parser.add_argument("--n", type=int, default=32, help="Number of samples to generate for each prompt")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Maximum number of new tokens to generate")
    parser.add_argument("--top_p_base", type=float, default=0.7, help="Top-p for base model sampling")
    parser.add_argument("--top_p_rl", type=float, default=0.7, help="Top-p for RL model sampling")
    parser.add_argument("--temperature_base", type=float, default=1.0, help="Temperature for base model sampling")
    parser.add_argument("--temperature_rl", type=float, default=1.0,  help="Temperature for RL model sampling")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument("--num_logprobs", type=int, default=50, help="Number of logprobs to return")
    parser.add_argument("--tp_size", type=int, default=8, help="Tensor parallel size for vLLM")
    parser.add_argument("--base_mem", type=float, default=0.5, help="GPU memory utilization for vLLM")
    parser.add_argument("--rl_mem", type=float, default=0.3, help="GPU memory utilization for vLLM")
    # RESAMPLING PARAMETERS
    parser.add_argument("--use_chat", type=int, default=0, help="Whether to use chat template (1 for chat, 0 for text prompts)")
    parser.add_argument("--max_rl_resample", type=int, default=30, help="Maximum number of RL resampling iterations")
    parser.add_argument("--top_ent", type=float, default=0.1, help="Top entropy percentage for selecting high-entropy tokens")
    parser.add_argument("--dyna_thresh", type=int, default=0, help="Whether to use dynamic thresholding for entropy selection")
    parser.add_argument("--thresh_level", type=str, default="prompt", choices=["response", "prompt", "dataset"], help="Level for entropy thresholding")
    parser.add_argument("--rl_tokens", type=int, default=1, help="Number of tokens to resample with RL model at each time")
    parser.add_argument("--rl_tokens_stop", type=str, default="hard", choices=["hard", "mean"], help="RL resampled tokens stopping strategies")
    parser.add_argument("--num_prefix_keep", type=int, default=0, help="Number of prefix tokens to keep in the first step")
    parser.add_argument("--include_prefix", type=int, default=1, help="Whether to include RL prefix pattern in the beginning") 
    parser.add_argument("--eot_replace", type=str, default="remove", choices=["keep", "replace", "remove"],
                        help="How to handle <|endoftext|> token in RL resampling: 'keep' to keep it, 'replace' to replace with another top-prob token, 'remove' to remove it")
    # OTHERS
    parser.add_argument("--save_dir", type=str, default="resample_results")
    parser.add_argument("--continue_from", type=int, default=0, help="Continue from a specific step (0 for fresh run)")
    args = parser.parse_args()
    print(f"Arguments: {args}")
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    def get_filename(args:argparse.Namespace):
        threshold_name = f"{'dyna' if args.dyna_thresh else 'fixed'}_{args.thresh_level}_thresh"
        rl_tokens_name = f"{args.rl_tokens}_rl_tokens-stop_by_{args.rl_tokens_stop}"
        save_name = f"{args.base_model.split('/')[-1]}-{args.rl_model.split('/')[-1]}-{args.dataset.split('/')[-1]}-seed{args.seed}-resample{args.max_rl_resample}"
        save_name += f"-top_ent{args.top_ent}-{threshold_name}-{rl_tokens_name}-{args.eot_replace}_eot-use_chat{args.use_chat}-include_prefix{args.include_prefix}"
        save_name += f"-keep_{args.num_prefix_keep}_prefix-top_p_{args.top_p_base}_{args.top_p_rl}-t_{args.temperature_base}_{args.temperature_rl}-n{args.n}"
        return os.path.join(args.save_dir, f"{save_name}.json")
    args.save_file = get_filename(args)
    print(f"Results will be saved to: {args.save_file}")

    # handling continue from a specific step
    continue_info = None
    if args.continue_from > 0:
        args_copied = deepcopy(args)
        args_copied.max_rl_resample = args.continue_from
        load_from_file = get_filename(args_copied)
        if not os.path.exists(load_from_file):
            raise FileNotFoundError(f"Cannot continue from step {args.continue_from}, since file {load_from_file} does not exist.")
        print(f"Continuing from step {args.continue_from}, loading from {load_from_file}")
        with open(load_from_file, 'r') as f:
            continue_info = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    dataset = load_dataset(args.dataset, split="train")
    base_model = LLM(model=args.base_model, tensor_parallel_size=args.tp_size, max_logprobs=args.num_logprobs,
                     gpu_memory_utilization=args.base_mem, enable_prefix_caching=True, enforce_eager=True)
    rl_model = LLM(model=args.rl_model, tensor_parallel_size=args.tp_size, max_logprobs=args.num_logprobs,
                   gpu_memory_utilization=args.rl_mem, enable_prefix_caching=True, enforce_eager=True)

    run_exp_aime(args, base_model, rl_model, tokenizer, dataset, prompt_key='problem', gt_key='answer', continue_info=continue_info)

