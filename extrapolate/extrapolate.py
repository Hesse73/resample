import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0" # make the scheduling deterministic
import time
import json
import sys
sys.path.append('..')
import argparse
import datasets
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from copy import deepcopy
from math_utils import get_acc_list
from typing import List, Tuple
from scipy.special import softmax, logsumexp


def weighted_sampling(outputs, outputs_assistant, disable_mask:List[bool], sampling_params:SamplingParams, weights:Tuple[float, float]=(1.05, -0.05)):
    """
    Perform weighted sampling based on the logprobs from two models.
    """
    sampled_next_tokens = []
    for output, output_assistant, disable in zip(outputs, outputs_assistant, disable_mask):
        if disable:
            # If the weighted sampling is disabled, we just use the main model's output
            sampled_next_tokens.append(output.outputs[0].token_ids[0]); continue
        # 0-idx means the first (and only) response and token respectively
        out, out_assistant = output.outputs[0], output_assistant.outputs[0]
        lp_info, lp_info_assistant = out.logprobs[0], out_assistant.logprobs[0]
        token_ids, logprobs = [key for key in lp_info], np.array([lp.logprob for lp in lp_info.values()])
        logp_map_assistant = {key: lp.logprob for key, lp in lp_info_assistant.items()}
        # 1. align the assistant's logprobs with the main model's token_ids
        pad_lp = min(logp_map_assistant.values())
        aligned_logprobs_assistant = np.array([logp_map_assistant[tid] if tid in logp_map_assistant \
                                               else pad_lp for tid in token_ids])
        # 2. reweight the logprobs & apply temperature
        logits = logprobs * weights[0] + aligned_logprobs_assistant * weights[1]
        logits /= sampling_params.temperature 
        # 3. top-p sampling
        logits_sorted_indices = np.argsort(logits)
        logits_sorted = logits[logits_sorted_indices]
        top_p_mask = ~(np.cumsum(softmax(logits_sorted)) <= 1 - sampling_params.top_p)
        top_p_mask[-1] = True  # Ensure the last token (highest prob) is always included
        sampled_idx = np.random.choice(logits_sorted_indices[top_p_mask], p=softmax(logits_sorted[top_p_mask]))
        sampled_next_tokens.append(token_ids[sampled_idx])
    return sampled_next_tokens

def decode_with_two_models(args:argparse.Namespace, llm:LLM, llm_assistant:LLM, prompts:List[str], gts:List[str]):
    tokenizer = llm.get_tokenizer()
    initial_prompts = sum([[tokenizer.encode(p, add_special_tokens=False) for p in prompts]] * args.n, [])
    all_generated_texts = {i:[] for i in range(len(initial_prompts))}
    print(f"Total prompts: {len(initial_prompts)}\nExample:\n{tokenizer.decode(initial_prompts[0])}")

    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=1,
        logprobs=args.logprobs,
        seed=args.seed,
        skip_special_tokens=False
    )
    window_size = len(prompts)
    # window_size = len(initial_prompts)
    cur_prompts = deepcopy(initial_prompts[:window_size])
    cur_idxs = list(range(len(cur_prompts)))
    cur_lengths = [0] * len(cur_prompts)
    max_idx = max(cur_idxs) + 1
    pbar = tqdm(total=len(initial_prompts), desc="Processing prompts")
    while cur_idxs:
        # We generate one token for each prompt in cur_prompts
        # If any response reaches EOS or the max tokens, we remove it from cur_idxs & cur_prompts, and add new prompts & idx into cur_prompts & cur_idxs
        start = time.time()
        inputs = [TokensPrompt(prompt_token_ids=p) for p in cur_prompts]
        outputs = llm.generate(inputs, sampling_params=sp, use_tqdm=False)
        outputs_assistant = llm_assistant.generate(inputs, sampling_params=sp, use_tqdm=False)
        # reweighting
        # sampled_next_tokens = [out.outputs[0].token_ids[0] for out in outputs]
        sampled_next_tokens = weighted_sampling(outputs, outputs_assistant, [False] * len(cur_prompts), sp, weights=args.weights)
        # update responses, prompts, and lengths
        finished_idxs = []
        for i, next_token in enumerate(sampled_next_tokens):
            all_generated_texts[cur_idxs[i]].append(next_token)
            cur_prompts[i] += [next_token]
            cur_lengths[i] += 1
            if next_token == tokenizer.eos_token_id or cur_lengths[i] >= args.max_tokens:
                finished_idxs.append(i)
        for i in reversed(finished_idxs):  # reverse to avoid index issues
            cur_idxs.pop(i)
            cur_prompts.pop(i)
            cur_lengths.pop(i)
        pbar.update(len(finished_idxs))
        # Then we add new prompts if there are still prompts to generate
        while max_idx < len(initial_prompts) and len(cur_prompts) < window_size:
            cur_prompts.append(deepcopy(initial_prompts[max_idx]))
            cur_idxs.append(max_idx)
            cur_lengths.append(0)
            max_idx += 1
        pbar.set_postfix_str(f"{2*len(cur_prompts)/(time.time()-start):.1f} tok/s") # 2* for two models
    pbar.close()

    all_info, avg_accs = [], []
    for idx, gt in enumerate(gts):
        # we get all the generated texts for the each prompt (which is repeated n times)
        cur_generated_texts = [all_generated_texts[i*len(prompts)+idx] for i in range(args.n)]
        cur_generated_texts = [tokenizer.decode(t, skip_special_tokens=False) for t in cur_generated_texts]
        accs = get_acc_list(cur_generated_texts, gt)
        all_info.append({
            'prompt': prompts[idx],
            'gt': gt,
            'generated_texts': cur_generated_texts,
            'accs': accs,
        })
        avg_accs.append(np.mean(accs))
    print(f"Average accuracy: {np.mean(avg_accs):.4f} ± {np.std(avg_accs):.4f}")
    return all_info



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode with two models")
    parser.add_argument("--model", type=str, default="../hf_models/DAPO-Qwen-32B", help="Path to the first model")
    parser.add_argument("--assistant", type=str, default="../hf_models/Qwen2.5-32B", help="Path to the second model")
    parser.add_argument("--dataset", type=str, default="../hf_datasets/aime24", help="Path to the dataset")
    # sampling params
    parser.add_argument("--n", type=int, default=32, help="#responses per prompt")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.7, help="Top-p for sampling")
    parser.add_argument("--logprobs", type=int, default=50, help="Number of logprobs to return")
    parser.add_argument("--max_tokens", type=int, default=20_000, help="Maximum tokens to generate")
    parser.add_argument("--gpu_util", type=float, default=0.4, help="GPU utilization for the model")
    parser.add_argument("--tp_size", type=int, default=8, help="Tensor parallel size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    # weights for the two models
    parser.add_argument("--weights", type=float, nargs=2, default=(1.0, 0.0), 
                        help="Weights for the two models' logprobs, e.g., 1.05 -0.05")
    # others
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save results")
    args = parser.parse_args()
    np.random.seed(args.seed)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if sum(args.weights) != 1.0:
        print(f"Warning: weights {args.weights} do not sum to 1.0, normalizing them.")
        args.weights = [w / sum(args.weights) for w in args.weights]
    save_path = f"{os.path.basename(args.model)}-{os.path.basename(args.assistant)}-{os.path.basename(args.dataset)}"
    save_path += f"-w{args.weights[0]}_{args.weights[1]}"
    save_path += f"-n{args.n}-t{args.temperature}-top_p{args.top_p}-seed{args.seed}.json"
    print(f"Results will be saved to {os.path.join(args.save_dir, save_path)}")
    
    # Load dataset
    ds = datasets.load_dataset(args.dataset, split='train')
    prompts, gts = ds['problem'], ds['answer']

    # Load models
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util, enable_prefix_caching=True, 
              enforce_eager=True, tensor_parallel_size=args.tp_size, max_logprobs=args.logprobs)
    assistant = LLM(model=args.assistant, gpu_memory_utilization=args.gpu_util, enable_prefix_caching=True, 
              enforce_eager=True, tensor_parallel_size=args.tp_size, max_logprobs=args.logprobs)

    # process the prompts
    tokenizer = llm.get_tokenizer()
    suffix = "\nPlease reason step by step, and put your final answer within \\boxed{}."
    prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': q.strip() + suffix}], tokenize=False, 
                                             add_generation_prompt=True) for q in prompts]
    
    info = decode_with_two_models(args, llm, assistant, prompts, gts)
    json.dump(info, open(os.path.join(args.save_dir, save_path), "w"), indent=4, ensure_ascii=False)
