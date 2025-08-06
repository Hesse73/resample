import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0" # make the scheduling deterministic
import sys
sys.path.append('..')
import json
import argparse
import datasets
import numpy as np
from vllm import LLM, SamplingParams
from math_utils import get_acc_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--model", type=str, default="../hf_models/DAPO-Qwen-32B", help="Path to the first model")
    parser.add_argument("--dataset", type=str, default="../hf_datasets/aime24", help="Path to the dataset")
    # sampling params
    parser.add_argument("--n", type=int, default=32, help="#responses per prompt")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.7, help="Top-p for sampling")
    parser.add_argument("--logprobs", type=int, default=50, help="Number of logprobs to return")
    parser.add_argument("--max_tokens", type=int, default=20_000, help="Maximum tokens to generate")
    parser.add_argument("--gpu_util", type=float, default=0.9, help="GPU utilization for the model")
    parser.add_argument("--tp_size", type=int, default=8, help="Tensor parallel size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")

    args = parser.parse_args()
    save_path = f'_test_results/{args.model.split("/")[-1]}_{args.dataset.split("/")[-1]}_results.json'
    if os.path.exists(save_path):
        print(f"Results already exist at {save_path}. Skipping evaluation.")
        # load existing results
        infos = json.load(open(save_path, 'r'))
        avg_acc = np.mean([info['accs'] for info in infos])
        print(f"Average accuracy from existing results: {avg_acc:.2f}")
        sys.exit(0)

    # Load dataset
    if 'aime24' in args.dataset:
        ds = datasets.load_dataset(args.dataset, split='train')
    elif 'aime25' in args.dataset:
        ds = datasets.load_dataset(args.dataset, split='test')
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    prompts, gts = ds['problem'], ds['answer']


    # Load models
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util, enable_prefix_caching=True, 
              enforce_eager=True, tensor_parallel_size=args.tp_size, max_logprobs=args.logprobs,
              max_num_batched_tokens=327680)
    sp = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        logprobs=args.logprobs,
        seed=args.seed,
        skip_special_tokens=False
    )

    # process the prompts
    tokenizer = llm.get_tokenizer()
    suffix = "\nPlease reason step by step, and put your final answer within \\boxed{}."
    prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': q.strip() + suffix}], tokenize=False, 
                                             add_generation_prompt=True) for q in prompts]
    print('Example prompt:', prompts[0])
    outputs = llm.generate(prompts, sampling_params=sp)
    
    avg_accs = []
    infos = []
    for p, gt, output in zip(prompts, gts, outputs):
        generated_texts = [out.text for out in output.outputs]
        accs, answers = get_acc_list(generated_texts, gt, return_parsed=True)
        infos.append({
            'prompt': p,
            'gt': gt,
            'answers': answers,
            'accs': accs,
        })
        avg_accs.append(np.mean(accs))
    
    avg_acc = np.mean(avg_accs)
    print(f"Average accuracy: {avg_acc:.2f}")
    
    # Save results
    json.dump(infos, open(save_path, 'w'), indent=4)