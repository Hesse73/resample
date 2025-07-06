import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

## VLLM
import os
from vllm import LLM, SamplingParams

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0" # to set seed

base_model_name = "/mgfs/shared/Group_GY/hf_models/Qwen2.5-32B"
rl_model_name = "/mgfs/shared/Group_GY/hf_models/DAPO-Qwen-32B"
# base_model_name = "hf/Qwen2.5-32B" 
# rl_model_name = "/mnt/workspace/xue.w/Qwen/Qwen2.5-32B/DAPO-Qwen-32B" 

base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load base and RL models using vLLM
mem_util = 0.4
rl_model = LLM(model=rl_model_name, tensor_parallel_size=8, max_logprobs=1000,
               gpu_memory_utilization=mem_util, enable_prefix_caching=True)
base_model = LLM(model=base_model_name, tensor_parallel_size=8, max_logprobs=1000,
                gpu_memory_utilization=mem_util, enable_prefix_caching=True)

def token_entropy(probs):
    """Compute entropy from probability distribution"""
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).cpu().numpy()

def format_prompt_with_system_message(prompt, tokenizer, system_message="You are a helpful assistant."):
    """
    Formats the input prompt with a system message using the chat template from the tokenizer.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return formatted_text

def format_prompt(prompt, tokenizer):
    """
    Formats the input prompt with a system message using the chat template from the tokenizer.
    """
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return formatted_text

def generate_with_entropy_return_full_vllm_batch(model, tokenizer, prompts, device='cuda', top_p=0.7, max_new_tokens=200, start_token=False, seed=1):
    # inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # input_ids = inputs["input_ids"]

    all_entropies = []
    all_generated_tokens = []
    all_probabilities = []
    all_texts = []

    num_logprobs = 50

    sampling_params = SamplingParams(
        temperature=1,
        top_p=top_p,
        max_tokens=max_new_tokens,
        seed=seed,
        logprobs=num_logprobs,
        skip_special_tokens=False
    )
        
    with torch.no_grad():
        outputs = model.generate(prompts, sampling_params=sampling_params)

    print(len(outputs))
    for output in outputs:
        out = output.outputs[0]
        entropies = []
        generated_tokens = []
        probabilities = []
        generated_ids = out.token_ids  # Generated token IDs
        logprob_list = out.logprobs  # List of logprob dicts for each step
        

        # Iterate over each generated token's logprobs
        for step_logprobs in logprob_list:
            probs = torch.tensor([lp.logprob for lp in step_logprobs.values()]).exp()
            probabilities.append(step_logprobs)
            probs /= probs.sum()

            # Compute entropy
            entropy = token_entropy(probs)
            entropies.append(entropy)
        
        all_entropies.append(entropies)
        all_probabilities.append(probabilities)
        all_generated_tokens.append(generated_ids)
        all_texts.append(out.text)

    return all_entropies, all_probabilities, all_generated_tokens, all_texts 


def batched_generate_with_entropy_return_full_vllm_sample(model, tokenizer, prompts, device='cuda', num_samples=5, 
                                                top_p=0.7, max_new_tokens=200, start_token=False, seed=1):
    # inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # input_ids = inputs["input_ids"]

    all_entropies = []
    all_generated_tokens = []
    all_probabilities = []
    all_texts = []

    num_logprobs = 50

    sampling_params = SamplingParams(
        temperature=1,
        top_p=top_p,
        max_tokens=max_new_tokens,
        n=num_samples,
        seed=seed,
        logprobs=num_logprobs,
        skip_special_tokens=False
    )
        
    with torch.no_grad():
        outputs = model.generate(prompts, sampling_params=sampling_params)

    for output in outputs:

        for out in output.outputs:
            entropies = []
            probabilities = []
            generated_ids = out.token_ids
            logprob_list = out.logprobs

            # Convert each token's logprobs to probs and compute entropy
            for step_logprobs in logprob_list:
                probs = torch.tensor([lp.logprob for lp in step_logprobs.values()]).exp()
                probabilities.append(step_logprobs)
                probs /= probs.sum()
                entropies.append(token_entropy(probs))

            all_entropies.append(entropies)
            all_probabilities.append(probabilities)
            all_generated_tokens.append(generated_ids)
            all_texts.append(out.text)

    return all_entropies, all_probabilities, all_generated_tokens, all_texts


def sample_batched_parallel_iterative_rl_resampling_vllm(base_model, rl_model, tokenizer, prompts, num_samples=5, top_p=0.7, top_p_rl=0.7,
                        max_new_tokens=2000, p=0.1, max_rl_resample=10, device='cuda', seed=1, num_prefix_keep=5, include_prefix=True):
    """
    Iteratively refine the base model's generation using the RL model for high-entropy tokens.
    
    Args:
        base_model: The base causal language model.
        rl_model: The RL fine-tuned model.
        tokenizer: Shared tokenizer.
        prompt: Initial prompt.
        p: Threshold percentile for entropy.
        max_rl_resample: Max number of refinement iterations.
        max_new_tokens: Max tokens to generate per iteration.

    Returns:
        Final refined sequence.
    """
    current_prompts = []
    current_prompt_lengths = []
    
    for prompt in prompts:
        current_prompts.extend([prompt] * num_samples)
        current_prompt_lengths.extend([0] * num_samples)

    all_responses = [[] for _ in range(len(prompts))]  # Store results per prompt
    all_replacements = [[] for _ in range(len(prompts))]
    all_lengths = [[] for _ in range(len(prompts))]
    all_num_kept = [[] for _ in range(len(prompts))]

    final_prefix = [] # Not actually needed
    # print(current_prompt)
    thresholds = []

    for i in range(max_rl_resample + 1):

        print(f"Iteration {i}:")
        responses = []
        lengths = []
        replacements = []
        num_kept = []

        start = time.time()
        # Generate with entropy
        if i == 0:
            batch_entropies, batch_probs, batch_generated_ids, batch_decoded_responses = batched_generate_with_entropy_return_full_vllm_sample(
                base_model, tokenizer, prompts, max_new_tokens=max_new_tokens, top_p=top_p, num_samples=num_samples, seed=seed
            )

        else:
            # TODO later should moodify to only generate until we get to token that crosses entropy threshold to be more efficient
            batch_entropies, batch_probs, batch_generated_ids, batch_decoded_responses = generate_with_entropy_return_full_vllm_batch(
                base_model, tokenizer, current_prompts, max_new_tokens=max_new_tokens, top_p=top_p, seed=seed
            )
        
        for samp in range(len(prompts) * num_samples):
            current_prompt = current_prompts[samp]
            entropies = batch_entropies[samp]
            #probs = base_probs[samp]

            curr_prompt_len = current_prompt_lengths[samp] + len(entropies)

            lengths.append(curr_prompt_len)

            if i == 0: # Set fixed threshold
                entropies_np = np.array(entropies)
                threshold = np.percentile(entropies_np, 100 - p * 100)
                thresholds.append(threshold)
                print(f"Threshold for sample {samp}: {threshold}")
                
            threshold = thresholds[samp]
            generated_ids = batch_generated_ids[samp]
            full_response = current_prompt + batch_decoded_responses[samp]
            # For the case that we store all responses for each number of RL resampled tokens
            responses.append(full_response)

            print("Number of generated tokens:", len(generated_ids))
            
            if i == 0:
                high_entropy_indices = [i for i, e in enumerate(entropies) if (e > threshold and i > num_prefix_keep)]
            else:
                high_entropy_indices = [i for i, e in enumerate(entropies) if e > threshold]
        
            if not high_entropy_indices:
                print("Taking token with highest entropy")
                high_entropy_indices = [np.argmax(entropies)]
                #current_prompts[samp] = prefix + rl_token_text
        
        # Replace high-entropy token with RL-generated token
            high_entropy_index = int(high_entropy_indices[0])  # Pick first high-entropy token
        #prefix = generated_ids[:, :input_length - 1 + high_entropy_index + 1]  # Include up to that token
            # generated_ids starts from first predicted token, so we don't want to include the high-entropy prediction
            prefix = current_prompt + tokenizer.decode(generated_ids[: high_entropy_index]) 
            replaced_token = tokenizer.decode(generated_ids[high_entropy_index])

            print("Number of tokens kept:", high_entropy_index)
            current_prompt_lengths[samp] += high_entropy_index + 1 # +1 from the rl generated token
            num_kept.append(high_entropy_index)
            
        # Generate replacement token from RL model
            num_logprobs = 50
            with torch.no_grad(): # TODO can probably batch this later on but doesn't seem like a bottleneck
                sampling_params_rl = SamplingParams(
                    temperature=1,
                    top_p=top_p_rl,
                    seed=seed,
                    logprobs=num_logprobs,
                    max_tokens=1,
                    #skip_special_tokens=False
                )
                rl_output = rl_model.generate(
                    [prefix],
                    sampling_params=sampling_params_rl,
                    use_tqdm=False
                )
                rl_token_text = rl_output[0].outputs[0].text  # Get the sampled token
            
            #replacements.append({replaced_token:probs[high_entropy_index].tolist(), rl_token_text:rl_output[0].outputs[0].logprobs})
            #replacements.append((replaced_token, rl_token_text))
            logprobs = rl_output[0].outputs[0].logprobs[0]
            rl_probs = torch.tensor([lp.logprob for lp in logprobs.values()]).exp().cpu().numpy()
            base_logprobs = batch_probs[samp][high_entropy_index]
            base_probs = torch.tensor([lp.logprob for lp in base_logprobs.values()]).exp().cpu().numpy()

            token_probs_rl = [(token_id, float(prob)) for token_id, prob in zip(logprobs.keys(), rl_probs)]
            token_probs_base = [(token_id, float(prob)) for token_id, prob in zip(base_logprobs.keys(), base_probs)]
            replacements.append(((replaced_token, token_probs_base), (rl_token_text, token_probs_rl)))

        # Replace the token in the sequence, and continue generating from that position onwards for the next iteration
        #generated_ids = torch.cat([prefix, rl_token], dim=1)

        # Update the prompt with the modified response
            #current_prompt = prefix + rl_token_text
            current_prompts[samp] = prefix + rl_token_text
            if i == max_rl_resample:
                final_prefix.append(prefix + rl_token_text)
        
        for prompt_index in range(len(prompts)):
            start = prompt_index * num_samples
            end = start + num_samples
            all_responses[prompt_index].append(responses[start:end])
            all_replacements[prompt_index].append(replacements[start:end])
            all_lengths[prompt_index].append(lengths[start:end])
            all_num_kept[prompt_index].append(num_kept[start:end])

    final_current_prompts = []
    for prompt_index in range(len(prompts)):
        start = prompt_index * num_samples
        end = start + num_samples
        final_current_prompts.append(current_prompts[start:end])
        #print("Current_prompt:")
        #print(current_prompt)

    return all_responses, all_lengths, all_replacements, all_num_kept, final_current_prompts


import re

def extract_final_answer(response: str) -> str:
    """
    Post-processing function to extract the final boxed answer from the response.
    Modify this depending on your task format.
    """
    # Example: Extract content inside \boxed{}
    match = re.search(r"\\boxed{([^}]*)}", response)
    return match.group(1).strip() if match else ""

def extract_last_boxed(text):
    """Extract content inside the last \boxed in LaTeX text"""
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    matches = list(re.finditer(pattern, text))
    # print(matches[-1])
    # print(matches[-1].group(0))
    # print(matches[-1].group(1))
    if matches:
        # return matches[-1].group(0) # seems like this returns the boxed, which we don't want (at least so far)
        return matches[-1].group(1) # This gets rid of the boxed
    #return None
    return ""

def extract_all_boxed(text):
    """Extract content inside the last \boxed in LaTeX text"""
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    matches = list(re.finditer(pattern, text))
    # print(matches[-1])
    # print(matches[-1].group(0))
    # print(matches[-1].group(1))
    if matches:
        # return matches[-1].group(0) # seems like this returns the boxed, which we don't want (at least so far)
        return [match.group(1) for match in matches] # This gets rid of the boxed
    #return None
    return [""]


def is_correct(prediction: str, ground_truth: str) -> bool:
    """
    Compares predicted answer with ground truth.
    """
    try:
        # Try numeric comparison
        return float(prediction) == float(ground_truth)
    except ValueError:
        # Fallback to string comparison
        return prediction.strip() == ground_truth.strip()

def response_is_correct(response, ground_truth):
    pred_last = extract_last_boxed(response)
    pred_all = extract_all_boxed(response)
    correct = is_correct(pred_last, ground_truth)
    correct_forgive = correct
    for pred in pred_all:
        if is_correct(pred, ground_truth):
            correct_forgive = True
            break
    return correct, correct_forgive
        


def aggregate_accuracy_per_prompt(samples, expected_answer):
    correct = 0
    predictions = []
    for response in samples:
        #pred = extract_final_answer(response)
        pred = extract_last_boxed(response)
        predictions.append(pred)
        if is_correct(pred, expected_answer):
            correct += 1
            print("Correct:", pred)
        else:
            print("Incorrect:", pred)
            #print(response)
    return correct / len(samples), predictions

def aggregate_accuracy_per_prompt_forgiving(samples, expected_answer):
    correct = 0
    final_correct = 0
    predictions = []
    for response in samples:
        #pred = extract_final_answer(response)
        preds = extract_all_boxed(response)
        predictions.append(preds)
        #print("Number of predictions:", len(preds))
        corr = False
        if is_correct(preds[-1], expected_answer):
            final_correct += 1
        for pred in preds:
            if is_correct(pred, expected_answer):
                corr = True
                #print("Correct:", pred)
                #print("No need to evaluate further")
                break
            #else:
                #print("Incorrect:", pred)
                #print(response)
        if corr:
            correct += 1
    return correct / len(samples), final_correct / len(samples), predictions

def aggregate_accuracy_per_prompt_group(grouped_samples, expected_answer):
    """
    For the case that we store across different number of RL resampling tokens
    """
    all_acc = []
    all_preds = []
    for i, samples in enumerate(grouped_samples):
        print("Number of RL resamples:", i)
        acc, preds = aggregate_accuracy_per_prompt(samples, expected_answer)
        all_acc.append(acc)
        all_preds.append(preds)
    return all_acc, all_preds
        

def aggregate_accuracy_per_prompt_group_forgiving(grouped_samples, expected_answer):
    """
    For the case that we store across different number of RL resampling tokens
    """
    all_acc = []
    all_acc_forgive = []
    all_preds = []
    for i, samples in enumerate(grouped_samples):
        print("Number of RL resamples:", i)
        acc_forgive, acc, preds = aggregate_accuracy_per_prompt_forgiving(samples, expected_answer)
        all_acc.append(acc)
        all_acc_forgive.append(acc_forgive)
        all_preds.append(preds)
    return all_acc_forgive, all_acc, all_preds

# ### Experiment Loop

from tqdm import tqdm
import json
import time

def write_json(data, filename):
    # Open a file in write mode ('w') and dump the data
    with open(filename, "w") as file:
        json.dump(data, file, indent=4) 

def run_batched_rl_resampling_experiment_vllm(
    # prompts,
    dataset,
    base_model,
    rl_model,
    tokenizer,
    num_samples_per_prompt=8,
    p=0.1,
    max_rl_resample=10,
    max_new_tokens=2000,
    seed=1,
    num_prefix_keep=5,
    top_p=0.7,
    top_p_rl=0.7,
    batch_size=8,
    no_skip=None,
    include_prefix=False
):
    results = {
        "base": {"accuracies_forgive": [], "accuracies": [], "responses": []},
        "rl": {"accuracies_forgive": [], "accuracies": [], "responses": []},
        #"iterative_rl_resampling": {"accuracies": [], "responses": []}
    }
    all_responses = {
        "base": [],
        "rl": []
    }
    stored_replace = {}

    for i in range(max_rl_resample+1):
        results[f"iterative_rl_resampling_{i}"] = {"accuracies_forgive": [], "accuracies": [], "responses": []}
        all_responses[f"iterative_rl_resampling_{i}"] = []
        stored_replace[f"iterative_rl_resampling_{i}"] = []
    
    num = 0
    
    suffix = "\nPlease reason step by step, and put your final answer within \\boxed{}."

    if include_prefix:
        rl_prefix = "To approach this problem"
    else:
        rl_prefix = ""
    

    #for item in tqdm(prompts, desc="Evaluating Prompts"):
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]  # Get a batch of examples
        questions = batch["problem"]
        expected_answers = batch["answer"]
        prompt_ids = batch["id"]

        # Format each prompt with chat template
        formatted_questions = [format_prompt(q + suffix, base_tokenizer) + rl_prefix for q in questions]
        

        print("Formatted Question (first):", formatted_questions[0])
        print("Expected Answer (first):", expected_answers[0])

        # Sample Iterative RL Resampling
        print("Running base with Iterative RL Resampling...")
        all_iter_samples, all_lengths, all_replacements, all_num_kept, all_final_prefix = sample_batched_parallel_iterative_rl_resampling_vllm(
            base_model, rl_model, tokenizer, formatted_questions, num_samples=num_samples_per_prompt, top_p=top_p, top_p_rl=top_p_rl,
            max_new_tokens=max_new_tokens, p=p, max_rl_resample=max_rl_resample, seed=seed, num_prefix_keep=num_prefix_keep, include_prefix=include_prefix
        )

        # for sample in iter_samples:
        #     print(sample)
        for j in range(len(questions)):
            question = questions[j]
            expected = expected_answers[j]
            iter_samples = all_iter_samples[j]
            final_prefix = all_final_prefix[j]
            iter_acc_forgive, iter_acc, iter_preds = aggregate_accuracy_per_prompt_group_forgiving(iter_samples, expected)

            for i, (acc_forgive, acc, preds, lengths, replacements, num_kept) in enumerate(zip(iter_acc_forgive, iter_acc, 
                                                            iter_preds, all_lengths[j], all_replacements[j], all_num_kept[j])):
                results[f"iterative_rl_resampling_{i}"]["accuracies_forgive"].append(acc_forgive)
                results[f"iterative_rl_resampling_{i}"]["accuracies"].append(acc)
                results[f"iterative_rl_resampling_{i}"]["responses"].append({
                    "prompt": question,
                    "predictions": preds,
                    "expected": expected,
                    "lengths": lengths,
                    "num_tokens_kept": num_kept
                })
                stored_replace[f"iterative_rl_resampling_{i}"].append({
                    "prompt": question,
                    "replacements": replacements,
                })
                if i == max_rl_resample:
                    all_responses[f"iterative_rl_resampling_{i}"].append({
                        "prompt": question,
                        "responses": iter_samples[i],
                        "final_prefixes": final_prefix
                    })
                else:
                    all_responses[f"iterative_rl_resampling_{i}"].append({
                        "prompt": question,
                        "responses": iter_samples[i],
                    })

        num += 1
        date = "july4"
        os.makedirs(f"rl_resample_results/Qwen2.5-32B/{date}", exist_ok=True)
        if include_prefix:
            filename = f"rl_resample_results/Qwen2.5-32B/{date}/ids{prompt_ids[0]}-{prompt_ids[-1]}_rl_prefix_seed{seed}_top_p{top_p}_samples{num_samples_per_prompt}_resample{max_rl_resample}_p{p}_keep{num_prefix_keep}_first_only.json"
            filename_responses = f"rl_resample_results/Qwen2.5-32B/{date}/responses_rl_prefix_id{prompt_ids[0]}-{prompt_ids[-1]}_seed{seed}_top_p{top_p}_samples{num_samples_per_prompt}_resample{max_rl_resample}_p{p}_keep{num_prefix_keep}_first_only.json"
            filename_replace = f"rl_resample_results/Qwen2.5-32B/{date}/replace_ids{prompt_ids[0]}-{prompt_ids[-1]}_rl_prefix_seed{seed}_top_p{top_p}_samples{num_samples_per_prompt}_resample{max_rl_resample}_p{p}_keep{num_prefix_keep}_first_only.json"

        else:
            filename = f"rl_resample_results/Qwen2.5-32B/{date}/ids{prompt_ids[0]}-{prompt_ids[-1]}_seed{seed}_top_p{top_p}_samples{num_samples_per_prompt}_resample{max_rl_resample}_p{p}_keep{num_prefix_keep}_first_only.json"
            filename_responses = f"rl_resample_results/Qwen2.5-32B/{date}/responses_id{prompt_ids[0]}-{prompt_ids[-1]}_seed{seed}_top_p{top_p}_samples{num_samples_per_prompt}_resample{max_rl_resample}_p{p}_keep{num_prefix_keep}_first_only.json"
            filename_replace = f"rl_resample_results/Qwen2.5-32B/{date}/replace_ids{prompt_ids[0]}-{prompt_ids[-1]}_seed{seed}_top_p{top_p}_samples{num_samples_per_prompt}_resample{max_rl_resample}_p{p}_keep{num_prefix_keep}_first_only.json"
        write_json(results, filename)
        #filename_responses = f"rl_resample_results/Qwen2.5-32B/june23/responses_rl_prefix_id{prompt_ids[0]}-{prompt_ids[-1]}_seed{seed}_top_p{top_p}_samples{num_samples_per_prompt}_resample{max_rl_resample}_p{p}_keep{num_prefix_keep}_first_only.json"
        write_json(all_responses, filename_responses)
        write_json(stored_replace, filename_replace)
        

    return results, all_responses



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RL Resampling Experiment")
    parser.add_argument("--max_rl_resample", type=int, default=60, help="Maximum number of RL resampling iterations")
    parser.add_argument("--p", type=float, default=0.03, help="Top entropy percentage for selecting high-entropy tokens")
    parser.add_argument("--num_prefix_keep", type=int, default=-1, help="Number of prefix tokens to keep in the first step")
    parser.add_argument("--include_prefix", type=int, default=1, help="Whether to include a RL prefix in the model generation (1 for True, 0 for False)")
    args = parser.parse_args()
    print(args)
    
    dataset = load_dataset("/mgfs/shared/Group_GY/hf_datasets/aime24", split="train")
    print("Dataset loaded:", dataset)
    # file_name = 'hf/datasets/aime2024/train-00000-of-00001.parquet'
    # dataset = load_dataset("parquet", data_files={'train': file_name})['train']
    

    # results, responses = run_batched_rl_resampling_experiment_vllm(
    #     dataset,
    #     base_model,
    #     rl_model,
    #     base_tokenizer,
    #     num_samples_per_prompt=32,
    #     p=0.05,
    #     max_rl_resample=30,
    #     max_new_tokens=4800,
    #     seed=1,
    #     num_prefix_keep=20,
    #     top_p=0.7,
    #     top_p_rl=0.7,
    #     batch_size=15,
    #     include_prefix=False
    # )


    results, responses = run_batched_rl_resampling_experiment_vllm(
        dataset,
        base_model,
        rl_model,
        base_tokenizer,
        num_samples_per_prompt=32,
        p=args.p,
        max_rl_resample=args.max_rl_resample,
        max_new_tokens=4800,
        seed=1,
        num_prefix_keep=args.num_prefix_keep,
        top_p=0.7,
        top_p_rl=0.7,
        batch_size=30,
        include_prefix=args.include_prefix == 1
    )

