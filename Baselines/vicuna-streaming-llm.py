import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
# Code adapted from https://huggingface.co/kaiokendev/superhot-13b-8k-no-rlhf-test/blob/main/llama_rope_scaled_monkey_patch.py

from functools import partial
import transformers
import torch
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import argparse
from LEval_config import *
from tqdm import tqdm
from streaming_llm.kv_cache import StartRecentKVCache
from streaming_llm.pos_shift.modify_llama import enable_llama_pos_shift_attention
from torch.nn import CrossEntropyLoss

nlls = []
loss_fn = CrossEntropyLoss(reduction="none")

@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len):
    response = ""
    
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )

        now = len(generated_text) - 1
        if now > pos:
            # print(" ".join(generated_text[pos:now]), end=" ", flush=True)
            response += " ".join(generated_text[pos:now]) + " " 
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:
            break
    # print(" ".join(generated_text[pos:]), flush=True)
    response += " ".join(generated_text[pos:])

    return response

def compute_past_kv_cache(model, tokenizer, encodings, kv_cache):
    seq_len = encodings.input_ids.size(1)
    print(f"seq_len: {seq_len}")
    pbar = tqdm(range(0, seq_len - 1))

    past_key_values = None
    
    for idx in pbar:
        input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
        with torch.no_grad():
            outputs = model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits.view(-1, model.config.vocab_size)
            past_key_values = outputs.past_key_values
            label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
            neg_log_likelihood = loss_fn(logits, label)
            if kv_cache is not None:
                past_key_values = kv_cache(past_key_values)
        nlls.append(neg_log_likelihood)
        pbar.set_description(
            f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
        )
        print(neg_log_likelihood.item(), flush=True)

    return past_key_values
    
def main():
    # openai.api_base = "https://api.openai-sb.com/v1"
    start_idx = 0
    for file_name in key_data_pairs:
        fw = open(f'{file_name}', "w")
        data = key_data_pairs[file_name]
        header = (
            "A chat between a curious user and an artificial intelligence assistant."
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )

        sys_prompt = get_sys_prompt(args, file_name)
        for d in tqdm(data):
            document = d['input']
            cnt = 0
            # truncate documents
            while num_tokens_from_string(document, tokenizer) > max_length:
                if "code" not in file_name:
                    document = " ".join(document.split(" ")[:max_length - cnt]) # chunk the input len from right
                else:
                    document = " ".join(document.split(" ")[cnt - max_length:]) # chunk the input len from left
                cnt += 250

            instructions = d['instructions']
            outputs = d['outputs']

            for inst, out in zip(instructions, outputs):
                save_d = {}
                save_d['query'] = inst
                save_d['gt'] = out

                pre_prompt = ""
                post_prompt = ""
                
                if "gsm" in file_name or "codeU" in file_name:
                    context = document + "\n\n" + inst
                    message = sys_prompt + context

                    pre_prompt = sys_prompt + document
                    post_prompt = inst
                elif "topic" in file_name:
                    context = document + "\n\n" + inst
                    message = header + " USER: " + sys_prompt + context
                    message += " \nASSISTANT: "

                    pre_prompt = header + " USER: " + sys_prompt + document
                    post_prompt = inst + " \nASSISTANT: "
                elif args.metric == "exam_eval":
                    context = "Document is as follows. {document} \nQuestion: {inst} "
                    message = header + " USER: " + sys_prompt + context
                    message += " \nAnswer:"

                    pre_prompt = header + " USER: " + sys_prompt + f"Document is as follows. {document}"
                    post_prompt = f"Question: {inst}" + " \nAnswer:"
                elif "coursera" in file_name:
                    context = "Document is as follows. {document} Question: {inst} "
                    message = header + " USER: " + sys_prompt + context + "\n Please only give the correct options (e.g., A)."
                    message += " \nASSISTANT: "

                    pre_prompt = header + " USER: " + sys_prompt + f"Document is as follows. {document}"
                    post_prompt = f"Question: {inst} " + "\n Please only give the correct options (e.g., A)." + " \nASSISTANT: "
                else:
                    context = "Document is as follows. {document} \nInstruction: {inst} " + f"The suggested output length is around {len(out.split())} words. "
                    message = header + " USER: " + sys_prompt + context
                    message += " \nASSISTANT: My english answer is:"

                    pre_prompt = header + " USER: " + sys_prompt + f"Document is as follows. {document}"
                    post_prompt = f"Instruction: {inst} " + f"The suggested output length is around {len(out.split())} words. " + " \nASSISTANT: My english answer is:"
                try:
                    text_inputs = message.format(document=document, inst=inst)
                except:
                    text_inputs = message
                save_d['prompt'] = message.replace(document, "<long input>")
                
                # inputs = tokenizer(text_inputs, return_tensors="pt").to(device)
                # prompt_length = inputs.input_ids.size()[-1]
                # sample = model.generate(inputs.input_ids.to(model.device), do_sample=False, max_new_tokens=max_new_tokens, use_cache=True)[0]
                # output = tokenizer.decode(sample[prompt_length:], skip_special_tokens=True)

                pre_inputs = tokenizer(pre_prompt, return_tensors="pt").to(device)
                past_key_values = compute_past_kv_cache(model, tokenizer, pre_inputs, kv_cache)

                inputs = tokenizer(post_prompt, return_tensors="pt").to(device)
                space_needed = inputs.input_ids.shape[-1] + 100
                past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)
                
                output = greedy_generate(
                    model, tokenizer, inputs.input_ids, past_key_values, max_gen_len=100
                )

                save_d[f'{open_source_model}_pred'] = output
                save_d['evaluation'] = d['evaluation']

                if "sci_fi" in file_name:
                    text_inputs = inst.replace("based on the world described in the document.", "based on the real-world knowledge and facts up until your last training") + "\nAnswer:"
                    inputs = tokenizer(text_inputs, return_tensors="pt").to(device)
                    
                    # sample = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)                    
                    # prompt_length = inputs.input_ids.size()[-1]
                    # output = tokenizer.decode(sample[0][prompt_length:])

                    space_needed = inputs.input_ids.shape[-1] + 100
                    past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)
      
                    output = greedy_generate(
                        model, tokenizer, inputs.input_ids, past_key_values, max_gen_len=100
                    )
                    
                    save_d[f'{open_source_model}_pred'] += f" [fact: {output}]"

                if start_idx < 5:
                    print('document len', num_tokens_from_string(document, tokenizer))
                    print("----------------- [output] vs [ground truth] -----------------")
                    print('[output]:', save_d[f'{open_source_model}_pred'], "\n\n", '[ground truth]:', save_d['gt'])
                    start_idx += 1

                fw.write(json.dumps(save_d) + '\n')
        fw.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', choices=["llm_turbo_eval", "llm_gpt4_eval", "exam_eval", "ngram_eval", "human_eval"],
                        help='metric name from choices', required=True)
    parser.add_argument('--max_length', default="2k", help='max length of the input, e.g., 2k, 16k')
    parser.add_argument('--gpu', type=int, default=0)
    # set this if you do not want to use data from huggingface
    parser.add_argument('--task_path', type=str, default=None,
                        help='set this if you want test a specific task , example: LEval-data/Closed-ended-tasks/coursera.jsonl or LEval-data/Closed-ended-tasks/ ')
    # set this if you do not want to test a specific task
    parser.add_argument('--task_name', type=str, default=None,
                        help='set this if you want test a specific task from huggingface, example: coursera')
    parser.add_argument('--mc_tasks', action='store_true', help='set this if you want to test all multiple choice tasks')

    # for llama based model
    parser.add_argument('--scale', default='7b', choices=['7b', '13b'])
    args = parser.parse_args()

    # 7b / 13b

    max_length = k_to_number(args.max_length) - max_new_tokens

    model_path = f"lmsys/vicuna-{args.scale}-v1.3"
    open_source_model = f"vicuna1.3-{args.scale}-" + args.max_length

    data_save_path = f"Predictions/{args.metric}/{open_source_model}"
    input(f"Your prediction file will be saved to: {data_save_path}  , press enter to confirm...")

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path, force_download=True, resume_download=False, cache_dir="/scratch/kn22/cache")
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, cache_dir="/scratch/kn22/cache").to(device)# bfloat16 not supported on T5 GPUs
    model.eval()
    enable_llama_pos_shift_attention(model)
    
    kv_cache = StartRecentKVCache(
        start_size=4,
        recent_size=16,
        k_seq_dim=2,
        v_seq_dim=2,
        use_sampling=False,
    )

    key_data_pairs = {}
    build_key_data_pairs(args, key_data_pairs, data_save_path)
    sys.exit(main())