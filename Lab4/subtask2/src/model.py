import json
from tqdm import tqdm
# from vllm import LLM, SamplingParams  # No longer needed for API calls
# from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM # No longer needed for API calls
# import torch # No longer needed for API calls
import os
import requests


def get_pred_from_api(prompt: str) -> str:
    """
    Calls the Qwen-max API to get a prediction for the given prompt.

    This function reads the API key from the environment variable 'API_KEY'.
    Make sure to set this variable in your terminal before running the script:
    export API_KEY='your_actual_api_key'

    Args:
        prompt: The complete prompt string to send to the model.

    Returns:
        The translated text string from the API response, or an error message.
    """
    # 1. Get API Key from environment variable
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the 'API_KEY' environment variable.")

    # 2. Set up the request headers and URL
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    # 3. Construct the request body
    body = {
        "model": "qwen-max-2025-01-25",
        "input": {
            "prompt": prompt
        },
        "parameters": {
            "result_format": "text"
        }
    }

    # 4. Make the API call
    try:
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # 5. Parse the response and return the result
        result_json = response.json()
        return result_json.get("output", {}).get("text", "").strip()

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while calling the API: {e}")
        return f"API_ERROR: {e}"


# """
# =======================================================================================
# The original code for local model inference is commented out below for your reference.
# =======================================================================================
#
# def load_model(model_name, model_path, n_gpu=1, use_vllm=True):
#     print("loading model...")
#     if use_vllm:
#         llm = LLM(model=model_path, trust_remote_code=True, tensor_parallel_size=n_gpu, max_model_len=8192)
#         print("loaded!")
#         return llm
#     else:
#         tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#         llm = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
#         print("loaded!")
#         return llm, tokenizer
#
# def get_pred(llm, sampling_params, prompt):
#     # print("prompt:", prompt)
#     outputs = llm.generate(prompt, sampling_params)
#     result = outputs[0].outputs[0].text.strip().split('\n')[0]
#     result = result.split("<|endoftext|>")[0].strip()
#     result = result.split("<|im_end|>")[0].strip()
#
#     # print("result:", result)
#     return result
#
# def get_pred_no_vllm(llm, tokenizer, prompt, args):
#     inputs = tokenizer(prompt, return_tensors="pt")
#     input_len = len(inputs['input_ids'][0])
#     # print("input_len", input_len)
#     inputs = inputs.to('cuda')
#     preds = llm.generate(
#         **inputs,
#         do_sample=args.do_sample,
#         top_k=args.top_k,
#         top_p=args.top_p,
#         temperature=args.temperature,
#         num_beams=args.num_beams,
#         repetition_penalty=args.repetition_penalty,
#         max_new_tokens=args.max_new_tokens,
#     )
#     pred = tokenizer.decode(preds[0][input_len:], skip_special_tokens=True)
#     output = pred
#     result = output.strip().split('\n')[0]
#     return result
# """


if __name__ == '__main__':
    # This is a test block to demonstrate how to use the new function.
    # You need to set your API key in your terminal first:
    # export API_KEY='sk-your-key-here'

    print("Testing the API call function...")
    test_prompt = "请将下面的壮语句子翻译成汉语：Gou dwg Vangz Gangh.\n## 在上面的句子中，壮语词语“gou”在汉语中可能的翻译是“我”；\n壮语词语“dwg”在汉语中可能的翻译是“是”；\n壮语词语“vangz”在汉语中可能的翻译是“王”；\n壮语词语“gangh”在汉语中可能的翻译是“刚”；\n## 所以，该壮语句子完整的汉语翻译是："

    try:
        translation = get_pred_from_api(test_prompt)
        print("\n--- Test Prompt ---")
        print(test_prompt)
        print("\n--- API Response ---")
        print(translation)
    except ValueError as e:
        print(e)