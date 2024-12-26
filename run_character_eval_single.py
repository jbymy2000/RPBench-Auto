import os
import json
import jsonlines
from utils import make_config, chat_completion, extract_and_parse_json
from string import Template
from tqdm.auto import tqdm
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

MAX_MESSAGES_PER_CHAR = 10
RPBENCH_PATH = "/home/xhai/bianjr/projects/RPBench-Auto/data/rpbench_chcracter_subset.jsonl"

TEMPLATE = Template(
    """$background

# NPC Profile:
## Name
$name_text

## Title
$title

## Description
$description

## Definition
$definition_text

## Long Definition
$long_definition_text
"""
)

JUDGER_TEMPLATE = Template(
    """# NPC Profile:
## Name
$name_text

## Title
$title

## Description
$description

## Definition
$definition_text

## Long Definition
$long_definition_text

You are an AI NPC system. You need to simulate a user and interact with AI NPC. For each round, You should give your response to AI NPC. It will be in a JSON format: {"winner": "model_a", "next_round_user_speaks": "YOUR RESPONSE AS THE SIMULATED USER", "decision_reason": "None"}.
"""
)


def chat_completion_judger(model, messages):
    while True:
        response = chat_completion(model, messages)
        try:
            parsed_response = extract_and_parse_json(response)
            if (
                "winner" in parsed_response
                and "next_round_user_speaks" in parsed_response
            ):
                return response
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Response: {response}")


def eval_models_pairwise(model_1, model_2, work_dir, tag,max_workers=10):

    eval_data = []
    win_lose_pairs = []
    eval_results = []
    ## 加载有关角色信息的数据
    with jsonlines.open(RPBENCH_PATH) as reader:
        for idx, obj in enumerate(reader):
            eval_data.append((idx, obj)) 
    print(f"Loaded {len(eval_data)} examples from {RPBENCH_PATH}")

    
    ## 调用gpt4o作为评判模型
    judger_config = make_config("/home/xhai/rex/bench_base/configs/rpbench/judger_config.yaml")
    assert len(judger_config) == 1, "Judger config should have only one model"
    judger_model_name = list(judger_config.keys())[0]
    judger_model = judger_config[judger_model_name]
    print(f"Judger model: `{judger_model_name}`")

    
    ## 其余的api候选调用
    candidate_config = make_config("/home/xhai/rex/bench_base/configs/rpbench/api_config.yaml")
    assert model_1 in candidate_config, f"{model_1} not found in candidate config"
    assert model_2 in candidate_config, f"{model_2} not found in candidate config"
    print(f"Comparing `{model_1}` and `{model_2}`")
    model_config = candidate_config[model_1]
    if model_config['source']=='local':
        start_backend_service(model_config)
    
    eval_results = []
    indexed_eval_results = []


    # Use ThreadPoolExecutor with controlled max_workers
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                process_single_character,
                character_data[1],
                model_1,
                candidate_config,
                judger_model,
                MAX_MESSAGES_PER_CHAR
            ): character_data[0]
            for character_data in eval_data
        }

        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx)):
            idx = future_to_idx[future]
            try:
                result = future.result()
                indexed_eval_results.append((idx, result))
            except Exception as e:
                print(f"Error processing data: {e}")
    indexed_eval_results.sort(key=lambda x: x[0])
    eval_results = [result for _, result in indexed_eval_results]
    output_dir = os.path.join(work_dir, tag)
    # 保存评估结果
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with jsonlines.open(
        f"{output_dir}/eval_{model_1}.jsonl", "a"
    ) as writer:
        writer.write_all(eval_results)
            
    return win_lose_pairs

def start_backend_service(model_config):
    port = model_config['endpoints']['api_port']
    api_key = model_config['endpoints']['api_key']
    dtype = model_config['endpoints']['dtype']
    model_path = model_config['model_path']

    # Start the backend service
    backend_command = [
        "vllm",
        "serve",
        model_path,
        "--port", str(port),
        "--dtype", dtype,
        "--api-key", api_key
    ]
    backend_command_str = " ".join(backend_command)
    print(backend_command_str)

    process = subprocess.Popen(backend_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    import time
    import traceback
    time.sleep(5)
    
    # Check if the process is still running
    if process.poll() is None:
        raise RuntimeError("Failed to start the backend service. Please check the logs for more details.")
        
def process_single_character(
    character_data,
    model_1,
    candidate_config,
    judger_model,
    max_messages_per_char=5,
):
    try:
        npc_profile = character_data["npc_profile"]
        conversation = character_data["conversation"]
        background = character_data["background"]
        greeting = "\n".join(conversation[0]["sentences"])
        #print("npc_profile",npc_profile)
        candidate_messages = [
            {
                "role": "system",
                "content": TEMPLATE.substitute(background=background, **npc_profile),
            },
            {"role": "assistant", "content": greeting},
        ]

        judger_messages = [
            {"role": "system", "content": JUDGER_TEMPLATE.substitute(npc_profile)},
            {"role": "user", "content": greeting},
        ]

        eval_results = []

        # 初始评判模型的响应
        judger_response = chat_completion_judger(judger_model, judger_messages)
        parsed_judger_response = extract_and_parse_json(judger_response)
        judger_messages.append({"role": "assistant", "content": judger_response})

        for _ in range(max_messages_per_char):
            # 设置模型名称
            model_a = model_1

            user_input = parsed_judger_response["next_round_user_speaks"]
            candidate_messages.append({"role": "user", "content": user_input})

            # 调用候选模型获取响应
            model_a_response = chat_completion(candidate_config[model_a], candidate_messages)

            # 将响应传递给评判模型
            judger_message_content = model_a_response
            judger_messages.append({"role": "user", "content": judger_message_content})
            judger_response = chat_completion_judger(judger_model, judger_messages)
            parsed_judger_response = extract_and_parse_json(judger_response)

            # 保存评估结果
            eval_result = {
                "candidate_messages": candidate_messages.copy(),
                "judger_messages": judger_messages.copy(),
                "judger_response": judger_response,
            }
            #print(eval_result)
            

            # 更新对话历史
            judger_messages.append({"role": "assistant", "content": judger_response})
            candidate_messages.append(
                {"role": "assistant", "content": model_a_response}
            )
            if _ == max_messages_per_char - 1:
                eval_results.append(eval_result)

        return eval_results
    except Exception as e:
        print(f"Error processing character data: {e}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name to evaluate")
    parser.add_argument("-w", "--work_dir", type=str, required=True, help="Directory to save evaluation results")
    parser.add_argument("--tag", type=str, required=True, help="Tag for the evaluation run")
    args = parser.parse_args()
    eval_models_pairwise(args.model, args.model, args.work_dir,args.tag)
