import os
import json
import jsonlines
from utils import make_config, chat_completion, extract_and_parse_json
from string import Template
from tqdm.auto import tqdm
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


MAX_MESSAGES_PER_CHAR = 10
RPBENCH_PATH = "/home/xhai/bianjr/projects/RPBench-Auto/data/rpbench_chcracter_subset.jsonl"
SAVE_PATH = "/home/xhai/bianjr/projects/RPBench-Auto/test_results/subset_results/方案三对比实验1"

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
        except:
            pass


def eval_models_pairwise(model_1, model_2, max_workers=10):

    eval_data = []
    win_lose_pairs = []
    eval_results = []
    ## 加载有关角色信息的数据
    with jsonlines.open(RPBENCH_PATH) as reader:
        for idx, obj in enumerate(reader):
            eval_data.append((idx, obj)) 
    print(f"Loaded {len(eval_data)} examples from {RPBENCH_PATH}")

    
    ## 调用gpt4o作为评判模型
    judger_config = make_config("/home/xhai/bianjr/projects/RP_bench/config/judger_config.yaml")
    assert len(judger_config) == 1, "Judger config should have only one model"
    judger_model_name = list(judger_config.keys())[0]
    judger_model = judger_config[judger_model_name]
    print(f"Judger model: `{judger_model_name}`")

    
    ## 其余的api候选调用
    candidate_config = make_config("/home/xhai/bianjr/projects/RP_bench/config/api_config.yaml")
    assert model_1 in candidate_config, f"{model_1} not found in candidate config"
    assert model_2 in candidate_config, f"{model_2} not found in candidate config"
    print(f"Comparing `{model_1}` and `{model_2}`")

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

    # 保存评估结果
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    with jsonlines.open(
        f"{SAVE_PATH}/eval_{model_1}.jsonl", "a"
    ) as writer:
        writer.write_all(eval_results)
            
    return win_lose_pairs

def process_single_character(
    character_data,
    model_1,
    candidate_config,
    judger_model,
    max_messages_per_char=5,
):
    """print(
    json.dumps(
        {
            "message": "haha",
            "character_data": character_data,
            "model_1": model_1,
            "candidate_config": candidate_config,
            "judger_model": judger_model,
            "max_messages_per_char": max_messages_per_char,
        },
        indent=4,  # 美化输出，4 个空格缩进
        default=str  # 确保非 JSON 原生类型（如对象）可以序列化为字符串
    )
    )"""
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_1", type=str, default="方案三")
    parser.add_argument("--model_2", type=str, default="方案三")
    args = parser.parse_args()
    eval_models_pairwise(args.model_1, args.model_2)
