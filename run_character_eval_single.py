import os
import jsonlines
from utils import make_config, chat_completion, extract_and_parse_json, get_free_gpus, get_open_port,start_backend_service
from tqdm.auto import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from prompt_template import get_template
from logger import get_logger
logger = get_logger(__name__)
dataset_config = make_config("/home/xhai/rex/bench_base/configs/rpbench/dataset_config.yaml")['datasets']

def eval_models_pairwise(model_1, model_2, work_dir, dataset_name, tag,language, max_workers=10, max_messages_per_char=10):
    model_template, judger_template = get_template(language)['model'],get_template(language)['judger']
    logger.info(f"开始对模型「{model_1}」进行benchmarking")
    dataset_path = get_datasource_path(dataset_name)
    eval_data = []
    win_lose_pairs = []
    eval_results = []
    ## 加载有关角色信息的数据
    with jsonlines.open(dataset_path) as reader:
        for idx, obj in enumerate(reader):
            eval_data.append((idx, obj)) 
    logger.info(f"Loaded {len(eval_data)} examples from {dataset_path}")

    ## 调用gpt4o作为评判模型
    judger_config = make_config("/home/xhai/rex/bench_base/configs/rpbench/judger_config.yaml")
    assert len(judger_config) == 1, "Judger config should have only one model"
    judger_model_name = list(judger_config.keys())[0]
    judger_model = judger_config[judger_model_name]
    logger.info(f"Judger model: `{judger_model_name}`")

    ## 其余的api候选调用
    candidate_config = make_config("/home/xhai/rex/bench_base/configs/rpbench/api_config.yaml")
    assert model_1 in candidate_config, f"{model_1} not found in candidate config"
    assert model_2 in candidate_config, f"{model_2} not found in candidate config"
    logger.info(f"Comparing `{model_1}` and `{model_2}`")
    model_config = candidate_config[model_1]
    if model_config['source'] == 'local':
        vllm_log_path = os.path.join(work_dir,tag, "vllm_log")
        if 'port' not in model_config['endpoints']:
            model_config['endpoints']['api_port'] = get_open_port()
        api_port = model_config['endpoints']['api_port']
        api_key = model_config['endpoints']['api_key']
        dtype = model_config['endpoints']['dtype']
        model_path = model_config['model_path']
        gpu_num = model_config.get('gpu_num', 1)
        start_backend_service(model_path,api_key,api_port,dtype,vllm_log_path,gpu_num)

    eval_results = []
    indexed_eval_results = []


    # Use ThreadPoolExecutor with controlled max_workers
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                process_single_character,
                character_data[1],
                model_config,
                judger_model,
                model_template,
                judger_template,
                max_messages_per_char,
            ): character_data[0]
            for character_data in eval_data
        }

        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx)):
            idx = future_to_idx[future]
            try:
                result = future.result()
                indexed_eval_results.append((idx, result))
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error processing data: {e}")
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



def process_single_character(
    character_data,
    model_config,
    judger_model,
    model_template,
    judger_template,
    max_messages_per_char=5,
):
    try:
        npc_profile = character_data["npc_profile"]
        conversation = character_data["conversation"]
        background = character_data["background"]
        greeting = "\n".join(conversation[0]["sentences"])

        candidate_messages = [
            {
                "role": "system",
                "content": model_template.substitute(background=background, **npc_profile),
            },
            {"role": "assistant", "content": greeting},
        ]

        judger_messages = [
            {"role": "system", "content": judger_template.substitute(npc_profile)},
            {"role": "user", "content": greeting},
        ]

        eval_results = []

        # 初始评判模型的响应
        judger_response = chat_completion_judger(judger_model, judger_messages)
        parsed_judger_response = extract_and_parse_json(judger_response)
        judger_messages.append({"role": "assistant", "content": judger_response})

        for _ in range(max_messages_per_char):
            user_input = parsed_judger_response["next_round_user_speaks"]
            candidate_messages.append({"role": "user", "content": user_input})

            # 调用候选模型获取响应
            model_a_response = chat_completion(model_config, candidate_messages)

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

            # 更新对话历史
            judger_messages.append({"role": "assistant", "content": judger_response})
            candidate_messages.append(
                {"role": "assistant", "content": model_a_response}
            )
            if _ == max_messages_per_char - 1:
                eval_results.append(eval_result)

        return eval_results
    except Exception as e:
        logger.error(f"Error processing character data: {e}")
        traceback.print_exc()
        raise


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
            logger.error(f"Error parsing response: {e}")
            raise



def get_datasource_path(dataset_name):
    if dataset_name not in dataset_config or 'path' not in dataset_config[dataset_name]:
        raise ValueError(f"Dataset `{dataset_name}` not found")
    return dataset_config[dataset_name]['path']
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name to evaluate")
    parser.add_argument("-w", "--work_dir", type=str, required=True, help="Directory to save evaluation results")
    parser.add_argument("--tag", type=str, required=True, help="Tag for the evaluation run")
    parser.add_argument("--turn_num", type=int, default=10, help="Maximum number of messages per character")
    parser.add_argument("--dataset", type=str, default='rpbench_character_subset', help="dataset file name define in dataset.yaml")
    parser.add_argument("--lang", type=str, default="zh")
    args = parser.parse_args()
    max_messages_per_char = args.turn_num
    eval_models_pairwise(args.model, args.model, args.work_dir, args.dataset, args.tag,args.lang, max_messages_per_char)
