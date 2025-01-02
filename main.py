import os
import jsonlines
from utils import make_config, chat_completion, extract_and_parse_json, chat_completion_judger
from tqdm.auto import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from prompt_template import get_template
from logger import setup_logger
import logging


dataset_config = make_config("/home/xhai/rex/bench_base/configs/rpbench/dataset_config.yaml")['datasets']

class Evaluator:
    def __init__(self, model_name, work_dir, dataset_name,tag, port, max_workers, max_messages_per_char):
        verbose=logging.INFO
        self.model_name = model_name
        self.work_dir = work_dir
        self.tag = tag
        self.dataset_name = dataset_name
        self.dataset_path = self.get_datasource_path(self.dataset_name)
        self.language = self.get_datasource_lang(self.dataset_name)
        self.port = port
        self.max_workers = max_workers
        self.max_messages_per_char = max_messages_per_char
        self.model_template,self.judger_template = get_template(self.language)['model'],get_template(self.language)['judger']
        self.logger = setup_logger(os.path.join(self.work_dir,self.tag), f"eval_{self.model_name}.log",logging.DEBUG if verbose else logging.INFO)
    
    def evaluate(self):
        print(f"开始对模型「{self.model_name}」进行benchmarking")
        eval_data = []
        win_lose_pairs = []
        eval_results = []
        ## 加载有关角色信息的数据
        with jsonlines.open(self.dataset_path) as reader:
            for idx, obj in enumerate(reader):
                eval_data.append((idx, obj)) 
        self.logger.info(f"Loaded {len(eval_data)} examples from {self.dataset_path}")

        ## 调用gpt4o作为评判模型
        judger_config = make_config("/home/xhai/rex/bench_base/configs/rpbench/judger_config.yaml")
        assert len(judger_config) == 1, "Judger config should have only one model"
        judger_model_name = list(judger_config.keys())[0]
        judger_model = judger_config[judger_model_name]
        # self.logger.info(f"Judger model: `{judger_model_name}`")

        ## 其余的api候选调用
        candidate_config = make_config("/home/xhai/rex/bench_base/configs/rpbench/api_config.yaml")
        assert self.model_name in candidate_config, f"{self.model_name} not found in candidate config"
        # self.logger.info(f"Comparing `{self.model_name}` and `{self.model_name}`")
        model_config = candidate_config[self.model_name]
        if model_config['source'] == 'local':
            if 'port' not in model_config['endpoints']:
                model_config['endpoints']['api_port'] = self.port
                model_config['endpoints']['api_base'] = f'http://localhost:{self.port}/v1'

        eval_results = []
        indexed_eval_results = []


        # Use ThreadPoolExecutor with controlled max_workers
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self.process_single_character,
                    character_data[1],
                    model_config,
                    judger_model,
                    self.max_messages_per_char,
                ): character_data[0]
                for character_data in eval_data
            }

            for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx)):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    indexed_eval_results.append((idx, result))
                except Exception as e:
                    self.logger.error(f"Task failed at index {idx}, exiting program, Error processing data: {e}")
                    #raise RuntimeError(f"Task failed at index {idx}, exiting program.") from e
        indexed_eval_results.sort(key=lambda x: x[0])
        eval_results = [result for _, result in indexed_eval_results]
        output_dir = os.path.join(self.work_dir, self.tag)
        # 保存评估结果
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with jsonlines.open(
            f"{output_dir}/eval_{self.model_name}.jsonl", "w"
        ) as writer:
            writer.write_all(eval_results)
                
        return win_lose_pairs


    def process_single_character(
        self,
        character_data,
        model_config,
        judger_model,
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
                    "content": self.model_template.substitute(background=background, **npc_profile),
                },
                {"role": "assistant", "content": greeting},
            ]

            judger_messages = [
                {"role": "system", "content": self.judger_template.substitute(npc_profile)},
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
                # print('model_a_response',model_a_response)
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
            self.logger.error(f"Error processing character data: {e}")
            raise


    def get_datasource_path(self,dataset_name):
        if dataset_name not in dataset_config or 'path' not in dataset_config[dataset_name]:
            raise ValueError(f"Dataset `{dataset_name}` not found")
        return dataset_config[dataset_name]['path']

    def get_datasource_lang(self,dataset_name):
        if dataset_name not in dataset_config or 'lang' not in dataset_config[dataset_name]:
            return 'en'
        return dataset_config[dataset_name]['lang']
    
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str, required=True, help="Model name to evaluate")
#     parser.add_argument("-w", "--work_dir", type=str, required=True, help="Directory to save evaluation results")
#     parser.add_argument("--tag", type=str, required=True, help="Tag for the evaluation run")
#     parser.add_argument("--turn_num", type=int, default=10, help="Maximum number of messages per character")
#     parser.add_argument("--num_workers", type=int, default=10, help="Maximum number of threads")
#     parser.add_argument("--dataset", type=str, default='rpbench_character_subset', help="dataset file name define in dataset.yaml")
#     parser.add_argument("--port", type=str)
#     parser.add_argument("--verbose", action="store_true", help="Enable verbose output for debugging")
#     args = parser.parse_args()
#     max_messages_per_char = args.turn_num
#     evaluator = Evaluator(args.model, args.work_dir, args.dataset, args.tag,args.port, args.num_workers, max_messages_per_char)
#     evaluator.evaluate()
