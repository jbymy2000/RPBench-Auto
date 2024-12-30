import argparse
from preprocess import start_service
from main import eval_models_pairwise
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="模型名称")
    parser.add_argument("-w", "--work_dir", type=str, required=True, help="工作目录")
    parser.add_argument("--tag", type=str, required=True, help="标识标签")
    parser.add_argument("--turn_num", type=int, default=10, help="Maximum number of messages per character")
    parser.add_argument("--dataset", type=str, default='rpbench_character_subset', help="dataset file name define in dataset.yaml")
    parser.add_argument("--lang", type=str, default="zh")
    args = parser.parse_args()
    max_messages_per_char = args.turn_num
    start_service(args.model, args.work_dir, args.tag)
    eval_models_pairwise(args.model, args.model, args.work_dir, args.dataset, args.tag,args.lang, max_messages_per_char)
    