import argparse
from preprocess import start_service
from main import Evaluator
from utils import make_config, get_open_port
import os,sys
import threading


NUM_WORKERS=10

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="模型名称")
    parser.add_argument("-w", "--work_dir", type=str, required=True, help="工作目录")
    parser.add_argument("--tag", type=str, required=True, help="标识标签")
    parser.add_argument("--turn_num", type=int, default=10, help="Maximum number of messages per character")
    parser.add_argument("--dataset", type=str, default='rpbench_character_subset', help="dataset file name define in dataset.yaml")
    args = parser.parse_args()
    max_messages_per_char = args.turn_num
    
    model,work_dir,tag = args.model,args.work_dir,args.tag
    # 加载 API 配置
    candidate_config = make_config("/home/xhai/rex/bench_base/configs/rpbench/api_config.yaml")
    assert model in candidate_config, f"{model} not found in candidate config"
    model_config = candidate_config[model]
    port = None
    stop_event = threading.Event()
    if model_config['source'] == 'local':
        port = get_open_port()
    else:
        port = model_config['endpoints']['api_port']
        
    thread = start_service(model_config,model, work_dir, tag, port,stop_event)
    try:
        evaluator = Evaluator(model, work_dir, args.dataset, tag, port, NUM_WORKERS, max_messages_per_char)
        evaluator.evaluate()
    except KeyboardInterrupt:
        thread.join()
        sys.exit(0) 
    finally:
        stop_event.set()
        import time
        time.sleep(2) # 等待主进程发信号给子进程
    