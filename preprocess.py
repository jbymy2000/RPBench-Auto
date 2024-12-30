import os
from utils import make_config, get_open_port, start_backend_service
from logger import get_logger

logger = get_logger(__name__)

def start_service(model_name, work_dir, tag):
    logger.info(f"开始启动模型服务: {model_name}")
    
    # 加载 API 配置
    candidate_config = make_config("/home/xhai/rex/bench_base/configs/rpbench/api_config.yaml")
    assert model_name in candidate_config, f"{model_name} not found in candidate config"

    model_config = candidate_config[model_name]
    if model_config['source'] == 'local':
        vllm_log_path = os.path.join(work_dir, tag, "vllm_log")
        os.makedirs(vllm_log_path, exist_ok=True)

        if 'port' not in model_config['endpoints']:
            model_config['endpoints']['api_port'] = get_open_port()

        api_port = model_config['endpoints']['api_port']
        api_key = model_config['endpoints']['api_key']
        dtype = model_config['endpoints']['dtype']
        model_path = model_config['model_path']
        gpu_num = model_config.get('gpu_num', 1)

        # 启动服务
        start_backend_service(model_path, api_key, api_port, dtype, vllm_log_path, gpu_num)
        logger.info(f"服务启动成功: {model_name}，端口: {api_port}")
    else:
        logger.warning(f"模型 {model_name} 的服务未配置为本地模式")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="模型名称")
    parser.add_argument("-w", "--work_dir", type=str, required=True, help="工作目录")
    parser.add_argument("--tag", type=str, required=True, help="标识标签")
    args = parser.parse_args()

    start_service(args.model, args.work_dir, args.tag)
