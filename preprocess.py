import os
from utils import get_free_gpus
from logger import setup_logger
from halo import Halo
import subprocess
import time,requests
import threading

def start_service(model_config,model_name, work_dir, tag, port, stop_event):
    vllm_log_path = os.path.join(work_dir, tag, "vllm_log")
    print(f"开始启动模型服务: {model_name}")
    api_key = model_config['endpoints']['api_key']
    dtype = model_config['endpoints']['dtype']
    model_path = model_config['model_path']
    gpu_num = model_config.get('gpu_num', 1)
    # 获取gpu
    target_devices = get_free_gpus(gpu_num)
    # 启动服务
    service_thread = threading.Thread(target=start_backend_service, args=(model_path, api_key, port, dtype, vllm_log_path,target_devices,stop_event), daemon=True)
    service_thread.start()
    time.sleep(1)
    checking_backend_service(port)
    print(f"服务启动成功: {model_name}，端口: {port}")
    return service_thread

def start_backend_service(
    model_path, api_key, api_port, dtype, vllm_log_path,target_devices,stop_event
):

    # 设置 CUDA_VISIBLE_DEVICES 环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, target_devices))
    # 日志路径文件
    log_file_path = os.path.join(vllm_log_path, "vllm_backend_service.log")
    os.makedirs(vllm_log_path, exist_ok=True)  # 确保日志路径存在
    # 构造启动命令
    backend_command = [
        "vllm",
        "serve",
        model_path,
        "--port", str(api_port),
        "--dtype", dtype,
        "--api-key", api_key,
        "--tensor-parallel-size", str(len(target_devices))
    ]

    print(f"启动vllm的命令是: {' '.join(backend_command)}")
    print(f"启动vllm日志路径是: {log_file_path}")
    with open(log_file_path, "w") as log_file:
        process = subprocess.Popen(
            backend_command,
            stdout=log_file,
            stderr=log_file,
            shell=False
        )
    while not stop_event.is_set():
        time.sleep(1)  # 定期检查退出信号

    print("收到退出信号，终止子进程...")
    process.terminate()
    process.wait()

            
    
def checking_backend_service(port):
    spinner = None
    spinner = Halo(text='等待后端服务启动...', spinner='dots')
    spinner.start()
    health_url = f"http://localhost:{port}/health"
    while True:
        try:
            response = requests.get(health_url)
            if response.status_code == 200:
                spinner.succeed("模型vllm后端服务启动成功!") 
                break
        except requests.ConnectionError:
            pass
        time.sleep(5)
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="模型名称")
    parser.add_argument("-w", "--work_dir", type=str, required=True, help="工作目录")
    parser.add_argument("--tag", type=str, required=True, help="标识标签")
    args = parser.parse_args()
    start_service(args.model, args.work_dir, args.tag)
