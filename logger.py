import logging
import os


def setup_logger(log_dir, log_file_name, log_level=logging.INFO):
    """
    设置日志记录器，将日志输出到指定目录的文件和控制台，并增加文件名和行号。
    
    Args:
        log_dir (str): 日志存储的目录。
        log_file_name (str): 日志文件名。
        log_level (int): 日志级别（默认 logging.INFO）。
    """
    # 创建日志目录（如果不存在）
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 完整日志文件路径
    log_file_path = os.path.join(log_dir, log_file_name)

    # 创建日志记录器
    logger = logging.getLogger("rpbench")
    logger.setLevel(log_level)  # 设置全局日志级别

    # 日志格式，增加文件名和行号
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )

    # 文件处理器：将日志写入到文件
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(log_level)  # 设置文件日志级别
    file_handler.setFormatter(formatter)

    # 控制台处理器：将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)  # 设置控制台日志级别
    console_handler.setFormatter(formatter)

    # 将处理器添加到记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# 示例：如何使用
if __name__ == "__main__":
    # 指定日志目录和文件名
    log_directory = "./logs"
    log_file = "app_with_lineno.log"

    # 初始化日志记录器
    logger = setup_logger(log_directory, log_file, logging.DEBUG)

    # 测试日志输出
    def some_function():
        try:
            1 / 0  # 故意制造错误
        except Exception as e:
            logger.error("An error occurred!", exc_info=True)

    logger.debug("This is a DEBUG message.")
    logger.info("This is an INFO message.")
    logger.warning("This is a WARNING message.")
    logger.error("This is an ERROR message.")
    logger.critical("This is a CRITICAL message.")
    
    some_function()
