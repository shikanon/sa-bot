import logging

class DebugLogger:
    """
    DebugLogger 类用于在调试模式下记录日志信息
    """
    def __init__(self, name):
        """
        初始化 DebugLogger 类

        参数:
            name (str): 类的名称，用于在日志记录中标识
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

    def debug(self, message):
        """
        在调试模式下记录日志信息

        参数:
            message (str): 要记录的日志信息
        """
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(message)

