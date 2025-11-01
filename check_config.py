#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置管理模块
从config.ini读取配置，支持环境变量覆盖
"""

import os
import configparser
from pathlib import Path


class Config:
    """配置管理器"""

    def __init__(self, config_file='config.ini'):
        self.config_file = Path(config_file)
        self.config = configparser.ConfigParser()

        # 尝试加载配置文件
        if self.config_file.exists():
            self.config.read(self.config_file, encoding='utf-8')
        else:
            print(f"警告: 配置文件不存在 {config_file}")
            print(f"请复制 config.template.ini 为 config.ini 并填写配置")

    def get(self, section, key, default=None):
        """获取配置值，支持环境变量覆盖"""
        # 环境变量名格式: SECTION_KEY (大写)
        env_key = f"{section.upper()}_{key.upper()}"

        # 优先使用环境变量
        env_value = os.getenv(env_key)
        if env_value is not None:
            return env_value

        # 然后使用配置文件
        try:
            return self.config.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default

    def get_bool(self, section, key, default=False):
        """获取布尔配置"""
        value = self.get(section, key, str(default))
        return value.lower() in ('true', '1', 'yes', 'on')

    def get_int(self, section, key, default=0):
        """获取整数配置"""
        value = self.get(section, key, str(default))
        try:
            return int(value)
        except ValueError:
            return default

    # API配置
    @property
    def deepseek_api_key(self):
        """DeepSeek API密钥"""
        return self.get('API', 'deepseek_api_key', os.getenv('DEEPSEEK_API_KEY', ''))

    # 服务配置
    @property
    def service_host(self):
        return self.get('Service', 'host', '127.0.0.1')

    @property
    def service_port(self):
        return self.get_int('Service', 'port', 50515)

    # 模型配置
    @property
    def asr_model_size(self):
        return self.get('Models', 'asr_model_size', 'medium')

    @property
    def translation_model(self):
        return self.get('Models', 'translation_model', 'facebook/nllb-200-distilled-1.3B')

    @property
    def use_gpu(self):
        return self.get_bool('Models', 'use_gpu', True)

    @property
    def beam_size(self):
        return self.get_int('Models', 'beam_size', 3)

    # 翻译配置
    @property
    def default_target_language(self):
        return self.get('Translation', 'default_target_language', 'zh')

    @property
    def use_deepseek_polish(self):
        return self.get_bool('Translation', 'use_deepseek_polish', False)


# 全局配置实例
config = Config()

if __name__ == '__main__':
    # 测试配置
    print("配置测试：")
    print(f"DeepSeek API密钥: {'已设置' if config.deepseek_api_key else '未设置'}")
    print(f"服务地址: {config.service_host}:{config.service_port}")
    print(f"ASR模型: {config.asr_model_size}")
    print(f"翻译模型: {config.translation_model}")
    print(f"使用GPU: {config.use_gpu}")
    print(f"beam_size: {config.beam_size}")
    print(f"默认目标语言: {config.default_target_language}")
    print(f"使用DeepSeek润色: {config.use_deepseek_polish}")