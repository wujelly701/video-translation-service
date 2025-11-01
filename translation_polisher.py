#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
翻译后处理服务 - 使用DeepSeek API润色NLLB翻译
用法：在现有翻译服务基础上添加后处理层
"""

import requests
import json
import os


class TranslationPolisher:
    """翻译润色器 - 使用LLM改善机器翻译质量"""

    def __init__(self, api_key=None, provider='deepseek'):
        """
        初始化润色器

        Args:
            api_key: API密钥（如果不提供，从环境变量读取）
            provider: 'deepseek', 'openai', 'claude' 等
        """
        self.provider = provider

        if api_key is None:
            if provider == 'deepseek':
                api_key = os.getenv('DEEPSEEK_API_KEY')
            elif provider == 'openai':
                api_key = os.getenv('OPENAI_API_KEY')

        self.api_key = api_key

        # API端点配置
        self.endpoints = {
            'deepseek': 'https://api.deepseek.com/v1/chat/completions',
            'openai': 'https://api.openai.com/v1/chat/completions',
        }

    def polish(self, text, source_lang='en', target_lang='zh', context=None):
        """
        润色翻译文本

        Args:
            text: 待润色的翻译文本
            source_lang: 源语言代码
            target_lang: 目标语言代码
            context: 可选的上下文信息

        Returns:
            str: 润色后的文本
        """
        if self.provider == 'deepseek':
            return self._polish_with_deepseek(text, source_lang, target_lang, context)
        else:
            raise ValueError(f"不支持的provider: {self.provider}")

    def _polish_with_deepseek(self, text, source_lang, target_lang, context):
        """使用DeepSeek API润色"""

        # 语言名称映射
        lang_names = {
            'en': '英语', 'zh': '中文', 'ja': '日语',
            'ko': '韩语', 'de': '德语', 'fr': '法语',
            'es': '西班牙语', 'ru': '俄语'
        }

        source_name = lang_names.get(source_lang, source_lang)
        target_name = lang_names.get(target_lang, target_lang)

        # 构建提示词
        prompt = f"""你是专业的{target_name}母语者和翻译专家。请将以下{source_name}到{target_name}的机器翻译改得更地道、自然、符合母语者的表达习惯。

        原始翻译：
        {text}

        润色要求：
        1. 保持原意100%准确
        2. 使用最自然、最地道的{target_name}口语表达
        3. 避免直译腔，使用{target_name}母语者的说话方式
        4. 语气要自然流畅，符合对话场景
        5. 只返回润色后的翻译，不要任何解释或额外内容

        润色后的翻译："""

        if context:
            prompt = f"上下文：{context}\n\n" + prompt

        try:
            response = requests.post(
                self.endpoints['deepseek'],
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'deepseek-chat',
                    'messages': [
                        {
                            'role': 'system',
                            'content': f'你是一位专业的{target_name}翻译专家，擅长将机器翻译改写得更自然、地道。'
                        },
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ],
                    'temperature': 0.5,  # 较低温度保证稳定性
                    'max_tokens': 1000
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                polished_text = result['choices'][0]['message']['content'].strip()

                # 移除可能的引号
                polished_text = polished_text.strip('"\'')

                return polished_text
            else:
                print(f"DeepSeek API错误: {response.status_code} - {response.text}")
                return text  # 失败时返回原文

        except Exception as e:
            print(f"润色失败: {e}")
            return text  # 失败时返回原文


def test_polisher():
    """测试润色功能"""
    print("=" * 70)
    print("翻译润色测试")
    print("=" * 70)

    # 测试用例
    test_cases = [
        {
            'text': '我迫不及待地等着你看到美丽的场景.',
            'source_lang': 'en',
            'target_lang': 'zh'
        },
        {
            'text': '您好,世界.',
            'source_lang': 'en',
            'target_lang': 'zh'
        },
        {
            'text': '人工智能正在改变我们的生活和工作方式,',
            'source_lang': 'en',
            'target_lang': 'zh'
        }
    ]

    # 检查API密钥
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("\n⚠ 未设置DEEPSEEK_API_KEY环境变量")
        print("请运行: set DEEPSEEK_API_KEY=your_api_key")
        print("或者在代码中直接指定API密钥")
        return

    polisher = TranslationPolisher(api_key=api_key)

    for i, case in enumerate(test_cases, 1):
        print(f"\n测试 {i}/{len(test_cases)}:")
        print(f"原始翻译: {case['text']}")

        try:
            polished = polisher.polish(
                case['text'],
                case['source_lang'],
                case['target_lang']
            )
            print(f"润色后: {polished}")
            print("✓ 成功")
        except Exception as e:
            print(f"✗ 失败: {e}")

    print("\n" + "=" * 70)


# Flask集成示例
def add_polish_endpoint_to_flask_app(app):
    """
    将润色功能添加到Flask应用
    在现有server.py中调用此函数
    """
    from flask import request, jsonify

    polisher = TranslationPolisher()

    @app.route('/translate_with_polish', methods=['POST'])
    def translate_with_polish():
        """翻译+润色的组合接口"""
        data = request.json
        text = data.get('text')
        source_lang = data.get('source_language', 'en')
        target_lang = data.get('target_language', 'zh')

        # 先调用原有翻译接口
        translate_response = requests.post(
            'http://127.0.0.1:50515/translate',
            json={
                'text': text,
                'source_language': source_lang,
                'target_language': target_lang
            }
        )

        if translate_response.status_code != 200:
            return translate_response.json(), translate_response.status_code

        translate_result = translate_response.json()
        nllb_translation = translate_result.get('translated_text')

        # 润色
        try:
            polished_translation = polisher.polish(
                nllb_translation,
                source_lang,
                target_lang
            )

            return jsonify({
                'success': True,
                'original_translation': nllb_translation,
                'polished_translation': polished_translation,
                'translation': polished_translation  # 默认返回润色版
            })
        except Exception as e:
            # 润色失败，返回原始翻译
            return jsonify({
                'success': True,
                'original_translation': nllb_translation,
                'polished_translation': nllb_translation,
                'translation': nllb_translation,
                'polish_error': str(e)
            })


# 命令行测试工具
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='翻译润色工具')
    parser.add_argument('--test', action='store_true', help='运行测试')
    parser.add_argument('--text', type=str, help='待润色的文本')
    parser.add_argument('--api-key', type=str, help='DeepSeek API密钥')
    parser.add_argument('--source', default='en', help='源语言')
    parser.add_argument('--target', default='zh', help='目标语言')

    args = parser.parse_args()

    if args.test:
        test_polisher()
    elif args.text:
        api_key = args.api_key or os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            print("错误: 需要提供API密钥")
            exit(1)

        polisher = TranslationPolisher(api_key=api_key)
        result = polisher.polish(args.text, args.source, args.target)
        print(f"原文: {args.text}")
        print(f"润色后: {result}")
    else:
        print("使用 --help 查看帮助")