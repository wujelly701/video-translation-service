#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量视频翻译工具
支持目录批量处理、DeepSeek润色、断点续传、日志记录
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
import requests
import subprocess

# 解决Windows终端编码问题
if sys.platform == 'win32':
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 导入配置管理器
try:
    from config_manager import config

    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("警告: 未找到config_manager.py，将仅使用环境变量或命令行参数")


def setup_logger():
    """配置日志系统"""
    # 创建log目录
    log_dir = Path('log')
    log_dir.mkdir(exist_ok=True)

    # 生成日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'translation_{timestamp}.log'

    # 配置日志格式
    formatter = logging.Formatter('%(message)s')

    # 文件处理器
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return log_file


class VideoTranslator:
    """视频批量翻译器"""

    def __init__(self, service_url='http://127.0.0.1:50515', deepseek_key=None, use_polish=False):
        self.service_url = service_url

        # 优先级：命令行参数 > 环境变量 > config.ini
        if deepseek_key:
            self.deepseek_key = deepseek_key
        elif os.getenv('DEEPSEEK_API_KEY'):
            self.deepseek_key = os.getenv('DEEPSEEK_API_KEY')
        elif CONFIG_AVAILABLE:
            self.deepseek_key = config.deepseek_api_key
        else:
            self.deepseek_key = None

        self.use_polish = use_polish and self.deepseek_key

        # 统计信息
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': None,
            'end_time': None
        }

        # 支持的视频格式
        self.video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.webm', '.m4v'}

    def check_service(self):
        """检查翻译服务是否可用"""
        try:
            response = requests.get(f"{self.service_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('ready'):
                    logging.info("✓ 翻译服务正常运行")
                    return True
                else:
                    logging.error("× 翻译服务未就绪，请等待模型加载")
                    return False
            else:
                logging.error(f"× 翻译服务异常: {response.status_code}")
                return False
        except Exception as e:
            logging.error(f"× 无法连接到翻译服务: {e}")
            logging.error(f"  请确保服务正在运行: python server_optimized.py")
            return False

    def extract_audio(self, video_path, output_path):
        """从视频提取音频"""
        try:
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                '-y', output_path
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"  × 音频提取失败: {e}")
            return False
        except FileNotFoundError:
            logging.error("  × 找不到ffmpeg，请安装ffmpeg并添加到PATH")
            return False

    def transcribe(self, audio_path):
        """语音识别"""
        try:
            logging.info("  正在识别中，长视频可能需要数分钟...")

            with open(audio_path, 'rb') as f:
                files = {'audio': f}
                response = requests.post(
                    f"{self.service_url}/transcribe",
                    files=files,
                    timeout=3600  # 1小时超时
                )

            if response.status_code == 200:
                return response.json()
            else:
                logging.error(f"\n  × 识别失败: {response.status_code}")
                return None
        except requests.exceptions.Timeout:
            logging.error("\n  × 识别超时（视频可能太长）")
            return None
        except Exception as e:
            logging.error(f"\n  × 识别错误: {e}")
            return None

    def translate_text(self, text, source_lang='en', target_lang='zh'):
        """翻译文本"""
        try:
            response = requests.post(
                f"{self.service_url}/translate",
                json={
                    'text': text,
                    'source_language': source_lang,
                    'target_language': target_lang
                },
                timeout=90
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('translated_text', '')
            else:
                return text
        except Exception as e:
            logging.warning(f"  ! 翻译错误: {e}")
            return text

    def polish_translation(self, text, source_lang='en', target_lang='zh'):
        """使用DeepSeek润色翻译"""
        if not self.use_polish:
            return text

        try:
            lang_names = {'en': '英语', 'zh': '中文', 'ja': '日语', 'ko': '韩语'}
            source_name = lang_names.get(source_lang, source_lang)
            target_name = lang_names.get(target_lang, target_lang)

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

            response = requests.post(
                'https://api.deepseek.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.deepseek_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'deepseek-chat',
                    'messages': [
                        {'role': 'system', 'content': f'你是专业的{target_name}翻译专家。'},
                        {'role': 'user', 'content': prompt}
                    ],
                    'temperature': 0.3,
                    'max_tokens': 500
                },
                timeout=90
            )

            if response.status_code == 200:
                result = response.json()
                polished = result['choices'][0]['message']['content'].strip()
                return polished.strip('"\'')
            else:
                return text
        except Exception as e:
            logging.warning(f"  ! 润色错误: {e}")
            return text

    def format_time(self, seconds):
        """格式化时间为SRT格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def generate_srt(self, segments, output_path, translation_only=False):
        """生成SRT字幕文件"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, seg in enumerate(segments, 1):
                    start = self.format_time(seg['start'])
                    end = self.format_time(seg['end'])
                    text = seg['text']
                    translated = seg.get('translated', text)

                    f.write(f"{i}\n")
                    f.write(f"{start} --> {end}\n")

                    if translation_only:
                        f.write(f"{translated}\n\n")
                    else:
                        f.write(f"{text}\n")
                        f.write(f"{translated}\n\n")

            return True
        except Exception as e:
            logging.error(f"  × 生成字幕失败: {e}")
            return False

    def translate_video(self, video_path, target_lang='zh', source_lang='auto',
                        translation_only=False, output_dir=None):
        """翻译单个视频"""
        video_path = Path(video_path)
        video_name = video_path.stem

        logging.info(f"\n{'=' * 70}")
        logging.info(f"处理: {video_path.name}")
        logging.info(f"{'=' * 70}")

        # 输出路径
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = video_path.parent

        srt_path = output_dir / f"{video_name}_{target_lang}.srt"

        # 检查是否已存在
        if srt_path.exists():
            logging.info(f"  跳过: 字幕文件已存在 - {srt_path.name}")
            self.stats['skipped'] += 1
            return True

        start_time = time.time()

        # 1. 提取音频
        logging.info("  [1/4] 提取音频...")
        audio_path = output_dir / f"{video_name}_temp.wav"

        if not self.extract_audio(str(video_path), str(audio_path)):
            self.stats['failed'] += 1
            return False
        logging.info(" ✓")

        # 2. 语音识别
        logging.info("  [2/4] 语音识别...")
        transcribe_start = time.time()
        result = self.transcribe(str(audio_path))

        # 删除临时音频
        try:
            audio_path.unlink()
        except:
            pass

        if not result or not result.get('success'):
            logging.info(" ×")
            self.stats['failed'] += 1
            return False

        segments = result.get('segments', [])
        detected_lang = result.get('language', source_lang)
        transcribe_time = time.time() - transcribe_start
        logging.info(f" ✓ ({len(segments)}段, {transcribe_time:.1f}秒)")

        # 3. 翻译
        polish_info = " (使用DeepSeek润色)" if self.use_polish else ""
        logging.info(f"  [3/4] 翻译 {len(segments)} 段{polish_info}...")
        translate_start = time.time()

        for i, seg in enumerate(segments, 1):
            # 翻译
            translated = self.translate_text(
                seg['text'],
                detected_lang if source_lang == 'auto' else source_lang,
                target_lang
            )

            # DeepSeek润色
            if self.use_polish:
                polished = self.polish_translation(
                    translated,
                    detected_lang if source_lang == 'auto' else source_lang,
                    target_lang
                )
                # 记录润色前后对比（只记录有变化的前3个示例）
                if polished != translated and i <= 3:
                    logging.info(f"    [{i}] 原译: {translated}")
                    logging.info(f"    [{i}] 润色: {polished}")
                translated = polished

            seg['translated'] = translated

            # 进度显示
            if i % 10 == 0 or i == len(segments):
                logging.info(f"    进度: {i}/{len(segments)}")

        translate_time = time.time() - translate_start
        polish_suffix = " (含DeepSeek润色)" if self.use_polish else ""
        logging.info(f"    进度: {len(segments)}/{len(segments)} ✓ ({translate_time:.1f}秒{polish_suffix})")

        # 4. 生成字幕
        logging.info("  [4/4] 生成字幕...")
        if not self.generate_srt(segments, str(srt_path), translation_only):
            self.stats['failed'] += 1
            return False
        logging.info(" ✓")

        total_time = time.time() - start_time

        logging.info(f"\n✓ 完成: {srt_path.name}")
        logging.info(f"  总耗时: {total_time:.1f}秒")
        logging.info(f"  语音识别: {transcribe_time:.1f}秒")
        logging.info(f"  文本翻译: {translate_time:.1f}秒{polish_suffix}")

        self.stats['success'] += 1
        return True

    def translate_directory(self, directory, target_lang='zh', source_lang='auto',
                            translation_only=False, recursive=False, output_dir=None):
        """批量翻译目录中的视频"""
        directory = Path(directory)

        if not directory.exists():
            logging.error(f"× 目录不存在: {directory}")
            return

        # 查找视频文件
        if recursive:
            video_files = []
            for ext in self.video_extensions:
                video_files.extend(directory.rglob(f"*{ext}"))
        else:
            video_files = []
            for ext in self.video_extensions:
                video_files.extend(directory.glob(f"*{ext}"))

        video_files = sorted(video_files)

        if not video_files:
            logging.error(f"× 未找到视频文件（支持格式: {', '.join(self.video_extensions)}）")
            return

        self.stats['total'] = len(video_files)
        self.stats['start_time'] = time.time()

        logging.info(f"\n{'=' * 70}")
        logging.info(f"批量翻译任务")
        logging.info(f"{'=' * 70}")
        logging.info(f"目录: {directory}")
        logging.info(f"视频数量: {len(video_files)}")
        logging.info(f"目标语言: {target_lang}")
        logging.info(f"字幕模式: {'仅译文' if translation_only else '双语字幕'}")
        logging.info(f"DeepSeek润色: {'启用' if self.use_polish else '禁用'}")
        logging.info(f"{'=' * 70}")

        # 处理每个视频
        for i, video_file in enumerate(video_files, 1):
            logging.info(f"\n[{i}/{len(video_files)}]")

            try:
                self.translate_video(
                    video_file,
                    target_lang,
                    source_lang,
                    translation_only,
                    output_dir
                )
            except KeyboardInterrupt:
                logging.info("\n\n用户中断")
                break
            except Exception as e:
                logging.error(f"× 处理失败: {e}")
                self.stats['failed'] += 1
                continue

        self.stats['end_time'] = time.time()

        # 打印统计报告
        self.print_report()

    def print_report(self):
        """打印处理报告"""
        if self.stats['start_time'] is None:
            return

        total_time = self.stats['end_time'] - self.stats['start_time']

        logging.info(f"\n{'=' * 70}")
        logging.info("处理报告")
        logging.info(f"{'=' * 70}")
        logging.info(f"总视频数: {self.stats['total']}")
        logging.info(f"成功: {self.stats['success']}")
        logging.info(f"失败: {self.stats['failed']}")
        logging.info(f"跳过: {self.stats['skipped']}")
        logging.info(f"总耗时: {total_time / 60:.1f}分钟")

        if self.stats['success'] > 0:
            avg_time = total_time / self.stats['success']
            logging.info(f"平均每个: {avg_time:.1f}秒")

        logging.info(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description='批量视频翻译工具 - 支持DeepSeek润色和日志记录',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 翻译单个视频
  python batch_translate.py video.mp4 -t zh

  # 翻译目录中的所有视频
  python batch_translate.py videos/ -t zh

  # 使用DeepSeek润色
  python batch_translate.py videos/ -t zh --polish

  # 递归处理子目录
  python batch_translate.py videos/ -t zh --recursive
        """
    )

    parser.add_argument('input', help='视频文件或目录路径')
    parser.add_argument('-t', '--target', default='zh', help='目标语言（默认: zh）')
    parser.add_argument('-s', '--source', default='auto', help='源语言（默认: auto自动检测）')
    parser.add_argument('-o', '--output', help='输出目录（默认: 与视频同目录）')
    parser.add_argument('--translation-only', action='store_true', help='仅生成译文字幕（不含原文）')
    parser.add_argument('--recursive', '-r', action='store_true', help='递归处理子目录')
    parser.add_argument('--polish', action='store_true', help='使用DeepSeek润色翻译')
    parser.add_argument('--deepseek-key', help='DeepSeek API密钥')
    parser.add_argument('--service-url', default='http://127.0.0.1:50515',
                        help='翻译服务地址（默认: http://127.0.0.1:50515）')

    args = parser.parse_args()

    # 设置日志系统
    log_file = setup_logger()
    logging.info(f"日志文件: {log_file}")
    logging.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("")

    # 检查输入
    input_path = Path(args.input)
    if not input_path.exists():
        logging.error(f"× 路径不存在: {args.input}")
        return 1

    # 确定是否使用润色
    # 优先级：命令行参数 > config.ini默认设置
    use_polish = args.polish
    if not use_polish and CONFIG_AVAILABLE:
        use_polish = config.use_deepseek_polish

    # 创建翻译器
    translator = VideoTranslator(
        service_url=args.service_url,
        deepseek_key=args.deepseek_key,
        use_polish=use_polish
    )

    # 检查服务
    if not translator.check_service():
        return 1

    # 显示配置信息
    if use_polish or args.polish:
        if translator.deepseek_key:
            logging.info(f"✓ DeepSeek API密钥已配置")
            if translator.use_polish:
                logging.info(f"✓ DeepSeek润色已启用")
        else:
            logging.error("× DeepSeek API密钥未配置")

    # 检查DeepSeek配置
    if (args.polish or use_polish) and not translator.deepseek_key:
        logging.error("× 启用润色功能需要DeepSeek API密钥")
        logging.error("  方法1: 在 config.ini 中配置 [API] deepseek_api_key")
        logging.error("  方法2: 设置环境变量 set DEEPSEEK_API_KEY=your_key")
        logging.error("  方法3: 使用参数 --deepseek-key your_key")
        return 1

    logging.info("")

    # 开始处理
    try:
        if input_path.is_file():
            # 单个视频
            translator.stats['total'] = 1
            translator.stats['start_time'] = time.time()

            translator.translate_video(
                input_path,
                args.target,
                args.source,
                args.translation_only,
                args.output
            )

            translator.stats['end_time'] = time.time()
            translator.print_report()
        else:
            # 目录批量处理
            translator.translate_directory(
                input_path,
                args.target,
                args.source,
                args.translation_only,
                args.recursive,
                args.output
            )
    except KeyboardInterrupt:
        logging.info("\n\n用户中断")
        return 1

    logging.info(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"日志已保存到: {log_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())