#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量视频翻译工具 v3.0
支持：上下文翻译、DeepSeek润色、断点续传、日志记录
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
    log_dir = Path('log')
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'translation_{timestamp}.log'

    formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

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

        # 进度管理
        self.progress_dir = Path('.progress')
        self.progress_dir.mkdir(exist_ok=True)
        self.progress_file = None
        self.progress_data = {}

    def load_progress(self, task_name):
        """加载进度文件"""
        self.progress_file = self.progress_dir / f'{task_name}.json'

        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    self.progress_data = json.load(f)
                logging.info(f"✓ 加载进度文件: {self.progress_file.name}")

                completed = sum(1 for v in self.progress_data.values() if v.get('status') == 'completed')
                failed = sum(1 for v in self.progress_data.values() if v.get('status') == 'failed')
                if completed > 0 or failed > 0:
                    logging.info(f"  已完成: {completed}, 已失败: {failed}")
            except:
                self.progress_data = {}
        else:
            self.progress_data = {}

    def save_progress(self):
        """保存进度"""
        if self.progress_file:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress_data, f, ensure_ascii=False, indent=2)

    def update_video_status(self, video_name, status, **kwargs):
        """更新视频状态"""
        self.progress_data[video_name] = {
            'status': status,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **kwargs
        }
        self.save_progress()

    def should_skip_video(self, video_path, srt_path):
        """检查是否应该跳过该视频"""
        video_name = video_path.name

        # 检查字幕文件是否存在
        if srt_path.exists():
            return True, '字幕文件已存在'

        if video_name not in self.progress_data:
            return False, None

        status = self.progress_data[video_name].get('status')

        if status == 'completed':
            # 已标记完成但文件不存在，重新处理
            return False, '上次完成但文件缺失，重新处理'
        elif status == 'processing':
            return False, '上次未完成，重新处理'
        elif status == 'failed':
            retry_count = self.progress_data[video_name].get('retry_count', 0)
            if retry_count >= 3:
                return True, f'已失败{retry_count}次，跳过'
            else:
                return False, f'重试第{retry_count + 1}次'

        return False, None

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

            subprocess.run(
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
            with open(audio_path, 'rb') as f:
                files = {'audio': f}
                response = requests.post(
                    f"{self.service_url}/transcribe",
                    files=files,
                    timeout=3600  # 1小时，支持3小时视频
                )

            if response.status_code == 200:
                return response.json()
            else:
                logging.error(f"\n  × 识别失败: {response.status_code}")
                return None
        except requests.exceptions.Timeout:
            logging.error("\n  × 识别超时（视频太长，超过1小时处理时间）")
            return None
        except Exception as e:
            logging.error(f"\n  × 识别错误: {e}")
            return None

    def translate_text(self, text, source_lang='en', target_lang='zh', max_retries=3):
        """翻译文本（带重试）"""
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.service_url}/translate",
                    json={
                        'text': text,
                        'source_language': source_lang,
                        'target_language': target_lang
                    },
                    timeout=90  # 1.5分钟
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get('translated_text', '')
                else:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return text
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    logging.warning(f"  ! 翻译超时，重试 {attempt + 1}/{max_retries}")
                    time.sleep(2)
                else:
                    logging.warning(f"  ! 翻译超时: {text[:30]}...")
                    return text
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.warning(f"  ! 翻译错误，重试 {attempt + 1}/{max_retries}: {e}")
                    time.sleep(2)
                else:
                    logging.warning(f"  ! 翻译失败: {e}")
                    return text
        return text

    def get_context_window(self, segments, index, window_size=2):
        """获取上下文窗口"""
        start = max(0, index - window_size)
        end = min(len(segments), index + window_size + 1)

        context_before = []
        for i in range(start, index):
            if 'translated' in segments[i]:
                context_before.append(segments[i]['translated'])

        context_after = []
        for i in range(index + 1, end):
            context_after.append(segments[i]['text'])

        return context_before, context_after

    def polish_translation_with_context(self, text, translated, context_before, context_after,
                                        source_lang='en', target_lang='zh', max_retries=3):
        """使用DeepSeek润色翻译（带上下文）"""
        if not self.use_polish:
            return translated

        lang_names = {'en': '英语', 'zh': '中文', 'ja': '日语', 'ko': '韩语'}
        source_name = lang_names.get(source_lang, source_lang)
        target_name = lang_names.get(target_lang, target_lang)

        # 构建带上下文的提示词
        context_str = ""
        if context_before:
            context_str += f"\n前文（已翻译）：\n" + "\n".join(f"- {c}" for c in context_before[-2:])

        if context_after:
            context_str += f"\n\n后文（原文）：\n" + "\n".join(f"- {c}" for c in context_after[:2])

        prompt = f"""你是专业的{target_name}影视字幕翻译专家。请结合上下文，将以下{source_name}对话翻译得更地道、自然。
{context_str}

当前句子：
原文：{text}
机器翻译：{translated}

润色要求：
1. 结合上下文理解对话情境和人物关系
2. 准确传达原意、语气和情感
3. 使用最自然地道的{target_name}口语表达
4. 避免书面语和直译腔
5. 保持与上下文的连贯性
6. 只返回润色后的翻译，不要任何解释

润色后："""

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    'https://api.deepseek.com/v1/chat/completions',
                    headers={
                        'Authorization': f'Bearer {self.deepseek_key}',
                        'Content-Type': 'application/json'
                    },
                    json={
                        'model': 'deepseek-chat',
                        'messages': [
                            {'role': 'system', 'content': f'你是专业的{target_name}影视字幕翻译专家。'},
                            {'role': 'user', 'content': prompt}
                        ],
                        'temperature': 0.5,
                        'max_tokens': 500
                    },
                    timeout=90  # 1.5分钟
                )

                if response.status_code == 200:
                    result = response.json()
                    polished = result['choices'][0]['message']['content'].strip()
                    return polished.strip('"\'')
                else:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return translated
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    logging.warning(f"  ! 润色超时，重试 {attempt + 1}/{max_retries}")
                    time.sleep(2)
                else:
                    logging.warning(f"  ! 润色超时，使用原译文")
                    return translated
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.warning(f"  ! 润色错误，重试 {attempt + 1}/{max_retries}: {e}")
                    time.sleep(2)
                else:
                    logging.warning(f"  ! 润色失败，使用原译文: {e}")
                    return translated
        return translated

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
        """翻译单个视频（带进度管理和上下文翻译）"""
        video_path = Path(video_path)
        video_name = video_path.name

        logging.info(f"\n{'=' * 70}")
        logging.info(f"处理: {video_path.name}")
        logging.info(f"{'=' * 70}")

        # 输出路径
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = video_path.parent

        srt_path = output_dir / f"{video_path.stem}_{target_lang}.srt"

        # 检查是否应该跳过
        should_skip, reason = self.should_skip_video(video_path, srt_path)
        if should_skip:
            logging.info(f"  跳过: {reason}")
            self.stats['skipped'] += 1
            return True
        elif reason:
            logging.info(f"  {reason}")

        # 标记为处理中
        self.update_video_status(video_name, 'processing')

        try:
            start_time = time.time()

            # 1. 提取音频
            logging.info("  [1/4] 提取音频...""")
            audio_path = output_dir / f"{video_path.stem}_temp.wav"

            if not self.extract_audio(str(video_path), str(audio_path)):
                raise Exception("音频提取失败")
            logging.info(" ✓")

            # 2. 语音识别
            logging.info("  [2/4] 语音识别（长视频可能需要数分钟）...""")
            transcribe_start = time.time()
            result = self.transcribe(str(audio_path))

            # 删除临时音频
            try:
                audio_path.unlink()
            except:
                pass

            if not result or not result.get('success'):
                raise Exception("语音识别失败")

            segments = result.get('segments', [])
            detected_lang = result.get('language', source_lang)
            transcribe_time = time.time() - transcribe_start
            logging.info(f" ✓ ({len(segments)}段, {transcribe_time:.1f}秒)")

            # 3. 翻译（分两步：先翻译，再润色）
            translate_start = time.time()

            # 步骤1：批量翻译
            logging.info(f"  [3/4] 翻译 {len(segments)} 段...")
            for i, seg in enumerate(segments, 1):
                translated = self.translate_text(
                    seg['text'],
                    detected_lang if source_lang == 'auto' else source_lang,
                    target_lang
                )
                seg['translated'] = translated

                if i % 10 == 0 or i == len(segments):
                    logging.info(f"    翻译进度: {i}/{len(segments)}")

            logging.info(f"    翻译进度: {len(segments)}/{len(segments)} ✓")

            # 步骤2：带上下文润色
            if self.use_polish:
                logging.info(f"  [4/5] DeepSeek润色（结合上下文）...")
                polish_count = 0

                for i, seg in enumerate(segments):
                    # 获取上下文
                    context_before, context_after = self.get_context_window(segments, i, window_size=2)

                    # 润色
                    polished = self.polish_translation_with_context(
                        seg['text'],
                        seg['translated'],
                        context_before,
                        context_after,
                        detected_lang if source_lang == 'auto' else source_lang,
                        target_lang
                    )

                    # 记录变化（前3个示例）
                    if polished != seg['translated'] and polish_count < 3:
                        if context_before:
                            logging.info(f"    上文: ...{context_before[-1]}")
                        logging.info(f"    [{i + 1}] 原译: {seg['translated']}")
                        logging.info(f"    [{i + 1}] 润色: {polished}")
                        if context_after:
                            logging.info(f"    下文: {context_after[0]}...")
                        polish_count += 1

                    seg['translated'] = polished

                    if (i + 1) % 10 == 0 or (i + 1) == len(segments):
                        logging.info(f"    润色进度: {i + 1}/{len(segments)}")

                logging.info(f"    润色进度: {len(segments)}/{len(segments)} ✓")

            translate_time = time.time() - translate_start
            polish_suffix = " (含DeepSeek上下文润色)" if self.use_polish else ""

            # 4. 生成字幕
            step_num = 5 if self.use_polish else 4
            logging.info(f"  [{step_num}/{step_num}] 生成字幕...""")
            if not self.generate_srt(segments, str(srt_path), translation_only):
                raise Exception("生成字幕失败")
            logging.info(" ✓")

            total_time = time.time() - start_time

            logging.info(f"\n✓ 完成: {srt_path.name}")
            logging.info(f"  总耗时: {total_time:.1f}秒")
            logging.info(f"  语音识别: {transcribe_time:.1f}秒")
            logging.info(f"  翻译+润色: {translate_time:.1f}秒{polish_suffix}")

            # 标记为已完成
            self.update_video_status(
                video_name,
                'completed',
                srt_file=str(srt_path.name),
                duration=total_time
            )

            self.stats['success'] += 1
            return True

        except Exception as e:
            logging.error(f"\n× 处理失败: {e}")

            # 更新失败状态
            retry_count = self.progress_data.get(video_name, {}).get('retry_count', 0)
            self.update_video_status(
                video_name,
                'failed',
                error=str(e),
                retry_count=retry_count + 1
            )

            self.stats['failed'] += 1
            return False

    def translate_directory(self, directory, target_lang='zh', source_lang='auto',
                            translation_only=False, recursive=False, output_dir=None):
        """批量翻译目录中的视频（带进度管理）"""
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

        # 生成任务名称
        task_name = directory.name.replace(' ', '_').replace('\\', '_').replace('/', '_')
        if not task_name:
            task_name = 'root'
        self.load_progress(task_name)

        self.stats['total'] = len(video_files)
        self.stats['start_time'] = time.time()

        logging.info(f"\n{'=' * 70}")
        logging.info(f"批量翻译任务: {task_name}")
        logging.info(f"{'=' * 70}")
        logging.info(f"目录: {directory}")
        logging.info(f"视频数量: {len(video_files)}")
        logging.info(f"目标语言: {target_lang}")
        logging.info(f"字幕模式: {'仅译文' if translation_only else '双语字幕'}")
        logging.info(f"DeepSeek润色: {'启用（上下文润色）' if self.use_polish else '禁用'}")
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
                logging.info("\n\n用户中断 - 进度已保存，下次运行将继续")
                break
            except Exception as e:
                logging.error(f"× 意外错误: {e}")
                continue

        self.stats['end_time'] = time.time()
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

    def show_progress(self, task_name):
        """显示进度"""
        self.load_progress(task_name)

        if not self.progress_data:
            logging.info("× 没有找到进度记录")
            return

        completed = [k for k, v in self.progress_data.items() if v.get('status') == 'completed']
        failed = [k for k, v in self.progress_data.items() if v.get('status') == 'failed']
        processing = [k for k, v in self.progress_data.items() if v.get('status') == 'processing']

        logging.info(f"\n{'=' * 70}")
        logging.info(f"进度报告: {task_name}")
        logging.info(f"{'=' * 70}")
        logging.info(f"已完成: {len(completed)}")
        logging.info(f"已失败: {len(failed)}")
        logging.info(f"处理中: {len(processing)}")
        logging.info(f"总计: {len(self.progress_data)}")
        logging.info(f"{'=' * 70}")

        if failed:
            logging.info("\n失败列表:")
            for video in failed[:10]:  # 只显示前10个
                error = self.progress_data[video].get('error', '未知错误')
                retry = self.progress_data[video].get('retry_count', 0)
                logging.info(f"  - {video}: {error} (重试{retry}次)")
            if len(failed) > 10:
                logging.info(f"  ... 还有 {len(failed) - 10} 个失败")

    def reset_progress(self, task_name):
        """重置进度"""
        progress_file = self.progress_dir / f'{task_name}.json'
        if progress_file.exists():
            progress_file.unlink()
            logging.info(f"✓ 已清除进度: {task_name}")
        else:
            logging.info(f"× 没有找到进度文件: {task_name}")


def main():
    parser = argparse.ArgumentParser(
        description='批量视频翻译工具 v3.0 - 上下文翻译、DeepSeek润色、断点续传',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 翻译单个视频
  python batch_translate.py video.mp4 -t zh

  # 批量翻译（自动断点续传）
  python batch_translate.py videos/ -t zh

  # 查看进度
  python batch_translate.py videos/ --show-progress

  # 清除进度重新开始
  python batch_translate.py videos/ -t zh --reset-progress
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
    parser.add_argument('--show-progress', action='store_true', help='显示当前进度')
    parser.add_argument('--reset-progress', action='store_true', help='清除进度记录，从头开始')

    args = parser.parse_args()

    # 检查输入
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"× 路径不存在: {args.input}")
        return 1

    # 生成任务名称
    if input_path.is_dir():
        task_name = input_path.name.replace(' ', '_').replace('\\', '_').replace('/', '_')
        if not task_name:
            task_name = 'root'
    else:
        task_name = input_path.parent.name.replace(' ', '_').replace('\\', '_').replace('/', '_')
        if not task_name:
            task_name = 'root'

    # 确定是否使用润色
    use_polish = args.polish
    if not use_polish and CONFIG_AVAILABLE:
        use_polish = config.use_deepseek_polish

    # 创建翻译器
    translator = VideoTranslator(
        service_url=args.service_url,
        deepseek_key=args.deepseek_key,
        use_polish=use_polish
    )

    # 处理进度命令
    if args.show_progress:
        translator.show_progress(task_name)
        return 0

    if args.reset_progress:
        translator.reset_progress(task_name)
        if not input_path.exists():
            return 0

    # 设置日志系统
    log_file = setup_logger()
    logging.info(f"日志文件: {log_file}")
    logging.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("")

    # 检查服务
    if not translator.check_service():
        return 1

    # 显示配置信息
    if use_polish or args.polish:
        if translator.deepseek_key:
            logging.info(f"✓ DeepSeek API密钥已配置")
            if translator.use_polish:
                logging.info(f"✓ DeepSeek上下文润色已启用")
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
            translator.load_progress(task_name)
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