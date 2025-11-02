#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
润色修复验证脚本
快速测试DeepSeek是否还会返回多行翻译
"""

import sys
import re


def check_srt_file(srt_path):
    """检查SRT文件中是否有多行翻译的问题"""
    print(f"\n🔍 检查字幕文件: {srt_path}")
    print("=" * 70)

    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 分割成字幕块
        blocks = content.strip().split('\n\n')

        issues = []
        total_blocks = 0

        for i, block in enumerate(blocks, 1):
            lines = block.strip().split('\n')

            # 字幕块至少有3行：序号、时间轴、文本
            if len(lines) < 3:
                continue

            total_blocks += 1

            # 检查是否是双语字幕（4行：序号、时间轴、原文、译文）
            if len(lines) == 4:
                translation = lines[3]
            # 或仅译文（3行：序号、时间轴、译文）
            elif len(lines) == 3:
                translation = lines[2]
            else:
                # 超过4行说明可能有问题
                issues.append({
                    'block': i,
                    'problem': '译文包含多行',
                    'lines': lines[2:],  # 从文本行开始
                    'count': len(lines) - 2
                })
                continue

            # 检查译文中是否包含换行（不应该有）
            if '\n' in translation:
                issues.append({
                    'block': i,
                    'problem': '译文内部包含换行符',
                    'lines': [translation],
                    'count': translation.count('\n') + 1
                })

        # 打印结果
        print(f"\n📊 检查结果:")
        print(f"  总字幕数: {total_blocks}")
        print(f"  问题数量: {len(issues)}")

        if issues:
            print(f"\n❌ 发现 {len(issues)} 个多行翻译问题：\n")

            for issue in issues[:5]:  # 只显示前5个
                print(f"字幕 #{issue['block']}:")
                print(f"  问题: {issue['problem']}")
                print(f"  行数: {issue['count']}")
                print(f"  内容:")
                for line in issue['lines']:
                    print(f"    {line}")
                print()

            if len(issues) > 5:
                print(f"  ... 还有 {len(issues) - 5} 个问题")

            print("\n💡 建议:")
            print("  1. 确认已使用修复后的 batch_translate.py")
            print("  2. 检查日志中的 '润色的结果' 输出")
            print("  3. 如果问题依然存在，请调整温度参数或提示词")

            return False
        else:
            print(f"\n✅ 太棒了！没有发现多行翻译问题")
            print(f"   所有 {total_blocks} 个字幕都是单行对应")
            return True

    except FileNotFoundError:
        print(f"❌ 文件不存在: {srt_path}")
        return False
    except Exception as e:
        print(f"❌ 检查出错: {e}")
        return False


def check_log_file(log_path):
    """检查日志中DeepSeek的返回结果"""
    print(f"\n🔍 检查日志文件: {log_path}")
    print("=" * 70)

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 查找所有润色结果
        pattern = r"润色的结果：\{.*?'content':\s*'([^']*)'.*?\}"
        matches = re.findall(pattern, content, re.DOTALL)

        if not matches:
            print("⚠️  未找到润色结果（可能日志中没有调试信息）")
            return None

        print(f"\n📊 找到 {len(matches)} 条润色记录\n")

        multiline_count = 0

        for i, content_text in enumerate(matches[:10], 1):  # 只检查前10条
            if '\\n' in content_text:
                multiline_count += 1
                print(f"❌ 第 {i} 条包含多行:")
                # 替换转义的换行符
                display = content_text.replace('\\n', '\n     ')
                print(f"   {display}\n")

        if multiline_count == 0:
            print(f"✅ 太棒了！前 {min(len(matches), 10)} 条都是单行翻译")
        else:
            print(f"\n⚠️  发现 {multiline_count}/{min(len(matches), 10)} 条是多行翻译")
            print("\n💡 这说明修复可能还不够完善，建议:")
            print("  1. 进一步优化提示词")
            print("  2. 降低temperature参数（如0.3）")
            print("  3. 在结果处理中增加更严格的过滤")

        return multiline_count == 0

    except FileNotFoundError:
        print(f"❌ 文件不存在: {log_path}")
        return None
    except Exception as e:
        print(f"❌ 检查出错: {e}")
        return None


def main():
    """主函数"""
    print("=" * 70)
    print("🔧 润色多行问题检查工具")
    print("=" * 70)

    if len(sys.argv) < 2:
        print("\n用法:")
        print("  检查SRT文件: python check_polish_fix.py video_zh.srt")
        print("  检查日志: python check_polish_fix.py log/translation_20231102_123456.log")
        print("  同时检查: python check_polish_fix.py video_zh.srt log/translation_xxx.log")
        return 1

    all_passed = True

    for file_path in sys.argv[1:]:
        if file_path.endswith('.srt'):
            result = check_srt_file(file_path)
            if result is False:
                all_passed = False
        elif file_path.endswith('.log'):
            result = check_log_file(file_path)
            if result is False:
                all_passed = False
        else:
            print(f"\n⚠️  不支持的文件类型: {file_path}")
            print("   支持: .srt (字幕文件), .log (日志文件)")

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 检查完成：所有测试通过！修复有效！")
    else:
        print("⚠️  检查完成：发现一些问题，请查看上面的详细信息")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
