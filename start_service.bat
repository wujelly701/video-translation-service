@echo off
REM AI视频翻译服务启动脚本

echo ========================================
echo    AI视频翻译服务
echo ========================================
echo.

REM 检查配置文件
if not exist "config.ini" (
    echo 错误: 配置文件不存在
    echo.
    echo 请执行以下步骤:
    echo 1. 复制 config.template.ini 为 config.ini
    echo 2. 编辑 config.ini 填入配置
    echo.
    pause
    exit /b 1
)

REM 激活虚拟环境（如果存在）
if exist ".venv311\Scripts\activate.bat" (
    echo 激活虚拟环境...
    call .venv311\Scripts\activate.bat
)

REM 显示配置
echo 读取配置文件: config.ini
echo.

REM 启动服务
echo 启动翻译服务...
python server_optimized.py

pause