@echo off
chcp 936 >nul
title 红影焕新 - 老照片修复台
cd /d "%~dp0"
setlocal

set PORT=8601
if not "%~1"=="" set PORT=%~1
set HOSTARG=
if not "%~2"=="" set HOSTARG=--host %~2

echo ============================================================
echo            红影焕新 . 老照片修复台
echo ============================================================
echo.
echo 目录: %~dp0
echo 端口: %PORT%
echo.
echo 启动后不要关掉本窗口；关掉窗口就等于停止服务。
echo 浏览器打开窗口里打印的「本机访问」地址即可。
echo 想让手机/别的电脑也能访问，这样启动：
echo     启动服务.bat %PORT% 0.0.0.0
echo.

where python >nul 2>&1
if errorlevel 1 goto nopython

python heritage_server.py --port %PORT% %HOSTARG%
echo.
echo 服务已停止。
pause
goto :eof

:nopython
echo [错误] 没找到 python 命令。
echo 请安装 Python 3.10 以上版本，安装时勾选 Add Python to PATH。
echo.
pause
