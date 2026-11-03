@echo off
chcp 936 >nul
title 红影焕新 - 开放防火墙端口
setlocal

echo ============================================================
echo            红影焕新 . 开放局域网访问端口
echo ============================================================
echo.
echo 手机/别的电脑打不开时，跑一次这个就行。
echo 需要管理员权限，弹出 UAC 请点「是」。
echo.

net session >nul 2>&1
if %errorlevel% neq 0 goto elevate

set PORT=8601
if not "%~1"=="" set PORT=%~1

echo 正在为 TCP %PORT% 添加入站放行规则...
netsh advfirewall firewall delete rule name="HeritageRestore %PORT%" >nul 2>&1
netsh advfirewall firewall add rule name="HeritageRestore %PORT%" dir=in action=allow protocol=TCP localport=%PORT% profile=any
if %errorlevel%==0 goto ok

echo.
echo [失败] 规则没加上，确认是不是以管理员身份运行。
goto done

:ok
echo.
echo [成功] 端口 %PORT% 已放行。
echo 同一 WiFi 下的手机，用服务窗口里「局域网访问」那一行地址就能打开。
goto done

:elevate
echo 正在申请管理员权限...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
exit /b

:done
echo.
pause
