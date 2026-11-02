@echo off
rem Ë«»÷Æô¶¯ÎÄÔ¨£»¼Ó -Stop Í£Ö¹
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0wenyuan.ps1" %*
if errorlevel 1 pause
