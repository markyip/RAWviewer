@echo off
REM Build RAWviewer full for Windows (semantic search + face scan; CUDA or DirectML).
REM Usage: build_windows_full.bat [cuda|directml]
cd /d "%~dp0..\..\.."
call scripts\Launch\bat\build_windows.bat %*
