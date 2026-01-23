@echo off
chcp 65001 >nul
cd /d D:\PycharmProjects\RL-Stock
call conda activate rl-stock
python simple_selector.py %*
pause
