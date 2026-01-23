@echo off
chcp 65001 >nul
echo ================================================================================
echo 训练32维增强模型
echo ================================================================================
echo.
echo 这将训练一个新的32维模型（包含技术指标）
echo 预计时间: 30-40分钟（50,000步）
echo.
echo 按任意键开始训练，或关闭窗口取消...
pause >nul

cd /d D:\PycharmProjects\RL-Stock
call conda activate rl-stock

python train_enhanced_32d.py

echo.
echo ================================================================================
echo 训练完成！
echo ================================================================================
echo.
echo 模型已保存到: models\ppo2_enhanced_32d.zip
echo.
echo 下一步:
echo   1. 运行测试: python quick_test.py
echo   2. 开始选股: python simple_selector.py
echo.
pause
