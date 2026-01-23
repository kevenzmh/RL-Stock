@echo off
chcp 65001 >nul
echo ================================================================================
echo 多股票训练 - 32维增强模型
echo ================================================================================
echo.
echo 这将使用10只不同类型的股票训练模型：
echo   - 银行股（价值股）
echo   - 科技股（成长股）
echo   - 消费股
echo   - 医药股
echo   - 地产股
echo.
echo 优势：
echo   ✓ 更好的泛化能力
echo   ✓ 适应各种股票类型
echo   ✓ 更稳定的预测
echo.
echo 预计时间: 1-1.5小时（100,000步）
echo.
echo 按任意键开始训练，或关闭窗口取消...
pause >nul

cd /d D:\PycharmProjects\RL-Stock
call conda activate rl-stock

python train_multi_stocks.py

echo.
echo ================================================================================
echo 训练完成！
echo ================================================================================
echo.
echo 模型已保存到: models\ppo2_enhanced_32d_multi.zip
echo.
pause
