import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time


# ==========================================
# 1. 模拟配置 (引入风险控制机制)
# ==========================================
class Config:
    PREDS_PATH = "../result/stacking_tuning_result/stacking_final_preds.csv"  # 建议用最强的 Stacking 结果
    TEST_DATA_PATH = "../data/X_test_final.csv"

    INITIAL_BANKROLL = 10000.0

    # --- 过滤器策略 ---
    PROB_THRESHOLD = 0.20  # 基础胜率门槛
    EV_THRESHOLD = 1.10  # 核心：期望值必须大于 1.10 (即认为有 10% 的利润空间)
    MIN_ODDS = 1.5
    MAX_ODDS = 15.0

    # --- 资金管理 (防爆仓核心) ---
    # 使用 Fractional Kelly (分数凯利)，现实中极少有人用全额凯利，因为模型误差会导致毁灭
    KELLY_FRACTION = 0.01  # 设为 1%：极其稳健，适合初次测试
    MAX_SINGLE_BET_PCT = 0.05  # 单笔投注绝不超过总本金的 5%
    STOP_LOSS_LIMIT = 0.20  # 账户剩余 20% 时强制停止模拟（保护余额）

    SLIPPAGE = 0.02  # 假设成交赔率比票面赔率低 2%


# ==========================================
# 2. 数据处理引擎
# ==========================================
def load_and_prepare():
    if not os.path.exists(Config.PREDS_PATH):
        # 兼容性处理：如果没 Stacking 就用 CatBoost
        Config.PREDS_PATH = "../result/cat_tuning_result/cat_tuned_preds.csv"

    df_preds = pd.read_csv(Config.PREDS_PATH)
    df_test = pd.read_csv(Config.TEST_DATA_PATH)

    # 核心字段对齐
    sim_df = pd.DataFrame(
        {
            "race_id": df_test["race_id"].values if "race_id" in df_test.columns else np.arange(len(df_test)),
            "odds": df_test["raw_win_odds"].values,
            "actual_rank": df_test["actual_rank"].values,
            "prob": df_preds["prob"].values,
        }
    )

    # 计算期望值 EV = 预测胜率 * 赔率
    sim_df["ev"] = sim_df["prob"] * sim_df["odds"]
    sim_df["is_winner"] = (sim_df["actual_rank"] == 1).astype(int)

    return sim_df


# ==========================================
# 3. 模拟核心引擎 (带风险熔断)
# ==========================================
def run_quant_simulation(df):
    bankroll = Config.INITIAL_BANKROLL
    history = [bankroll]
    trade_log = []

    # 按场次进行遍历
    for race_id, group in df.groupby("race_id"):
        # 止损熔断
        if bankroll < Config.INITIAL_BANKROLL * Config.STOP_LOSS_LIMIT:
            print(f"⚠️ 触及全局止损线 ({Config.STOP_LOSS_LIMIT*100}%)，停止交易。")
            break

        # 策略筛选：找出本场 EV 最高且符合条件的马
        eligible = group[
            (group["prob"] > Config.PROB_THRESHOLD)
            & (group["ev"] > Config.EV_THRESHOLD)
            & (group["odds"] >= Config.MIN_ODDS)
            & (group["odds"] <= Config.MAX_ODDS)
        ]

        if eligible.empty:
            continue

        # 选取最优目标
        target = eligible.loc[eligible["ev"].idxmax()]

        p = target["prob"]
        odds = target["odds"] * (1 - Config.SLIPPAGE)  # 考虑滑点后的有效赔率
        b = odds - 1

        # 凯利公式：f = (bp - q) / b
        kelly_f = (b * p - (1 - p)) / b

        if kelly_f > 0:
            # 投注金额 = 本金 * 凯利比例 * 缩减系数
            bet_amount = bankroll * kelly_f * Config.KELLY_FRACTION

            # 强制硬限额：单笔不超总本金 5%
            max_allowed = bankroll * Config.MAX_SINGLE_BET_PCT
            bet_amount = min(bet_amount, max_allowed)

            # 结算
            if target["is_winner"] == 1:
                profit = bet_amount * b
                bankroll += profit
                outcome = "WIN"
            else:
                bankroll -= bet_amount
                outcome = "LOSS"

            history.append(bankroll)
            trade_log.append(
                {"race_id": race_id, "bet": bet_amount, "odds": odds, "outcome": outcome, "balance": bankroll}
            )

    return history, pd.DataFrame(trade_log)


# ==========================================
# 4. 表现分析看板
# ==========================================
def analyze_performance(history, df_log):
    if df_log.empty:
        print("❌ 模拟期间没有产生任何有效投注。请调整阈值。")
        return

    final_val = history[-1]
    total_return = (final_val - Config.INITIAL_BANKROLL) / Config.INITIAL_BANKROLL * 100
    win_rate = (df_log["outcome"] == "WIN").mean()

    # 计算最大回撤
    h = np.array(history)
    drawdown = (np.maximum.accumulate(h) - h) / np.maximum.accumulate(h)
    max_dd = np.max(drawdown) * 100

    print("\n" + "═" * 45)
    print(f"💰 最终余额: ${final_val:,.2f}")
    print(f"📈 累计 ROI: {total_return:.2f}%")
    print(f"📉 最大回撤: {max_dd:.2f}%")
    print(f"🎯 胜率: {win_rate*100:.2f}% | 投注数: {len(df_log)}")
    print(f"⚖️ 平均注单占比: {(df_log['bet'].mean()/final_val)*100:.2f}%")
    print("═" * 45)

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(history, label="Portfolio Value", color="#27ae60", lw=2)
    plt.fill_between(range(len(history)), history, Config.INITIAL_BANKROLL, color="#27ae60", alpha=0.1)
    plt.axhline(Config.INITIAL_BANKROLL, color="red", ls="--", label="Initial")
    plt.title("Equity Curve (Fractional Kelly Strategy)", fontsize=14)
    plt.xlabel("Trade Count")
    plt.ylabel("Balance ($)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print("🚀 启动量化回测引擎...")
    # 确保函数名与第 33 行定义的名称完全一致
    data = load_and_prepare()
    hist, log = run_quant_simulation(data)
    analyze_performance(hist, log)
