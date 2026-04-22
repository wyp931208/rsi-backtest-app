import os
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import tushare as ts

from fpdf import FPDF
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# =========================
# 字体工具
# =========================
def get_font_paths():
    """
    返回可用的中文字体路径（regular, bold）
    优先使用你仓库中的 ttf 字体
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fonts_dir = os.path.join(base_dir, "fonts")

    regular_candidates = [
        os.path.join(fonts_dir, "NotoSansSC-6.ttf"),
        os.path.join(fonts_dir, "MSYH.TTC"),
        os.path.join(fonts_dir, "noto-sans-sc-regular.otf"),
    ]

    bold_candidates = [
        os.path.join(fonts_dir, "Noto-Sans-SC-Bold-2.ttf"),
        os.path.join(fonts_dir, "MSYH.TTC"),
        os.path.join(fonts_dir, "NotoSansSC-Bold.otf"),
    ]

    regular_path = next((p for p in regular_candidates if os.path.exists(p)), None)
    bold_path = next((p for p in bold_candidates if os.path.exists(p)), None)

    return regular_path, bold_path


# =========================
# matplotlib 中文显示设置
# =========================
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


# =========================
# 全局参数
# =========================
INITIAL_CAPITAL = 100000
MIN_COMMISSION = 5
RISK_FREE_RATE = 0.02


# =========================
# 工具函数
# =========================
def init_tushare(token, custom_url=None):
    pro = ts.pro_api(token)
    if custom_url:
        pro._DataApi__http_url = custom_url
    return pro


def is_fund_code(ts_code):
    return ts_code.startswith(("5", "1"))


def is_index_code(ts_code):
    return ts_code in ["000300.SH", "000001.SH", "399006.SZ", "399001.SZ"]


def get_sell_tax(ts_code):
    return 0.0 if is_fund_code(ts_code) else 0.001


def calc_buy_cost(amount):
    return max(amount * 0.0003, MIN_COMMISSION)


def calc_sell_cost(amount, ts_code):
    commission = max(amount * 0.0003, MIN_COMMISSION)
    tax = amount * get_sell_tax(ts_code)
    return commission + tax


def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.where(avg_loss != 0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi


def max_drawdown_from_nav(nav):
    rolling_max = nav.cummax()
    drawdown = nav / rolling_max - 1
    return drawdown.min()


def annualize_return(total_return, periods):
    if periods <= 0:
        return np.nan
    return (1 + total_return) ** (252 / periods) - 1


def build_date_index(data_dict, benchmark_df):
    if not data_dict or benchmark_df is None or benchmark_df.empty:
        return pd.to_datetime([])

    date_sets = [set(df["date"]) for df in data_dict.values() if df is not None and not df.empty]
    date_sets.append(set(benchmark_df["date"]))
    common_dates = sorted(set.intersection(*date_sets)) if date_sets else []
    return pd.to_datetime(common_dates)


def get_row_by_date(df, dt):
    row = df.loc[df["date"] == dt]
    if row.empty:
        return None
    return row.iloc[0]


# =========================
# 数据获取
# =========================
def fetch_asset_data(pro, ts_code, start_date, end_date):
    if is_index_code(ts_code):
        df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    elif is_fund_code(ts_code):
        df = pro.fund_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    else:
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

    if df is None or df.empty:
        return None

    df = df.sort_values("trade_date").reset_index(drop=True)
    df = df.rename(columns={"trade_date": "date", "vol": "volume"})
    df["date"] = pd.to_datetime(df["date"])

    keep_cols = ["date", "open", "high", "low", "close"]
    if "volume" in df.columns:
        keep_cols.append("volume")

    df = df[keep_cols].copy()

    if "volume" not in df.columns:
        df["volume"] = np.nan

    return df


def load_all_data(pro, security_list, benchmark, start_date, end_date, rsi_period=14, ma_period=20):
    data_dict = {}
    for code in security_list:
        with st.spinner(f"正在获取 {code} 数据..."):
            df = fetch_asset_data(pro, code, start_date, end_date)
            if df is not None and not df.empty:
                df["rsi"] = calc_rsi(df["close"], rsi_period)
                df["ma"] = df["close"].rolling(ma_period).mean()
                data_dict[code] = df

    benchmark_df = None
    with st.spinner(f"正在获取基准 {benchmark} 数据..."):
        benchmark_df = fetch_asset_data(pro, benchmark, start_date, end_date)
        if benchmark_df is not None and not benchmark_df.empty:
            benchmark_df["rsi"] = calc_rsi(benchmark_df["close"], rsi_period)
            benchmark_df["ma"] = benchmark_df["close"].rolling(ma_period).mean()

    return data_dict, benchmark_df


# =========================
# 信号判断
# =========================
def should_buy(df, i, params):
    if i < max(params.get("rsi_period", 14), params.get("ma_period", 20)):
        return False

    prev_rsi = df.iloc[i - 1]["rsi"]
    curr_rsi = df.iloc[i]["rsi"]
    current_price = df.iloc[i]["close"]
    ma_value = df.iloc[i]["ma"]

    if pd.isna(prev_rsi) or pd.isna(curr_rsi) or pd.isna(current_price):
        return False

    if params.get("use_ma_filter", True) and params.get("price_above_ma_buy", True):
        if pd.isna(ma_value) or current_price <= ma_value:
            return False

    if params.get("buy_signal_mode") == "cross_up":
        return prev_rsi < params.get("rsi_buy", 30) and curr_rsi >= params.get("rsi_buy", 30)
    elif params.get("buy_signal_mode") == "below":
        return curr_rsi < params.get("rsi_buy", 30)

    return False


def should_sell(df, i, avg_cost, params):
    if i < max(params.get("rsi_period", 14), params.get("ma_period", 20)):
        return False

    current_price = df.iloc[i]["close"]
    if pd.isna(current_price) or current_price <= 0:
        return False

    if avg_cost is not None and avg_cost > 0:
        ret = (current_price - avg_cost) / avg_cost
        if params.get("use_stop_loss", True) and ret <= -params.get("stop_loss_pct", 0.08):
            return True
        if params.get("use_take_profit", False) and ret >= params.get("take_profit_pct", 0.15):
            return True

    prev_rsi = df.iloc[i - 1]["rsi"]
    curr_rsi = df.iloc[i]["rsi"]
    ma_value = df.iloc[i]["ma"]

    if pd.isna(prev_rsi) or pd.isna(curr_rsi):
        return False

    if params.get("use_ma_filter", True) and params.get("price_below_ma_sell", False):
        if not pd.isna(ma_value) and current_price < ma_value:
            return True

    if params.get("sell_signal_mode") == "cross_down":
        return prev_rsi > params.get("rsi_sell", 70) and curr_rsi <= params.get("rsi_sell", 70)
    elif params.get("sell_signal_mode") == "above":
        return curr_rsi > params.get("rsi_sell", 70)

    return False


# =========================
# 组合回测
# =========================
def backtest_portfolio(data_dict, benchmark_df, params):
    dates = build_date_index(data_dict, benchmark_df)
    cash = INITIAL_CAPITAL
    positions = {}
    cost_basis = {}
    trade_records = []
    daily_records = []

    for dt in dates:
        # 1. 先卖出
        for code in list(positions.keys()):
            df = data_dict.get(code)
            if df is None:
                continue

            idx_list = df.index[df["date"] == dt].tolist()
            if not idx_list:
                continue

            i = idx_list[0]
            current_price = df.iloc[i]["close"]
            if pd.isna(current_price) or current_price <= 0:
                continue

            if should_sell(df, i, cost_basis.get(code), params):
                amount = positions[code] * current_price
                sell_cost = calc_sell_cost(amount, code)
                net_amount = amount - sell_cost
                cash += net_amount

                trade_records.append({
                    "date": dt,
                    "code": code,
                    "action": "SELL",
                    "price": current_price,
                    "rsi": df.iloc[i]["rsi"],
                    "shares": positions[code],
                    "amount": amount,
                    "cost": sell_cost,
                    "net_amount": net_amount
                })

                del positions[code]
                del cost_basis[code]

        # 2. 再买入
        available_slots = params.get("max_holdings", 3) - len(positions)
        if available_slots > 0:
            buy_candidates = [code for code, df in data_dict.items() if code not in positions]
            buy_list = []

            for code in buy_candidates:
                df = data_dict[code]
                idx_list = df.index[df["date"] == dt].tolist()
                if idx_list and should_buy(df, idx_list[0], params):
                    buy_list.append(code)

            buy_list = buy_list[:available_slots]

            if buy_list:
                total_value = cash
                for held_code, shares in positions.items():
                    p = get_row_by_date(data_dict[held_code], dt)
                    if p is not None and not pd.isna(p["close"]):
                        total_value += shares * p["close"]

                target_ratio = min(
                    1.0 / params.get("max_holdings", 3),
                    params.get("max_position_per_security", 0.4)
                )
                target_value = total_value * target_ratio

                for code in buy_list:
                    df = data_dict[code]
                    idx_list = df.index[df["date"] == dt].tolist()
                    if not idx_list:
                        continue
                    i = idx_list[0]

                    price_row = get_row_by_date(data_dict[code], dt)
                    if price_row is None or pd.isna(price_row["close"]):
                        continue

                    price = price_row["close"]
                    amount = min(target_value, cash)

                    if amount <= MIN_COMMISSION:
                        continue

                    buy_cost = calc_buy_cost(amount)
                    investable = amount - buy_cost
                    if investable <= 0:
                        continue

                    shares = investable / price
                    actual_amount = shares * price
                    actual_cost = calc_buy_cost(actual_amount)
                    total_deduction = actual_amount + actual_cost

                    if total_deduction > cash:
                        actual_amount = max(cash - MIN_COMMISSION, 0)
                        if actual_amount <= 0:
                            continue
                        actual_cost = calc_buy_cost(actual_amount)
                        actual_amount = cash - actual_cost
                        if actual_amount <= 0:
                            continue
                        shares = actual_amount / price
                        total_deduction = actual_amount + actual_cost

                    cash -= total_deduction
                    positions[code] = shares
                    cost_basis[code] = price

                    trade_records.append({
                        "date": dt,
                        "code": code,
                        "action": "BUY",
                        "price": price,
                        "rsi": df.iloc[i]["rsi"],
                        "shares": shares,
                        "amount": actual_amount,
                        "cost": actual_cost,
                        "net_amount": -total_deduction
                    })

        # 3. 每日记录
        total_equity = cash
        holding_details = []

        for code, shares in positions.items():
            p = get_row_by_date(data_dict[code], dt)
            if p is not None and not pd.isna(p["close"]):
                mv = shares * p["close"]
                total_equity += mv
                holding_details.append(f"{code}:{mv:.2f}")

        daily_records.append({
            "date": dt,
            "cash": cash,
            "holding_count": len(positions),
            "holdings": " | ".join(holding_details),
            "equity": total_equity
        })

    result_df = pd.DataFrame(daily_records)
    trade_df = pd.DataFrame(trade_records)

    if result_df.empty:
        return result_df, trade_df, {}

    result_df["strategy_nav"] = result_df["equity"] / result_df["equity"].iloc[0]
    result_df["strategy_return"] = result_df["strategy_nav"].pct_change().fillna(0)

    benchmark_nav_df = benchmark_df[["date", "close"]].copy().sort_values("date").reset_index(drop=True)
    benchmark_nav_df["benchmark_nav"] = benchmark_nav_df["close"] / benchmark_nav_df["close"].iloc[0]
    benchmark_nav_df["benchmark_return"] = benchmark_nav_df["benchmark_nav"].pct_change().fillna(0)

    merged_df = pd.merge(
        result_df,
        benchmark_nav_df[["date", "benchmark_nav", "benchmark_return"]],
        on="date",
        how="inner"
    )

    metrics = calculate_performance(merged_df)
    return merged_df, trade_df, metrics


# =========================
# 单标的回测
# =========================
def backtest_single_asset(df, benchmark_df, ts_code, params):
    dates = sorted(set(df["date"]).intersection(set(benchmark_df["date"])))
    dates = pd.to_datetime(dates)

    cash = INITIAL_CAPITAL
    shares = 0.0
    cost_basis = None
    trade_records = []
    daily_records = []

    for dt in dates:
        idx_list = df.index[df["date"] == dt].tolist()
        if not idx_list:
            continue

        i = idx_list[0]
        current_price = df.iloc[i]["close"]

        if pd.isna(current_price) or current_price <= 0:
            continue

        # 先卖
        if shares > 0 and should_sell(df, i, cost_basis, params):
            amount = shares * current_price
            sell_cost = calc_sell_cost(amount, ts_code)
            net_amount = amount - sell_cost
            cash += net_amount

            trade_records.append({
                "date": dt,
                "code": ts_code,
                "action": "SELL",
                "price": current_price,
                "rsi": df.iloc[i]["rsi"],
                "shares": shares,
                "amount": amount,
                "cost": sell_cost,
                "net_amount": net_amount
            })

            shares = 0.0
            cost_basis = None

        # 再买
        elif shares == 0 and should_buy(df, i, params):
            amount = cash
            if amount > MIN_COMMISSION:
                buy_cost = calc_buy_cost(amount)
                investable = amount - buy_cost

                if investable > 0:
                    shares = investable / current_price
                    actual_amount = shares * current_price
                    actual_cost = calc_buy_cost(actual_amount)
                    total_deduction = actual_amount + actual_cost

                    cash -= total_deduction
                    cost_basis = current_price

                    trade_records.append({
                        "date": dt,
                        "code": ts_code,
                        "action": "BUY",
                        "price": current_price,
                        "rsi": df.iloc[i]["rsi"],
                        "shares": shares,
                        "amount": actual_amount,
                        "cost": actual_cost,
                        "net_amount": -total_deduction
                    })

        total_equity = cash
        if shares > 0:
            total_equity += shares * current_price

        daily_records.append({
            "date": dt,
            "cash": cash,
            "holding_count": 1 if shares > 0 else 0,
            "holdings": f"{ts_code}:{shares * current_price:.2f}" if shares > 0 else "",
            "equity": total_equity
        })

    result_df = pd.DataFrame(daily_records)
    trade_df = pd.DataFrame(trade_records)

    if result_df.empty:
        return result_df, trade_df, {}

    result_df["strategy_nav"] = result_df["equity"] / result_df["equity"].iloc[0]
    result_df["strategy_return"] = result_df["strategy_nav"].pct_change().fillna(0)

    benchmark_nav_df = benchmark_df[["date", "close"]].copy().sort_values("date").reset_index(drop=True)
    benchmark_nav_df["benchmark_nav"] = benchmark_nav_df["close"] / benchmark_nav_df["close"].iloc[0]
    benchmark_nav_df["benchmark_return"] = benchmark_nav_df["benchmark_nav"].pct_change().fillna(0)

    merged_df = pd.merge(
        result_df,
        benchmark_nav_df[["date", "benchmark_nav", "benchmark_return"]],
        on="date",
        how="inner"
    )

    metrics = calculate_performance(merged_df)
    return merged_df, trade_df, metrics


# =========================
# 绩效计算
# =========================
def calculate_performance(df):
    if df.empty or "strategy_nav" not in df.columns:
        return {
            k: np.nan for k in [
                "策略收益", "基准收益", "策略年化收益", "基准年化收益",
                "Alpha", "Beta", "Sharpe", "最大回撤"
            ]
        }

    strategy_total = df["strategy_nav"].iloc[-1] - 1
    benchmark_total = df["benchmark_nav"].iloc[-1] - 1
    periods = len(df)

    strategy_annual = annualize_return(strategy_total, periods)
    benchmark_annual = annualize_return(benchmark_total, periods)
    max_dd = max_drawdown_from_nav(df["strategy_nav"])

    strategy_daily = df["strategy_return"]
    benchmark_daily = df["benchmark_return"]
    daily_rf = RISK_FREE_RATE / 252
    excess = strategy_daily - daily_rf

    sharpe = (
        excess.mean() / strategy_daily.std(ddof=1) * np.sqrt(252)
        if strategy_daily.std(ddof=1) > 0 else np.nan
    )

    beta = np.nan
    if benchmark_daily.var(ddof=1) > 0:
        beta = strategy_daily.cov(benchmark_daily) / benchmark_daily.var(ddof=1)

    alpha = (
        strategy_annual - (RISK_FREE_RATE + beta * (benchmark_annual - RISK_FREE_RATE))
        if pd.notna(beta) else np.nan
    )

    return {
        "策略收益": strategy_total,
        "基准收益": benchmark_total,
        "策略年化收益": strategy_annual,
        "基准年化收益": benchmark_annual,
        "Alpha": alpha,
        "Beta": beta,
        "Sharpe": sharpe,
        "最大回撤": max_dd
    }


# =========================
# 使用说明
# =========================
def generate_user_guide():
    content = """
RSI多资产组合回测工具使用说明

一、功能简介
本工具支持：
1. 多标的组合回测
2. 多策略参数对比
3. 单标的独立回测对比
4. 最新RSI信号查看
5. PDF回测报告下载

二、参数说明
1. 开始日期 / 结束日期
用于设定回测时间范围。

2. 基准
用于和策略结果做比较的基准指数，例如 000300.SH。

3. 标的代码
多个代码用英文逗号分隔，例如：
000001.SZ,510300.SH,159915.SZ

4. RSI周期
RSI计算所使用的周期，常用14。

5. 买入阈值
例如30，通常表示超卖区域参考线。

6. 卖出阈值
例如70，通常表示超买区域参考线。

7. 买入模式
- 上穿RSI值：昨天RSI低于阈值，今天RSI上穿阈值才买入，偏稳健
- 低于RSI值：只要当天RSI低于阈值就买入，偏激进

8. 卖出模式
- 下穿RSI值：昨天RSI高于阈值，今天RSI跌破阈值才卖出，偏稳健
- 高于RSI值：只要当天RSI高于阈值就卖出，偏激进

9. 启用止损
当持仓亏损达到设定比例时卖出。

10. 启用止盈
当持仓盈利达到设定比例时卖出。

11. 最大持仓数量
组合里最多同时持有多少个标的。

12. 单票最大仓位
单个标的最多占总资产的比例。

13. 开启多策略对比
会自动生成标准/激进/稳健三组参数进行比较。

三、结果说明
1. 策略净值对比
显示组合策略或多策略净值随时间变化曲线。

2. 单标的回测对比
显示各个标的单独投资时，与基准的净值曲线比较。

3. 策略绩效对比
显示年化收益、Sharpe、最大回撤、Alpha等指标。

4. 单标的绩效对比
显示每个标的单独投资时的表现。

5. 实时信号
显示最新价格、RSI值和交易信号（买入/卖出/观望）。

四、指标含义
1. 策略收益
整个回测周期内策略总收益率。

2. 策略年化收益
按年折算后的平均收益水平。

3. Sharpe
风险调整后收益指标，通常越高越好。

4. 最大回撤
回测期间从高点回落的最大幅度，通常越小越好。

5. Alpha
相对于基准的超额收益能力。

6. Beta
策略收益对基准波动的敏感度。

五、注意事项
1. 本工具当前主要基于日线收盘价信号，不是盘中实时交易系统。
2. 实时信号本质上更接近“最新日线信号”。
3. 若出现字体或PDF报错，请检查 fonts 目录中的字体文件是否存在。
"""
    return content.encode("utf-8")


# =========================
# PDF 辅助函数
# =========================
def ensure_pdf_fonts(pdf):
    """
    为 FPDF 注册中文字体
    返回:
        font_name: 字体名
        font_prop: matplotlib 用字体
    """
    regular_path, bold_path = get_font_paths()

    if not regular_path:
        raise FileNotFoundError(
            "未找到可用中文字体，请确认 fonts 目录中存在 NotoSansSC-6.ttf 或 MSYH.TTC。"
        )

    font_name = "CNFont"

    # FPDF 对 ttf 支持更稳
    pdf.add_font(font_name, "", regular_path)

    if bold_path:
        pdf.add_font(font_name, "B", bold_path)
    else:
        pdf.add_font(font_name, "B", regular_path)

    font_prop = fm.FontProperties(fname=regular_path)
    return font_name, font_prop


def add_trade_table_to_pdf(pdf, title, trade_df, font_name):
    pdf.add_page()
    pdf.set_font(font_name, "B", 12)
    pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font(font_name, "", 8)

    if trade_df is not None and not trade_df.empty:
        show_df = trade_df.tail(20).copy()

        for _, row in show_df.iterrows():
            line = (
                f"{str(row.get('date', ''))[:10]} | "
                f"{row.get('code', '')} | "
                f"{row.get('action', '')} | "
                f"价格:{row.get('price', 0):.2f} | "
                f"股数:{row.get('shares', 0):.2f}"
            )
            pdf.multi_cell(190, 6, line, new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.cell(0, 8, "暂无交易记录", new_x="LMARGIN", new_y="NEXT")


def safe_pdf_bytes(pdf):
    """
    兼容不同 FPDF/fpdf2 版本的输出
    """
    out = pdf.output(dest="S")
    if isinstance(out, bytes):
        return out
    if isinstance(out, str):
        return out.encode("latin1", errors="replace")
    return bytes(out)


def generate_pdf_report(results, trade_df, single_results, settings_summary):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    font_name, font_prop = ensure_pdf_fonts(pdf)

    # =========================
    # 第1页：首页 + 策略设置
    # =========================
    pdf.add_page()
    pdf.set_font(font_name, "B", 16)
    pdf.cell(0, 10, "RSI策略回测报告", new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.ln(5)
    pdf.set_font(font_name, "B", 12)
    pdf.cell(0, 8, "一、策略设置", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font(font_name, "", 10)
    for line in settings_summary:
        wrapped_lines = textwrap.wrap(str(line), width=40)
        for wline in wrapped_lines:
            pdf.multi_cell(190, 6, wline, new_x="LMARGIN", new_y="NEXT")

    # =========================
    # 第2页：绩效
    # =========================
    pdf.add_page()
    pdf.set_font(font_name, "B", 12)
    pdf.cell(0, 8, "二、策略绩效概览", new_x="LMARGIN", new_y="NEXT")

    if results:
        for res in results:
            pdf.set_font(font_name, "B", 11)
            pdf.cell(0, 7, res.get("name", "策略"), new_x="LMARGIN", new_y="NEXT")

            pdf.set_font(font_name, "", 10)
            for k, v in res.get("metrics", {}).items():
                if isinstance(v, (float, int, np.floating, np.integer)) and not pd.isna(v):
                    if "收益" in k or "回撤" in k:
                        text = f"{k}: {v:.2%}"
                    else:
                        text = f"{k}: {v:.4f}"
                else:
                    text = f"{k}: {v}"

                pdf.multi_cell(190, 6, text, new_x="LMARGIN", new_y="NEXT")

            pdf.ln(2)
    else:
        pdf.set_font(font_name, "", 10)
        pdf.cell(0, 8, "暂无绩效数据", new_x="LMARGIN", new_y="NEXT")

    # =========================
    # 第3页：柱状图
    # =========================
    pdf.add_page()
    pdf.set_font(font_name, "B", 12)
    pdf.cell(0, 8, "三、单标的收益对比", new_x="LMARGIN", new_y="NEXT")

    if single_results:
        fig, ax = plt.subplots(figsize=(10, 5))

        codes = [res["code"] for res in single_results]
        returns = [
            res["metrics"]["策略收益"] if pd.notna(res["metrics"].get("策略收益", np.nan)) else 0
            for res in single_results
        ]

        ax.bar(codes, returns)
        ax.set_title("单标的策略总收益对比", fontproperties=font_prop)
        ax.set_ylabel("收益率", fontproperties=font_prop)
        ax.tick_params(axis="x", rotation=30)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(font_prop)

        img_path = "bar.png"
        fig.tight_layout()
        fig.savefig(img_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        pdf.image(img_path, x=10, w=180)

        if os.path.exists(img_path):
            try:
                os.remove(img_path)
            except Exception:
                pass
    else:
        pdf.set_font(font_name, "", 10)
        pdf.cell(0, 8, "暂无单标的结果", new_x="LMARGIN", new_y="NEXT")

    # =========================
    # 第4页：交易记录
    # =========================
    add_trade_table_to_pdf(pdf, "四、组合交易明细", trade_df, font_name)

    return safe_pdf_bytes(pdf)


# =========================
# Streamlit 主界面
# =========================
st.set_page_config(page_title="RSI回测工具", layout="wide")
st.title("🚀 RSI多资产组合回测 & 实时信号小工具")

with st.sidebar:
    st.header("🔑 API 设置")
    token = st.text_input("tushare Token", type="password", value="hTASoWevdIQVKNJgEUGoDEWIMufHKYuLTSUGZfUOImwssjguKASNmWMywBkFgpjF")
    custom_url = st.text_input("自定义URL", value="http://124.222.60.121:8020/")

    st.header("📅 回测设置")
    start_date = st.date_input("开始日期", datetime(2019, 1, 1))
    end_date = st.date_input("结束日期", datetime(2026, 4, 21))
    benchmark = st.text_input("基准", "000300.SH")
    security_input = st.text_input("标的代码（逗号分隔）", "000001.SZ,510300.SH,159915.SZ")

    st.header("📊 RSI 参数")
    rsi_period = st.slider("RSI周期", 5, 30, 14)
    rsi_buy = st.slider("买入阈值", 0, 100, 30)
    rsi_sell = st.slider("卖出阈值", 0, 100, 70)

    buy_mode_label = st.selectbox(
        "买入模式",
        ["上穿RSI值（稳健）", "低于RSI值（激进）"]
    )
    sell_mode_label = st.selectbox(
        "卖出模式",
        ["下穿RSI值（稳健）", "高于RSI值（激进）"]
    )

    buy_mode_map = {
        "上穿RSI值（稳健）": "cross_up",
        "低于RSI值（激进）": "below"
    }
    sell_mode_map = {
        "下穿RSI值（稳健）": "cross_down",
        "高于RSI值（激进）": "above"
    }

    st.header("🛡️ 风控与仓位")
    use_stop_loss = st.checkbox("启用止损", True)
    stop_loss_pct = st.slider("止损比例 (%)", 1, 20, 8) / 100
    use_take_profit = st.checkbox("启用止盈", False)
    take_profit_pct = st.slider("止盈比例 (%)", 5, 30, 15) / 100
    max_holdings = st.slider("最大持仓数量", 1, 10, 3)
    max_pos_per = st.slider("单票最大仓位", 0.1, 1.0, 0.4)

    enable_multi = st.toggle("开启多策略对比", False)

    st.download_button(
        "📘 下载使用说明",
        generate_user_guide(),
        file_name="RSI工具使用说明.txt",
        mime="text/plain"
    )

# 主页面
tab1, tab2 = st.tabs(["📈 回测结果", "📡 实时信号"])

with tab1:
    if st.button("🚀 开始回测", type="primary", use_container_width=True):
        if not token:
            st.error("请输入 tushare Token")
        else:
            try:
                pro = init_tushare(token, custom_url if custom_url else None)
                security_list = [c.strip() for c in security_input.split(",") if c.strip()]

                base_params = {
                    "rsi_period": rsi_period,
                    "rsi_buy": rsi_buy,
                    "rsi_sell": rsi_sell,
                    "buy_signal_mode": buy_mode_map[buy_mode_label],
                    "sell_signal_mode": sell_mode_map[sell_mode_label],
                    "use_ma_filter": True,
                    "ma_period": 20,
                    "price_above_ma_buy": True,
                    "price_below_ma_sell": False,
                    "use_stop_loss": use_stop_loss,
                    "stop_loss_pct": stop_loss_pct,
                    "use_take_profit": use_take_profit,
                    "take_profit_pct": take_profit_pct,
                    "max_holdings": max_holdings,
                    "max_position_per_security": max_pos_per
                }

                data_dict, benchmark_df = load_all_data(
                    pro,
                    security_list,
                    benchmark,
                    start_date.strftime("%Y%m%d"),
                    end_date.strftime("%Y%m%d"),
                    rsi_period=rsi_period,
                    ma_period=20
                )

                if not data_dict or benchmark_df is None or benchmark_df.empty:
                    st.error("数据获取失败，请检查代码、Token、网络或标的代码")
                else:
                    results = []
                    trade_df = pd.DataFrame()

                    if enable_multi:
                        st.info("多策略对比模式已开启 🚀")
                        strategy_params_list = [
                            {"name": "RSI标准", **base_params},
                            {"name": "激进策略", **{
                                **base_params,
                                "rsi_buy": 25,
                                "rsi_sell": 75,
                                "stop_loss_pct": 0.05
                            }},
                            {"name": "稳健策略", **{
                                **base_params,
                                "rsi_buy": 35,
                                "rsi_sell": 65,
                                "stop_loss_pct": 0.10
                            }}
                        ]

                        for idx, p in enumerate(strategy_params_list):
                            name = p["name"]
                            params_copy = {k: v for k, v in p.items() if k != "name"}
                            result_df, tmp_trade_df, metrics = backtest_portfolio(
                                data_dict, benchmark_df, params_copy
                            )

                            if idx == 0:
                                trade_df = tmp_trade_df.copy()

                            results.append({
                                "name": name,
                                "result_df": result_df,
                                "metrics": metrics
                            })
                    else:
                        result_df, trade_df, metrics = backtest_portfolio(
                            data_dict, benchmark_df, base_params
                        )
                        results.append({
                            "name": "组合策略",
                            "result_df": result_df,
                            "metrics": metrics
                        })

                    single_results = []
                    for code, df_asset in data_dict.items():
                        single_result_df, single_trade_df, single_metrics = backtest_single_asset(
                            df_asset, benchmark_df, code, base_params
                        )

                        if not single_result_df.empty:
                            single_results.append({
                                "name": f"单标的-{code}",
                                "code": code,
                                "result_df": single_result_df,
                                "metrics": single_metrics,
                                "trade_df": single_trade_df
                            })

                    st.success("回测完成！")

                    st.subheader("📈 策略净值对比")
                    fig = go.Figure()

                    for res in results:
                        df = res["result_df"]
                        if df is not None and not df.empty:
                            fig.add_trace(go.Scatter(
                                x=df["date"],
                                y=df["strategy_nav"],
                                mode="lines",
                                name=res["name"]
                            ))

                    if results and results[0]["result_df"] is not None and not results[0]["result_df"].empty:
                        fig.add_trace(go.Scatter(
                            x=results[0]["result_df"]["date"],
                            y=results[0]["result_df"]["benchmark_nav"],
                            mode="lines",
                            name="基准",
                            line=dict(dash="dash")
                        ))

                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("📊 策略绩效对比")
                    metrics_table = []

                    for res in results:
                        m = res["metrics"]
                        metrics_table.append({
                            "策略": res["name"],
                            "策略总收益": f"{m['策略收益']:.2%}" if pd.notna(m["策略收益"]) else "N/A",
                            "年化收益": f"{m['策略年化收益']:.2%}" if pd.notna(m["策略年化收益"]) else "N/A",
                            "Sharpe": f"{m['Sharpe']:.2f}" if pd.notna(m["Sharpe"]) else "N/A",
                            "最大回撤": f"{m['最大回撤']:.2%}" if pd.notna(m["最大回撤"]) else "N/A",
                            "Alpha": f"{m['Alpha']:.4f}" if pd.notna(m["Alpha"]) else "N/A"
                        })

                    st.dataframe(pd.DataFrame(metrics_table), use_container_width=True)

                    if single_results:
                        st.subheader("📈 单标的回测对比")

                        fig_single = go.Figure()

                        for res in single_results:
                            fig_single.add_trace(go.Scatter(
                                x=res["result_df"]["date"],
                                y=res["result_df"]["strategy_nav"],
                                mode="lines",
                                name=res["code"]
                            ))

                        fig_single.add_trace(go.Scatter(
                            x=single_results[0]["result_df"]["date"],
                            y=single_results[0]["result_df"]["benchmark_nav"],
                            mode="lines",
                            name="基准",
                            line=dict(dash="dash")
                        ))

                        st.plotly_chart(fig_single, use_container_width=True)

                        st.subheader("📊 单标的绩效对比")
                        single_metrics_table = []

                        for res in single_results:
                            m = res["metrics"]
                            single_metrics_table.append({
                                "标的": res["code"],
                                "策略总收益": f"{m['策略收益']:.2%}" if pd.notna(m["策略收益"]) else "N/A",
                                "年化收益": f"{m['策略年化收益']:.2%}" if pd.notna(m["策略年化收益"]) else "N/A",
                                "Sharpe": f"{m['Sharpe']:.2f}" if pd.notna(m["Sharpe"]) else "N/A",
                                "最大回撤": f"{m['最大回撤']:.2%}" if pd.notna(m["最大回撤"]) else "N/A",
                                "Alpha": f"{m['Alpha']:.4f}" if pd.notna(m["Alpha"]) else "N/A"
                            })

                        st.dataframe(pd.DataFrame(single_metrics_table), use_container_width=True)

                    settings_summary = [
                        f"回测区间：{start_date} 至 {end_date}",
                        f"基准：{benchmark}",
                        f"标的列表：{', '.join(security_list)}",
                        f"RSI周期：{rsi_period}",
                        f"买入阈值：{rsi_buy}",
                        f"卖出阈值：{rsi_sell}",
                        f"买入模式：{buy_mode_label}",
                        f"卖出模式：{sell_mode_label}",
                        f"启用止损：{'是' if use_stop_loss else '否'}，止损比例：{stop_loss_pct:.0%}",
                        f"启用止盈：{'是' if use_take_profit else '否'}，止盈比例：{take_profit_pct:.0%}",
                        f"最大持仓数量：{max_holdings}",
                        f"单票最大仓位：{max_pos_per:.0%}",
                        f"多策略对比：{'开启' if enable_multi else '关闭'}"
                    ]

                    try:
                        pdf_bytes = generate_pdf_report(results, trade_df, single_results, settings_summary)
                        st.download_button(
                            "📄 下载 PDF 回测报告",
                            pdf_bytes,
                            file_name=f"RSI回测报告_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as pdf_err:
                        st.error(f"PDF 生成失败：{pdf_err}")

            except Exception as e:
                st.error(f"运行报错：{e}")

with tab2:
    st.subheader("📡 实时信号")

    if st.button("获取最新信号"):
        if not token:
            st.error("请输入 tushare Token")
        else:
            try:
                pro = init_tushare(token, custom_url if custom_url else None)
                security_list = [c.strip() for c in security_input.split(",") if c.strip()]
                signals = []

                for code in security_list:
                    df = fetch_asset_data(
                        pro,
                        code,
                        (datetime.now() - pd.Timedelta(days=60)).strftime("%Y%m%d"),
                        datetime.now().strftime("%Y%m%d")
                    )

                    if df is None or len(df) < 20:
                        continue

                    df["rsi"] = calc_rsi(df["close"], rsi_period)
                    df["ma"] = df["close"].rolling(20).mean()

                    i = len(df) - 1
                    price = df.iloc[i]["close"]
                    rsi_val = df.iloc[i]["rsi"]
                    signal = "观望"

                    if should_buy(df, i, {
                        "rsi_period": rsi_period,
                        "rsi_buy": rsi_buy,
                        "buy_signal_mode": buy_mode_map[buy_mode_label],
                        "use_ma_filter": True,
                        "ma_period": 20,
                        "price_above_ma_buy": True
                    }):
                        signal = "🟢 买入"

                    elif should_sell(df, i, None, {
                        "rsi_period": rsi_period,
                        "rsi_sell": rsi_sell,
                        "sell_signal_mode": sell_mode_map[sell_mode_label],
                        "use_ma_filter": True,
                        "ma_period": 20,
                        "price_below_ma_sell": False
                    }):
                        signal = "🔴 卖出"

                    signals.append({
                        "代码": code,
                        "价格": round(price, 2),
                        "RSI": round(rsi_val, 2) if pd.notna(rsi_val) else np.nan,
                        "信号": signal
                    })

                if signals:
                    st.dataframe(pd.DataFrame(signals), use_container_width=True)
                else:
                    st.warning("没有获取到数据")
            except Exception as e:
                st.error(f"获取实时信号失败：{e}")

st.caption("如仍有报错，请把具体错误信息发给我，我继续帮你修复。")
