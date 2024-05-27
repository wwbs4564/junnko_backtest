import streamlit as st
import pandas as pd
from junnko_backtest import *
from streamlit_echarts import st_pyecharts, st_echarts
from pyecharts import options as opts
from pyecharts.charts import Boxplot, Line,Bar, Grid, Page, Timeline
from pyecharts.globals import CurrentConfig, NotebookType
from stqdm import stqdm
from streamlit_ace import st_ace
import empyrical


backtest = Junnko_Backtest()

st.logo('logo.bmp')
st.header("Junnko Backtest")
st.header('Author: wwbs')

event_tab, factor_tab, data_tab, test_tab = st.tabs(["事件化回测", "向量化回测", '数据检视', '调试'])

with event_tab:
    event_backtest_name = st.text_input("策略名称（不要重复）:", value='沪深300中ROE前60')
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    event_backtest_benchmark = st.text_input("基准", value='399300.SZ')
    event_backtest_start = st.button("运行回测")
    strategy_expander = st.expander('在此编写策略')

    metric_placeholder = st.container()
    plot_placeholder = st.empty()
    table_placeholder = st.expander('回测结果表格')

    with col1:
        event_backtest_start_date = st.date_input("开始日期:", value=pd.to_datetime("2010-01-01"))
    with col2:
        event_backtest_end_date = st.date_input("结束日期:", value=pd.to_datetime("2020-01-01")) 
    with col3:
        event_backtest_initial_capital = st.number_input("初始资金:", min_value=0, value=1000000)
    with col4:
        event_backtest_commission_rate = st.number_input("交易费率（单位为千分之一）:", min_value=0, value=4)
    

    with strategy_expander:
        event_tab_wrap = st.checkbox('代码过长自动换行', key=1)
        strategy_code = st_ace(value='''# 持有沪深300中ROE前60的股票
def my_strategy(self):
    # 判断当天是否是当月第一个交易日
    trade_cal = self.get_trade_cal(exchange='SSE')
    trade_cal = trade_cal[trade_cal['is_open']==1]
    latest_trade_date = trade_cal['cal_date'].max()
    part_trade_cal = trade_cal[trade_cal['cal_date'].str.contains(latest_trade_date[:6])]
    if len(part_trade_cal) == 1:
        codes = self.get_index_component(index_code='399300.SZ', date=None)['con_code'].tolist()
        net_profit = self.get_daily_data(codes=codes, col='n_income', table='income', date=None)
        equity = self.get_daily_data(codes=codes, col='total_hldr_eqy_inc_min_int', table='balancesheet', date=None)
        roe = net_profit['n_income'] / equity['total_hldr_eqy_inc_min_int']
        roe = roe.sort_values(ascending=False)
        top_60_roe = roe.index.tolist()[:60]
        current_position = list(self.position.keys())
        for code in current_position:
            if code not in top_60_roe:
                self.order_target_shares(code, 0)
        net_value = self.calculate_net_value()
        for code in top_60_roe:
            self.order_target_value(code, net_value/60)
                ''', language='python', auto_update=True, font_size=20, wrap=event_tab_wrap)
    
    if event_backtest_start:
        # 动态定义策略函数
        exec(strategy_code, globals())
        strategy = globals()['my_strategy']
        all_dates = pd.date_range(event_backtest_start_date, event_backtest_end_date)
        event_backtest_result = pd.DataFrame(index=all_dates, columns=["现金", "策略", '基准'])

        # 执行回测并动态更新图表
        for date_idx, cash, net_value in backtest.run_event_backtest(event_backtest_start_date.strftime('%Y%m%d'), event_backtest_end_date.strftime('%Y%m%d'), event_backtest_initial_capital, event_backtest_commission_rate/1000, strategy):
            event_backtest_result.loc[all_dates[date_idx], "现金"] = cash
            event_backtest_result.loc[all_dates[date_idx], "策略"] = net_value
            event_backtest_result.loc[all_dates[date_idx], "基准"] = backtest.get_daily_data(col='close', table='index_daily_price', codes=event_backtest_benchmark, date=None)['close'].iloc[0]
            
        line = (
            Line()
            .add_xaxis(list(event_backtest_result.index.strftime("%Y-%m-%d")))
            .add_yaxis("资产净值", list(event_backtest_result["策略"]/event_backtest_result['策略'].iloc[0]))
            .add_yaxis("基准", list(event_backtest_result["基准"]/event_backtest_result['基准'].iloc[0]))
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
                yaxis_opts=opts.AxisOpts(type_="value"),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                datazoom_opts=[opts.DataZoomOpts(type_="inside", range_start=0, range_end=100), opts.DataZoomOpts(type_="slider", range_start=0, range_end=100)]
            )
        )
        with plot_placeholder:
            with st.container(height=500):
                st_pyecharts(line, height=500)
        
        metrics = calc_metrics(event_backtest_result['策略'].pct_change(), event_backtest_result['基准'].pct_change())
        with metric_placeholder:
            annual_ret_col, annual_vol_col = st.columns(2)
            cum_ret_col, excess_cum_ret_col, sharpe_ret_col = st.columns(3)
            with annual_ret_col:
                st.metric(label="年化收益", value=f"{round(metrics['年化收益']*100, 2)}%")
            with annual_vol_col:
                st.metric(label="年化波动", value=f"{round(metrics['年化波动']*100, 2)}%")
            with cum_ret_col:
                st.metric(label="累计收益", value=f"{round(metrics['累计收益']*100, 2)}%")
            with excess_cum_ret_col:
                st.metric(label="累计超额收益", value=f"{round(metrics['累计超额收益']*100, 2)}%")
            with sharpe_ret_col:
                st.metric(label="夏普比率", value=round(metrics['sharpe比率'], 2))
            with st.expander('更多指标'):
                st.table(metrics)
        
        table_placeholder.dataframe(event_backtest_result)

        os.makedirs(f'event_result/{event_backtest_name}')
        # 保存策略代码
        with open(f'event_result/{event_backtest_name}/strategy.py', 'w') as f:
            f.write(strategy_code)
        # 保存回测结果图（html格式）
        line.render(f'event_result/{event_backtest_name}/backtest_result.html')
        # 保存回测结果表格（xlsx格式）
        event_backtest_result.to_excel(f'event_result/{event_backtest_name}/backtest_result.xlsx')
        # 保存回测结果指标（xlsx格式）
        metrics.to_excel(f'event_result/{event_backtest_name}/metrics.xlsx')

with factor_tab:
    factor_backtest_name = st.text_input("因子名称（不要重复）:", value='ROE')
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col5, col6 = st.columns(2)
    factor_file = st.file_uploader("上传因子文件", type=['parquet'])
    factor_backtest_start = st.button("运行回测", key='factor_start_backtest')

    metric_placeholder = st.container()
    plot_placeholder = st.empty()
    table_placeholder = st.expander('回测结果表格')

    with col1:
        factor_backtest_start_date = st.date_input("开始日期:", value=pd.to_datetime("2010-01-01"), key='factor_start_date')
    with col2:
        factor_backtest_end_date = st.date_input("结束日期:", value=pd.to_datetime("2020-01-01"), key='factor_end_date')
    with col3:
        factor_backtest_num_groups = st.number_input("分组数量:", min_value=0, value=5)
    with col4:
        factor_backtest_freq = st.selectbox("调仓频率:", ['D', 'W', 'M', 'Q', 'Y'], index=2)
    with col5:
        factor_backtest_benchmark = st.text_input("基准:", value='399300.SZ', key='factor_benchmark')
    with col6:
        factor_backtest_stock_pool = st.text_input("股票池:", value='399300.SZ')
    
    if factor_backtest_start:
        factor = pd.read_parquet(factor_file)
        factor_backtest_result = backtest.run_factor_backtest(
            start_date=factor_backtest_start_date.strftime('%Y%m%d'), 
            end_date=factor_backtest_end_date.strftime('%Y%m%d'),
            freq=factor_backtest_freq,
            factor=factor,
            num_groups=factor_backtest_num_groups,
            stock_pool=factor_backtest_stock_pool
            )
        benchmark_daily_price = backtest.get_historical_data(col='close', table='index_daily_price', code=factor_backtest_benchmark, start_date=factor_backtest_start_date, end_date=factor_backtest_end_date, during_backtest=False).drop('ts_code', axis=1)
        benchmark_daily_price.index = pd.to_datetime(benchmark_daily_price.index)
        benchmark_daily_price = benchmark_daily_price.reindex(pd.date_range(start=factor_backtest_start_date, end=factor_backtest_end_date, freq=factor_backtest_freq), method='ffill')
        factor_backtest_result['基准'] = benchmark_daily_price['close'].pct_change()
        factor_backtest_result = factor_backtest_result.fillna(0)

        metrics = calc_metrics(factor_backtest_result[factor_backtest_num_groups-1], factor_backtest_result['基准'])
        with metric_placeholder:
            annual_ret_col, annual_vol_col = st.columns(2)
            cum_ret_col, excess_cum_ret_col, sharpe_ret_col = st.columns(3)
            with annual_ret_col:
                st.metric(label="年化收益", value=f"{round(metrics['年化收益']*100, 2)}%")
            with annual_vol_col:
                st.metric(label="年化波动", value=f"{round(metrics['年化波动']*100, 2)}%")
            with cum_ret_col:
                st.metric(label="累计收益", value=f"{round(metrics['累计收益']*100, 2)}%")
            with excess_cum_ret_col:
                st.metric(label="累计超额收益", value=f"{round(metrics['累计超额收益']*100, 2)}%")
            with sharpe_ret_col:
                st.metric(label="夏普比率", value=round(metrics['sharpe比率'], 2))
            with st.expander('更多指标'):
                st.table(metrics)
        factor_backtest_result = (factor_backtest_result+1).cumprod()
        line = (
                Line()
                .add_xaxis(list(factor_backtest_result.index.strftime("%Y-%m-%d")))
                .add_yaxis('基准', factor_backtest_result['基准'])
                .set_global_opts(
                    xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
                    yaxis_opts=opts.AxisOpts(type_="value"),
                    tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                    datazoom_opts=[opts.DataZoomOpts(type_="inside", range_start=0, range_end=100), opts.DataZoomOpts(type_="slider", range_start=0, range_end=100)]
                )
            )
        for group in range(factor_backtest_num_groups):
            line.add_yaxis(f'第{group}组', factor_backtest_result[group])
        with plot_placeholder:
            with st.container(height=500):
                st_pyecharts(line, height=500)
        with st.expander('回测结果表格'):
            st.dataframe(factor_backtest_result)
        
        os.makedirs(f'factor_result/{factor_backtest_name}')
        # 保存回测结果图（html格式）
        line.render(f'factor_result/{factor_backtest_name}/backtest_result.html')
        # 保存回测结果表格（xlsx格式）
        factor_backtest_result.to_excel(f'factor_result/{factor_backtest_name}/backtest_result.xlsx')
        # 保存回测结果指标（xlsx格式）
        metrics.to_excel(f'factor_result/{factor_backtest_name}/metrics.xlsx')
        # 保存因子文件（parquet格式）
        factor.to_parquet(f'factor_result/{factor_backtest_name}/factor.parquet')
        

with data_tab:
    tables = backtest.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    tables = [row[0] for row in tables]
    selected_table = st.selectbox("请选择一个表格:", tables, index=None)

    if selected_table in ['stock_info', 'index_info']:
        query = f"SELECT * FROM {selected_table}"
        df = pd.read_sql(query, backtest.conn).sort_values(by='list_date', ascending=False)
        # list_date字段数据类型混杂，暂时不显示
        # st.markdown(f'最新**上市日期**: {df["list_date"].max()}')
        st.markdown(f'最近**更新时间**：{df["INSERT_TIME"].max()}')

        update_button = st.button('更新当前表格')
        if update_button:
            backtest.update_info(table=selected_table)

        st.dataframe(df)
    
    elif selected_table in ['trade_cal']:
        query = f"SELECT DISTINCT exchange FROM {selected_table}"
        all_exchanges = pd.read_sql(query, backtest.conn)['exchange'].tolist()

        selected_exchange = st.selectbox("请选择一个交易所:", all_exchanges, index=None)
        query = f"SELECT * FROM {selected_table} WHERE exchange = '{selected_exchange}'"
        df = pd.read_sql(query, backtest.conn).sort_values(by='cal_date', ascending=False)

        st.markdown(f'最新**交易日期**: {df["cal_date"].max()}')
        st.markdown(f'最近**更新时间**：{df["INSERT_TIME"].max()}')

        update_button = st.button('更新交易日历')
        if update_button:
            backtest.update_trade_cal(exchange=selected_exchange)
            st.success(f'已更新{selected_exchange}的交易日历')

        st.dataframe(df)
    
    elif selected_table in ['income', 'balancesheet', 'cashflow', 'stock_daily_price', 'index_daily_price', 'stock_daily_basic']:
        query = f"SELECT DISTINCT ts_code FROM {selected_table}"
        all_codes = pd.read_sql(query, backtest.conn)['ts_code'].tolist()
        selected_code = st.selectbox("请选择一个代码:", all_codes, index=0)
        if selected_code is None:
            selected_code = st.text_input("请输入一个代码:", value='000001.SZ')

        query = f"SELECT * FROM {selected_table} WHERE ts_code = '{selected_code}'"
        if selected_table in ['income', 'balancesheet', 'cashflow']:
            df = pd.read_sql(query, backtest.conn).sort_values(by='f_ann_date', ascending=False)
            st.markdown(f'最新**实际公告日期**: {df["f_ann_date"].max()}')
            st.markdown(f'最新**报告期**: {df["end_date"].max()}')
            st.markdown(f'最近**更新时间**：{df["INSERT_TIME"].max()}')

            update_current_code = st.button('更新当前股票数据')
            update_all_code = st.button('更新所有股票数据')

            if update_current_code:
                backtest.update_financial_statement(table=selected_table, code=selected_code, start_date=df["f_ann_date"].max())
                st.success(f'已更新{selected_code}的数据')
            elif update_all_code:
                query = f"select ts_code from stock_info"
                codes = pd.read_sql(query, backtest.conn)['ts_code'].to_list()
                for c in tqdm(codes):
                    backtest.update_financial_statement(table=selected_table, code=c, start_date=None)
            
        elif selected_table in ['stock_daily_price', 'index_daily_price', 'stock_daily_basic']:
            df = pd.read_sql(query, backtest.conn).sort_values(by='trade_date', ascending=False)
            st.markdown(f'最新**交易日期**: {df["trade_date"].max()}')
            st.markdown(f'最近**更新时间**：{df["INSERT_TIME"].max()}')

            update_current_code = st.button('更新当前代码数据')
            update_all_code = st.button('更新所有代码数据')

            if update_current_code:
                if selected_table in ['stock_daily_price', 'index_daily_price']:
                    backtest.update_daily_price(table=selected_table, code=selected_code, start_date=df["trade_date"].max())
                elif selected_table in ['stock_daily_basic']:
                    backtest.update_daily_basic(code=selected_code, start_date=df["trade_date"].max())

            elif update_all_code:
                if selected_table  == 'stock_daily_price':
                    query = f"select ts_code from stock_info"
                    stock_codes = pd.read_sql(query, backtest.conn)['ts_code'].to_list()
                    for c in tqdm(stock_codes):
                        query = f'select max(trade_date) from stock_daily_price where ts_code = "{c}"'
                        start_date = pd.read_sql(query, backtest.conn).iloc[0, 0]
                        backtest.update_daily_price(table=selected_table, code=c, start_date=start_date)

                elif selected_table == 'index_daily_price':
                    query = f"select index_code from index_info"
                    index_codes = pd.read_sql(query, backtest.conn)['index_code'].to_list()
                    for c in index_codes:
                        query = f'select max(trade_date) from index_daily_price where index_code = "{c}"'
                        start_date = pd.read_sql(query, backtest.conn).iloc[0, 0]
                        backtest.update_daily_price(table=selected_table, code=c, start_date=None)
                
                elif selected_table == 'stock_daily_basic':
                    query = f"select ts_code from stock_info"
                    stock_codes = pd.read_sql(query, backtest.conn)['ts_code'].to_list()
                    for c in tqdm(stock_codes):
                        query = f'select max(trade_date) from stock_daily_basic where ts_code = "{c}"'
                        start_date = pd.read_sql(query, backtest.conn).iloc[0, 0]
                        backtest.update_stock_daily_basic(code=c, start_date=start_date)

        st.dataframe(df)
    
    elif selected_table in ['index_component']:
        query = f"SELECT DISTINCT index_code FROM {selected_table}"
        all_index_codes = pd.read_sql(query, backtest.conn)['index_code'].tolist()
        selected_index_code = st.selectbox("请选择一个指数:", all_index_codes, index=None)

        query = f"SELECT * FROM {selected_table} WHERE index_code = '{selected_index_code}'"
        df = pd.read_sql(query, backtest.conn).sort_values(by='trade_date', ascending=False)
        st.markdown(f'最新**交易日期**: {df["trade_date"].max()}')
        st.markdown(f'最近**更新时间**：{df["INSERT_TIME"].max()}')

        
        update_button = st.button('更新指数成分股')
        if update_button:
            backtest.update_index_component(index_code=selected_index_code, last_date=df["trade_date"].max())
        
        st.dataframe(df)
        
        st.markdown(f'指数成分股数据量过大，不支持批量更新，若想获取当前不存在的指数成分，请在下方文本框中输入代码后点击获取按钮')

        index_code_to_update = st.text_input("请输入一个指数代码:", value='')
        get_start_date = st.text_input("请输入开始日期:", value='20000101')
        get_new_index_component_button = st.button('获取指数成分股')
        if get_new_index_component_button:
            backtest.update_index_component(index_code=index_code_to_update, last_date=get_start_date)

with test_tab:
    st.warning('如果要调用数据获取函数，请务必将during_backtest设置为False')
    test_tab_wrap = st.checkbox('代码过长自动换行', key=2)
    factor_construction_code = st_ace(value="test_result = backtest.get_daily_data(col='close', table='stock_daily_price', codes=['000001.SZ', '000002.SZ'], date='20180101', during_backtest=False)", language='python', auto_update=True, font_size=20, wrap=test_tab_wrap)
    if st.button('运行代码'):
        exec(factor_construction_code, globals())
        st.dataframe(globals()['test_result'])
        

