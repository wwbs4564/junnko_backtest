import streamlit as st
import pandas as pd
from junnko_backtest import *
from streamlit_echarts import st_pyecharts, st_echarts
from stqdm import stqdm
from streamlit_ace import st_ace
import empyrical
import gc
import base64


backtest = Junnko_Backtest()

event_tab, factor_tab, data_tab, test_tab = st.tabs(["事件化回测", "向量化回测", '数据检视', '调试'])

with st.sidebar:
    st.logo('logo.bmp')
    st.header("Junnko Backtest")
    st.header('Author: wwbs')
    backtest_start_date = st.date_input("开始日期:", value=pd.to_datetime("2010-01-01"))
    backtest_end_date = st.date_input("结束日期:", value=pd.to_datetime("2024-01-01")) 
    backtest_benchmark = st.text_input("基准", value='399300.SZ')

with event_tab:
    
    col1, col2, col3 = st.columns(3)
    event_backtest_start = st.button("运行回测")

    metric_placeholder = st.container()
    plot_placeholder = st.empty()
    table_placeholder = st.expander('回测结果表格')

    strategy_expander = st.expander('在此编写策略')

    
    with col1:
        event_backtest_name = st.text_input("策略名称（不要重复）:", value='沪深300中ROE前60')
    with col2:
        event_backtest_initial_capital = st.number_input("初始资金:", min_value=0, value=1000000)
    with col3:
        event_backtest_commission_rate = st.number_input("交易费率（单位为千分之一）:", min_value=0, value=4)
    

    with strategy_expander:
        event_tab_wrap = st.checkbox('代码过长自动换行', key=1)
        strategy_code = st_ace(value='''# 持有沪深300中（年度）ROE前60的股票
def my_strategy(self):
    if self.current_date.month in [5, 11]:
        # 判断当天是否是当月第一个交易日
        trade_cal = self.get_trade_cal(exchange='SSE')
        trade_cal = trade_cal[trade_cal['is_open']==1]
        latest_trade_date = trade_cal['cal_date'].iloc[-1]
        part_trade_cal = trade_cal[trade_cal['cal_date'].str.contains(latest_trade_date[:6])]
        if len(part_trade_cal) == 1:
            codes = self.get_index_component(index_code='399300.SZ', date=None)['con_code'].tolist()
            net_profit = self.get_daily_report_data(codes=codes, col='n_income', table='income', date=None, report_type=4)
            equity = self.get_daily_report_data(codes=codes, col='total_hldr_eqy_inc_min_int', table='balancesheet', date=None, report_type=4)
            roe = net_profit['n_income'] / equity['total_hldr_eqy_inc_min_int']
            roe = roe.sort_values(ascending=False)
            top_60_roe = roe.index.tolist()[:60]
            current_position = list(self.position.keys())
            for code in current_position:
                if code not in top_60_roe:
                    self.order_target_shares(code, 0)
            market_value = self.get_daily_data(col='circ_mv', table='stock_daily_basic', codes = top_60_roe, date=None)
            net_value = self.calculate_net_value()
            for code in top_60_roe:
                self.order_target_value(code, net_value*(market_value.loc[code, 'circ_mv']/market_value['circ_mv'].sum()))
                ''', language='python', auto_update=True, font_size=20, wrap=event_tab_wrap)
    
    if event_backtest_start:
        # 执行回测并动态更新图表
        for event_backtest_result in backtest.run_event_backtest(
            name=event_backtest_name,
            start_date=backtest_start_date.strftime('%Y%m%d'), 
            end_date=backtest_end_date.strftime('%Y%m%d'), 
            initial_capital=event_backtest_initial_capital, 
            commission_rate=event_backtest_commission_rate/1000, 
            strategy_code=strategy_code,
            benchmark=backtest_benchmark
            ):
                  
            with plot_placeholder:
                with st.container(height=500):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(name='策略', x=event_backtest_result.index, y=event_backtest_result['策略_标准化'], mode='lines'))
                    fig.add_trace(go.Scatter(name=backtest_benchmark, x=event_backtest_result.index, y=event_backtest_result[f'{backtest_benchmark}_标准化'], mode='lines'))
                    fig.add_trace(go.Scatter(name='现金', x=event_backtest_result.index, y=event_backtest_result['现金_标准化'], mode='lines'))
                    st.write(fig)

        metrics = calc_metrics(event_backtest_result['策略'].pct_change(), event_backtest_result[backtest_benchmark].pct_change())
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


with factor_tab:
    col1, col2, col3, col4 = st.columns(4)
    factor_file = st.file_uploader("上传因子文件（用于单因子回测）", type=['parquet'])
    factor_folder = st.text_input("因子文件夹路径（用于批量因子回测）")
    factor_backtest_start = st.button("单因子回测", key='factor_backtest_start')
    multiple_factor_backtest_start = st.button("批量因子回测", key='multiple_factor_backtest_start')

    metric_placeholder = st.container()
    plot_placeholder_1 = st.empty()
    plot_placeholder_2 = st.empty()
    plot_placeholder_3 = st.empty()
    plot_placeholder_4 = st.empty()
    table_placeholder = st.expander('回测结果表格')

    with col1:
        factor_backtest_name = st.text_input("因子名称（不要重复）:", value='ROE')
    with col2:
        factor_backtest_num_groups = st.number_input("分组数量:", min_value=0, value=5)
    with col3:
        factor_backtest_stock_pool = st.text_input("股票池:", value='399300.SZ')
    with col4:
        factor_shift = st.slider("因子的滞后天数，防止未来函数", min_value=0, max_value=10, value=1, step=1)
    
    if factor_backtest_start:
        factor = pd.read_parquet(factor_file)
        factor_backtest_result, metrics, line, ic_plots = backtest.run_factor_backtest(
            name = factor_backtest_name,
            start_date=backtest_start_date.strftime('%Y%m%d'), 
            end_date=backtest_end_date.strftime('%Y%m%d'),
            factor=factor,
            num_groups=factor_backtest_num_groups,
            stock_pool=factor_backtest_stock_pool,
            benchmark=backtest_benchmark,
            factor_shift=factor_shift
            )

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
        
        with plot_placeholder_1:
            st_pyecharts(line, height=500)
        with plot_placeholder_2:
            st_pyecharts(ic_plots[0])
        with plot_placeholder_3:
            st_pyecharts(ic_plots[1])
        with plot_placeholder_4:
            st_pyecharts(ic_plots[2])

        with st.expander('回测结果表格'):
            st.dataframe(factor_backtest_result)
    
    if multiple_factor_backtest_start:
        factor_filenames = os.listdir(factor_folder)
        factor_files = [pd.read_parquet(os.path.join(factor_folder, f)) for f in factor_filenames]
        factor_names = [f.split('.')[0] for f in factor_filenames]
        for i in stqdm(range(len(factor_files))):
            factor_backtest_result, metrics, line, ic_plots = backtest.run_factor_backtest(
                name = factor_names[i],
                start_date=backtest_start_date.strftime('%Y%m%d'), 
                end_date=backtest_end_date.strftime('%Y%m%d'),
                factor=factor_files[i],
                num_groups=factor_backtest_num_groups,
                stock_pool=factor_backtest_stock_pool,
                benchmark=backtest_benchmark,
                factor_shift=factor_shift
                )
            

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
    
    elif selected_table in ['income', 'balancesheet', 'cashflow', 'stock_daily_price', 'stock_daily_price_hfq', 'index_daily_price', 'stock_daily_basic']:
        query = f"SELECT DISTINCT ts_code FROM {selected_table}"
        all_codes = pd.read_sql(query, backtest.conn)['ts_code'].tolist()
        selected_code = st.selectbox("请选择一个代码:", all_codes, index=0)

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
            
            code_to_update_fs = st.text_input("请输入获取的代码:", value='')
            get_start_date_fs = st.text_input("请输入开始日期:", value='20000101')
            get_new_button_fs = st.button('获取数据')
            if get_new_button_fs:
                backtest.update_financial_statement(table=selected_table, code=code_to_update_fs, start_date=get_start_date_fs)
            
        elif selected_table in ['stock_daily_price', 'stock_daily_price_hfq', 'index_daily_price', 'stock_daily_basic']:
            df = pd.read_sql(query, backtest.conn).sort_values(by='trade_date', ascending=False)
            st.markdown(f'最新**交易日期**: {df["trade_date"].max()}')
            st.markdown(f'最近**更新时间**：{df["INSERT_TIME"].max()}')

            update_current_code = st.button('更新当前代码数据')
            update_all_code = st.button('更新所有代码数据')

            if update_current_code:
                if selected_table in ['stock_daily_price', 'stock_daily_price_hfq', 'index_daily_price']:
                    backtest.update_daily_price(table=selected_table, code=selected_code, start_date=df["trade_date"].max())
                elif selected_table in ['stock_daily_basic']:
                    backtest.update_daily_basic(code=selected_code, start_date=df["trade_date"].max())

            elif update_all_code:
                if selected_table  in ['stock_daily_price', 'stock_daily_price_hfq']:
                    query = f"select ts_code from stock_info"
                    stock_codes = pd.read_sql(query, backtest.conn)['ts_code'].to_list()
                    for c in tqdm(stock_codes):
                        query = f'select max(trade_date) from {selected_table} where ts_code = "{c}"'
                        start_date = pd.read_sql(query, backtest.conn).iloc[0, 0]
                        backtest.update_daily_price(table=selected_table, code=c, start_date=start_date)

                elif selected_table == 'index_daily_price':
                    query = f"select index_code from index_info"
                    index_codes = pd.read_sql(query, backtest.conn)['index_code'].to_list()
                    for c in tqdm(index_codes):
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
        

