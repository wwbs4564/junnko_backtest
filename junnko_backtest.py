import pandas as pd
import numpy as np
import tushare as ts
import os
import datetime
import empyrical
import pickle
from tqdm import tqdm
from tqdm.notebook import tqdm as ntqdm
import inspect
import sqlite3
import time
from functools import wraps
from stqdm import stqdm
import warnings
warnings.filterwarnings('ignore')

with open('token.txt', 'r') as f:
    TS_TOKEN = f.read()

with open('db_path.txt', 'r') as f:
    DB_PATH = f.read()

if not os.path.exists('event_result'):
    os.mkdir('event_result')
if not os.path.exists('factor_result'):
    os.mkdir('factor_result')


pro = ts.pro_api(TS_TOKEN)
ts.set_token(TS_TOKEN)


if '功能函数':
    def decide_date_col(table):
        if table in ['income', 'balancesheet', 'cashflow']:
            date_col = 'f_ann_date'
        elif table in ['stock_daily_price', 'stock_daily_price_hfq', 'stock_daily_basic', 'index_daily_price']:
            date_col = 'trade_date'
        return date_col
    
    def get_month_bounds(date):
        date = datetime.datetime.strptime(date, '%Y%m%d')
        first_day = date.replace(day=1)
        if first_day.month != 12:
            last_day = first_day.replace(month=first_day.month+1) - datetime.timedelta(days=1) 
        else:
            last_day = first_day.replace(year=first_day.year+1, month=1) - datetime.timedelta(days=1)
        return first_day.strftime('%Y%m%d'), last_day.strftime('%Y%m%d')
    
    def iter_rate_limiter(max_iter_per_min, actual_iter_time):
        sec_per_iter = 60 / max_iter_per_min
        if sec_per_iter <= actual_iter_time:
            pass
        else:
            time.sleep(sec_per_iter - actual_iter_time)

    def get_primary_key(table, conn):
        query = f"PRAGMA table_info({table});"
        table_info = pd.read_sql(query, conn)
        return table_info[table_info['pk']>0]['name'].to_list()
    
    def calc_metrics(rets, benchmark_rets):
        metrics = {
            '累计收益': empyrical.cum_returns(rets).iloc[-1],  # 使用最后一个累计收益值
            '平均周收益': empyrical.aggregate_returns(rets, 'weekly').mean(),
            '平均月收益': empyrical.aggregate_returns(rets, 'monthly').mean(),
            '平均年收益': empyrical.aggregate_returns(rets, 'yearly').mean(),
            '最大回撤': empyrical.max_drawdown(rets),
            '年化收益': empyrical.annual_return(rets, period='daily'),
            '年化波动': empyrical.annual_volatility(rets, period='daily'),
            'calmar比率': empyrical.calmar_ratio(rets),
            'omega比率': empyrical.omega_ratio(rets),
            'sharpe比率': empyrical.sharpe_ratio(rets),
            'sortino比率': empyrical.sortino_ratio(rets),
            'alpha': empyrical.alpha(rets, benchmark_rets),
            'beta': empyrical.beta(rets, benchmark_rets)
        }
        metrics = pd.Series(metrics)
        cum_benchmark_ret = empyrical.cum_returns(benchmark_rets).iloc[-1]
        metrics['累计超额收益'] = metrics['累计收益'] - cum_benchmark_ret
        return metrics
    
    def map_code(code):
        code_map = {
            '000022.SZ' : '001872.SZ',
            '600087.SH' : '601975.SH'
        }
        if code in code_map.keys():
            return code_map[code]
        else:
            return code
    
        
class Database:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self.cursor = self.conn.cursor()
    
    def verify_date(self, date):
        if date is None:
            date = self.current_date.strftime('%Y%m%d')
        elif datetime.datetime.strptime(date, '%Y%m%d') > self.current_date:
            date = self.current_date.strftime('%Y%m%d')
        return date

    def set_update_fields(self, table):
        query = f'select * from {table} limit 1'
        df = pd.read_sql(query, self.conn)
        fields = df.columns.to_list()
        fields.remove('INSERT_TIME')
        fields = (',').join(fields)
        return fields
    
    
    def verify_codes(self, codes):
        if isinstance(codes, str):
            codes = [codes, 'just a place holder']
        elif isinstance(codes, list) and len(codes) == 1:
            codes.append('just a place holder')
        elif codes is None:
            codes = self.get_stock_info(codes=None).index.to_list()
        codes = [map_code(code) for code in codes]
        return codes

    if '获取数据':
        def get_ttm_data(self, col, table, codes, date, during_backtest=True):
            if table not in ['income', 'cashflow']:
                raise ValueError('get_ttm_data函数只支持利润表和现金流量表！')
            date_col = decide_date_col(table)
            if during_backtest:
                date = self.verify_date(date)
            
            if isinstance(codes, str):
                codes = [codes, 'just a place holder']
            elif isinstance(codes, list) and len(codes) == 1:
                codes.append('just a place holder')
            elif codes is None:
                codes = self.get_stock_info(codes=None).index.to_list()
            
            query = f"""
                    select * 
                    from {table} 
                    where ts_code in {tuple(codes)} 
                    and 
                    {date_col} <= '{date}'
                    """
            df = pd.read_sql(query, self.conn)
            cols_to_keep = get_primary_key(table, self.conn)
            if col not in cols_to_keep:
                cols_to_keep.append(col)
            if date_col not in cols_to_keep:
                cols_to_keep.append(date_col)

            df = df[cols_to_keep]
            df.set_index('ts_code', inplace=True)
            final_df = df.groupby('ts_code').last()
            ttm_col = f'{col}_ttm'
            final_df[ttm_col] = None
            if len(final_df) > 0:
                for code in final_df.index:
                    code_end_date = final_df.loc[code, 'end_date']
                    last_year_fourth_date = str(int(code_end_date[:4])-1) + '1231'
                    last_year_third_date = str(int(code_end_date[:4])-1) + '0930'
                    last_year_second_date = str(int(code_end_date[:4])-1) + '0630'
                    last_year_first_date = str(int(code_end_date[:4])-1) + '0331'
                    if code_end_date.endswith('1231'):
                        pass
                    elif code_end_date.endswith('0930'):
                        last_year_fourth_report = df[(df.index==code)&(df['end_date']==last_year_fourth_date)]
                        last_year_third_report = df[(df.index==code)&(df['end_date']==last_year_third_date)]
                        if len(last_year_fourth_report) > 0 and len(last_year_third_report) > 0:
                            last_year_4q_data = last_year_fourth_report[col].iloc[0] - last_year_third_report[col].iloc[0]
                            final_df.loc[code, ttm_col] = final_df.loc[code, col] + last_year_4q_data
                    elif code_end_date.endswith('0630'):
                        last_year_fourth_report = df[(df.index==code)&(df['end_date']==last_year_fourth_date)]
                        last_year_second_report = df[(df.index==code)&(df['end_date']==last_year_second_date)]
                        if len(last_year_fourth_report) > 0 and len(last_year_second_report) > 0:
                            last_year_3q_4q_data = last_year_fourth_report[col].iloc[0] - last_year_second_report[col].iloc[0]
                            final_df.loc[code, ttm_col] = final_df.loc[code, col] + last_year_3q_4q_data
                    elif code_end_date.endswith('0331'):
                        last_year_fourth_report = df[(df.index==code)&(df['end_date']==last_year_fourth_date)]
                        last_year_first_report = df[(df.index==code)&(df['end_date']==last_year_first_date)]
                        if len(last_year_fourth_report) > 0 and len(last_year_first_report) > 0:
                            last_year_2q_3q_4q_data = last_year_fourth_report[col].iloc[0] - last_year_first_report[col].iloc[0]
                            final_df.loc[code, ttm_col] = final_df.loc[code, col] + last_year_2q_3q_4q_data
            return final_df
                    

        def get_daily_data(self, col, table, codes, date, during_backtest=True):
            date_col = decide_date_col(table)
            if during_backtest:
                date = self.verify_date(date)

            codes = self.verify_codes(codes)
            
            query = f"""
                    select *
                    from {table} 
                    where ts_code in {tuple(codes)} 
                    and 
                    {date_col} <= '{date}'
                    """
            df = pd.read_sql(query, self.conn)
            if col is not None:
                df = df[['ts_code', date_col, col]]
            # cols_to_keep = get_primary_key(table, self.conn)
            # if col not in cols_to_keep:
            #     cols_to_keep.append(col)
            # if date_col not in cols_to_keep:
            #     cols_to_keep.append(date_col)
            # df = df[cols_to_keep]
            df = df.groupby('ts_code').last()
            return df
        
        def get_historical_data(self, col, table, codes, start_date, end_date, during_backtest=True):
            date_col = decide_date_col(table)
            if during_backtest:
                end_date = self.verify_date(end_date)
            
            codes = self.verify_codes(codes)

            query = f"""
                    select ts_code, {date_col}, {col} 
                    from {table} 
                    where ts_code in {tuple(codes)}
                    and 
                    {date_col} >= '{start_date}'
                    and 
                    {date_col} <= '{end_date}'
                    """
            df = pd.read_sql(query, self.conn)            
            return df
        
        def get_historical_report_period_data(self, col, table, codes, start_date, end_date, report_type, during_backtest=True):
            assert table in ['income', 'balancesheet', 'cashflow']
            assert report_type in [1, 2, 3, 4]
            date_col = decide_date_col(table)
            if during_backtest:
                end_date = self.verify_date(end_date)
            codes = self.verify_codes(codes)

            query = f"""
                    select ts_code, {date_col}, end_date, {col} 
                    from {table} 
                    where ts_code in {tuple(codes)}
                    and
                    {date_col} >= '{start_date}' 
                    and 
                    {date_col} <= '{end_date}'
                    and 
                    end_type = '{report_type}'
                    """
            df = pd.read_sql(query, self.conn)
            return df
        
        def get_daily_report_period_data(self, col, table, codes, date, report_type, during_backtest=True):
            assert table in ['income', 'balancesheet', 'cashflow']
            assert report_type in [1, 2, 3, 4]
            date_col = decide_date_col(table)
            if during_backtest:
                date = self.verify_date(date)
            codes = self.verify_codes(codes)
            
            query = f"""
                    select *
                    from {table} 
                    where ts_code in {tuple(codes)}
                    and
                    {date_col} <= '{date}'
                    and 
                    end_type = '{report_type}'
                    """
            df = pd.read_sql(query, self.conn)
            if col is not None:
                df = df[['ts_code', date_col, 'end_date', col]]
            df = df.groupby('ts_code').last()
            #df.set_index('ts_code', inplace=True)
            return df


        def get_stock_info(self, codes, date, during_backtest=True):
            if during_backtest:
                date = self.verify_date(date)
        
            if isinstance(codes, str):
                codes = [codes, 'just a place holder']
            elif isinstance(codes, list) and len(codes) == 1:
                codes.append('just a place holder')

            if codes is not None:
                query = f"""
                        select *
                        from stock_info 
                        where ts_code in {tuple(codes)}
                        and list_date <= '{date}'
                        """
            else:
                query = f"""
                        select *
                        from stock_info
                        where list_date <= '{date}'
                        """
            df = pd.read_sql(query, self.conn)
            df.set_index('ts_code', inplace=True)
            df.drop('INSERT_TIME', axis=1, inplace=True)
            return df
        
        def get_index_info(self, codes):
            if isinstance(codes, str):
                codes = [codes, 'just a place holder']
            elif isinstance(codes, list) and len(codes) == 1:
                codes.append('just a place holder')

            if codes is not None:
                query = f"""
                        select *
                        from index_info 
                        where ts_code in {tuple(codes)}
                        """
            else:
                query = f"""
                        select *
                        from index_info
                        """
            df = pd.read_sql(query, self.conn)
            df.set_index('ts_code', inplace=True)
            df.drop('INSERT_TIME', axis=1, inplace=True)
            return df
        
        def get_trade_cal(self, exchange, during_backtest=True):
            query = f"""select *
                        from trade_cal 
                        where exchange = '{exchange}'
                        """
            df = pd.read_sql(query, self.conn)
            if during_backtest:
                df = df[df['cal_date']<=self.current_date.strftime('%Y%m%d')]
            df.drop('INSERT_TIME', axis=1, inplace=True)
            return df
        
        def get_stock_trading_status(self, date, code, during_backtest=True):
            if during_backtest:
                date = self.verify_date(date)
            df = self.get_daily_data(col='pct_chg', table='stock_daily_price', codes=code, date=date)
            if len(df) == 0:
                return False
            # 这个语句要求前面必须执行verify_date，否则date为None的时候会误判为停牌
            elif df['trade_date'].iloc[0] != date:
                return False
            # 如果当天涨跌幅为0，或者涨跌幅绝对值超过9%，就判断为“停牌”
            elif df['pct_chg'].iloc[0] == 0 or abs(df['pct_chg'].iloc[0]) >= 9:
                    return False
            else:
                return True
        
        '''
        def get_index_component(self, index_code, date, during_backtest=True):
            if during_backtest:
                date = self.verify_date(date)
            # 如果当前日期下指数未上市或已退市，则返回错误
            index_info = self.get_index_info(index_code)
            list_date, exp_date = index_info.loc[index_code, 'list_date'], index_info.loc[index_code, 'exp_date']
            if date < list_date:
                raise ValueError(f'{index_code}指数上市日期为{list_date}，当前为{date}，无法获取成分股信息')
            if exp_date is not None and date > exp_date:
                raise ValueError(f'{index_code}指数已于{exp_date}退市，当前为{date}，无法获取成分股信息')

            # 判断数据库中的成分股数据是否需要更新
            need_update = False
            self.cursor.execute(f"""
                SELECT MAX(trade_date) 
                FROM index_component 
                WHERE index_code = '{index_code}'
            """)
            last_date = self.cursor.fetchone()[0]
            # 第一种情况，数据库中没有该指数的成分股数据，则需要更新
            if last_date is None:
                need_update = True
            else:
                # 第二种情况，数据库中有该指数的成分股数据，但最新日期距离当前日期超过60天，则需要更新
                date_diff = datetime.datetime.strptime(date, '%Y%m%d') - datetime.datetime.strptime(last_date, '%Y%m%d')
                if date_diff.days > 60:
                    need_update = True
            
            if need_update:
                # 如果last_date为None，则从上市日期开始获取
                if last_date is None:
                    last_date = list_date
                self.update_index_component(index_code, last_date)

            return self._get_index_component(index_code, date)
        '''
            
        def get_index_component(self, index_code, date, during_backtest=True):
            if during_backtest:
                date = self.verify_date(date)
            query = f"""
                SELECT * 
                FROM index_component 
                WHERE index_code = '{index_code}'
                AND trade_date = (
                    SELECT MAX(trade_date) 
                    FROM index_component 
                    WHERE index_code = '{index_code}'
                    AND trade_date <= '{date}'
                )
            """
            df = pd.read_sql(query, self.conn)
            df.drop('INSERT_TIME', axis=1, inplace=True)
            return df
        
        def get_historical_index_component(self, index_code, start_date, end_date):
            # 先读取数据库
            query = f"""
                SELECT * 
                FROM index_component 
                WHERE index_code = '{index_code}'
                AND trade_date >= '{start_date}'
                AND trade_date <= '{end_date}'
            """
            df = pd.read_sql(query, self.conn)
            df.drop('INSERT_TIME', axis=1, inplace=True)

            # 创建透视表
            all_codes = df['con_code'].unique()
            index_component_df = pd.DataFrame(index=pd.date_range(start_date, end_date), columns=all_codes).fillna(False)

            # 获取数据库中的所有交易日期
            all_trade_dates = df['trade_date'].unique()
            all_trade_dates.sort()
            # 每个交易日期和下一个交易日期之间，取前者的成分股，填充到透视表中
            for trade_date_index in tqdm(range(len(all_trade_dates))):
                this_trade_date = all_trade_dates[trade_date_index]
                if trade_date_index == len(all_trade_dates) - 1:
                    next_trade_date = datetime.datetime.strptime(end_date, '%Y%m%d')
                else:
                    next_trade_date = all_trade_dates[trade_date_index+1]
                this_to_next = pd.date_range(this_trade_date, next_trade_date, inclusive='left')
                con_codes = df[df['trade_date'] == this_trade_date]['con_code'].unique()
                index_component_df.loc[this_to_next, con_codes] = True
            return index_component_df
            
    if '更新数据':
        def update_index_component(self, index_code, last_date):
            # 从tushare获取数据，从last_date所在的月开始获取
            dfs = []
            for date in tqdm(pd.date_range(start=last_date, end=datetime.datetime.now(), freq='M'), desc=f'更新{index_code}成分股数据中'):
                first_day, last_day = get_month_bounds(date.strftime('%Y%m%d'))
                time1 = time.time()
                df = pro.index_weight(index_code=index_code, start_date=first_day, end_date=last_day, fields='index_code,con_code,trade_date,weight')
                dfs.append(df)
                time2 = time.time()
                iter_rate_limiter(400, time2-time1)
            df = pd.concat(dfs, axis=0)
            self.general_update_func(table='index_component', id=index_code, id_col='index_code', df=df)

        def general_update_func(self, table, id, id_col, df):
            if df is not None and len(df) > 0:
                # 获取表格的主键列，提取主键列的数据
                primary_key = get_primary_key(table, self.conn)

                # 如果指定了id和id_col，则只提取符合id的部分数据
                if id is not None and id_col is not None:
                    query = f"""
                            select {', '.join(primary_key)} 
                            from {table} 
                            where {id_col} = '{id}' 
                            """
                # 如果id和id_col均为None，则提取所有数据
                elif id is None and id_col is None:
                    query = f"""
                            select {', '.join(primary_key)} 
                            from {table}
                            """
                existing_data = pd.read_sql(query, self.conn)

                # 从df中删除已有的数据
                df_pk = df[primary_key]
                duplicates = df_pk[df_pk.apply(tuple, 1).isin(existing_data.apply(tuple, 1))]
                df = df.drop(duplicates.index)

                df.to_sql(table, self.conn, if_exists='append', index=False)

        def update_financial_statement(self, table, code, start_date):
            supported_tables = ['income', 'balancesheet', 'cashflow']
            if table not in supported_tables:
                raise ValueError(f'该方法支持的表名：{supported_tables}，指定表名{table}不在范围内')
            
            if pd.isna(start_date):
                start_date = '20000101'

            
            # 要指定字段，否则获取的数据不全
            fields = self.set_update_fields(table)

            # 获取最新数据
            local_scope = {}
            exec(f'df = pro.{table}(ts_code="{code}", start_date="{start_date}", fields="{fields}")', globals(), local_scope)
            df = local_scope['df']

            self.general_update_func(table=table, id=code, id_col='ts_code', df=df)

        def update_daily_price(self, table, code, start_date):
            supported_tables = ['stock_daily_price', 'stock_daily_price_hfq', 'index_daily_price']
            if table not in supported_tables:
                raise ValueError(f'该方法支持的表名：{supported_tables}，指定表名{table}不在范围内')
            
            if pd.isna(start_date):
                start_date = '20000101'
                            
            if table == 'stock_daily_price_hfq':
                df = ts.pro_bar(ts_code=code, adj='hfq', start_date=start_date)
            elif table == 'stock_daily_price':
                df = ts.pro_bar(ts_code=code, adj=None, start_date=start_date)
            elif table == 'index_daily_price':
                df = ts.pro_bar(ts_code=code, asset='I', start_date=start_date)

            self.general_update_func(table=table, id=code, id_col='ts_code', df=df)
        
        def update_stock_daily_basic(self, code, start_date):        
            fields = self.set_update_fields('stock_daily_basic')
            if pd.isna(start_date):
                start_date = '20000101'

            df = pro.daily_basic(ts_code=code, start_date=start_date, fields=fields)
            self.general_update_func(table='stock_daily_basic', id=code, id_col='ts_code', df=df)

        def update_trade_cal(self, exchange):
            df = pro.trade_cal(exchange=exchange)
            self.general_update_func(table='trade_cal', id=exchange, id_col='exchange', df=df)

        def update_info(self, table):
            if table == 'stock_info':
                fields = self.set_update_fields('stock_info')
                dfs = []
                for list_status in ['L', 'D', 'P']:
                    df = pro.stock_basic(exchange='', list_status=list_status, fields=fields)
                    dfs.append(df)
                df = pd.concat(dfs, axis=0)
                self.general_update_func(table=table, id=None, id_col=None, df=df)
            elif table == 'index_info':
                fields = self.set_update_fields('index_info')
                dfs = []
                for market in ['CSI', 'SSE', 'SZSE', 'CICC', 'SW', 'OTH']:
                    df = pro.index_basic(market=market, fields=fields)
                df = pd.concat(dfs, axis=0)
                self.general_update_func(table=table, id=None, id_col=None, df=df)


class Junnko_Backtest(Database):
    def __init__(self):
        super().__init__()
    
    def buy(self, code, shares):
        price = self.get_daily_data('close', 'stock_daily_price_hfq', code, None)
        if code in price.index:
            price = price.loc[code, 'close']
            money = price * shares * (1+self.commission_rate)
            if money > self.cash:
                money = self.cash
                shares = money / (price * (1+self.commission_rate))

            shares = shares//100*100
            if shares > 0:
                commission = shares*price*self.commission_rate
                self.cash -= shares*price
                self.cash -= commission

                if code in self.position.keys():
                    self.position[code] += shares
                else:
                    self.position[code] = shares
            
                print(f'{self.current_date} 买入{code} {shares}股 手续费{commission}')
    
    def sell(self, code, shares):
        if code in self.position.keys():
            current_position = self.position[code]
            if shares > current_position:
                shares = current_position
            
            shares = shares//100*100
            if shares > 0:
                price = self.get_daily_data('close', 'stock_daily_price_hfq', code, None)
                if code in price.index:
                    price = price.loc[code, 'close']
                    self.cash += shares*price
                    commision = shares*price*self.commission_rate
                    self.cash -= commision
                    self.position[code] -= shares
                    if self.position[code] == 0:
                        self.position.pop(code)
                    print(f'{self.current_date} 卖出{code} {shares}股 手续费{commision}')
    
    def order(self, code, shares):
        if self.get_stock_trading_status(code=code, date=None):
            if shares > 0:
                self.buy(code, shares)
            elif shares < 0:
                self.sell(code, -shares)
    
    def order_target_shares(self, code, target_shares):
        if code in self.position.keys():
            current_position = self.position[code]
        else:
            current_position = 0
        shares = target_shares - current_position
        self.order(code, shares)
    
    def order_target_value(self, code, target_value):
        if code in self.position.keys():
            current_position = self.position[code]
        else:
            current_position = 0
        if self.get_stock_trading_status(code=code, date=None):
            current_price = self.get_daily_data('open', 'stock_daily_price_hfq', code, None).loc[code, 'open']
            target_shares = target_value // current_price
            shares = target_shares - current_position
            self.order(code, shares)
    
    def calculate_net_value(self):
        net_value = self.cash
        for code in self.position.keys():
            net_value += self.position[code] * self.get_daily_data(col='close', table='stock_daily_price_hfq', codes=code, date = None).loc[code, 'close']
        return net_value

    def run_event_backtest(self, start_date, end_date, initial_capital, commission_rate, strategy):
        all_dates = pd.date_range(start_date, end_date)
        self.strategy = strategy
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.position = {}
        for i in tqdm(range(len(all_dates)), desc='回测进行中'):
            self.current_date = all_dates[i]
            self.strategy(self)
            net_value = self.calculate_net_value()
            yield i, self.cash, net_value
    
    def run_factor_backtest(self, start_date, end_date, freq, factor, num_groups, stock_pool):
        all_dates = pd.date_range(start_date, end_date)

        print('正在生成股票池')
        stock_pool_pivot_df = self.get_historical_index_component(stock_pool, start_date, end_date)
        all_codes = stock_pool_pivot_df.columns.to_list()
        factor = factor[[c for c in all_codes if c in factor.columns]]

        print('正在获取股票日收益率')
        query = f'select * from stock_daily_price_hfq where ts_code in {tuple(all_codes)}'
        stock_daily_ret = pd.read_sql(query, self.conn)
        stock_daily_ret = stock_daily_ret.pivot_table(values='close', index='trade_date', columns='ts_code').pct_change()


        # 处理缺失日期
        stock_daily_ret.index = pd.to_datetime(stock_daily_ret.index)
        stock_daily_ret = stock_daily_ret.reindex(all_dates, method=None).fillna(0)
        factor.index = pd.to_datetime(factor.index)
        factor = factor.reindex(all_dates, method='ffill')

        
        print('正在生成分组持仓')
        factor = factor[stock_pool_pivot_df]
        factor_rank = factor.rank(pct=True, axis=1)
        cut = 1 / num_groups
        positions = {}
        for group in tqdm(range(num_groups)):
            position = pd.DataFrame(index=all_dates, columns=all_codes).fillna(False)
            if group == 0:
                position[(factor_rank >= cut*group) & (factor_rank <= cut*(group+1))] = True
            else:
                position[(factor_rank > cut*group) & (factor_rank <= cut*(group+1))] = True
            positions[group] = position
        
        if freq != 'D':
            print('正在按持仓频率调整持仓')
            for group in tqdm(range(num_groups)):
                position = positions[group]
                positions[group] = position.resample(freq, label='left').first().reindex(position.index, method='ffill')


        print('正在计算分组收益')
        group_rets = pd.DataFrame(index=all_dates, columns=list(range(num_groups)))
        for group in tqdm(range(num_groups)):
            group_ret = stock_daily_ret.loc[start_date:end_date][positions[group]].mean(axis=1, skipna=True)
            group_rets[group] = group_ret
        return group_rets
