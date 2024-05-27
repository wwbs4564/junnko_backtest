# junnko_backtest_readme

# 项目介绍

junnko_backtest是一个小白友好的量化回测框架，它具有如下特点：

1. **图形界面**。本项目基于streamlit开发了一个图形界面，目的在于最大程度减少用户的编程负担和编程环境配置负担，检视数据、测试函数、调整回测参数、编写策略代码、查看回测结果等功能均能够在图形界面中实现。
2. 同时兼容**事件化回测+向量化回测**。现有的回测框架大多只支持一种回测模式，而本项目同时兼容两种回测模式，意在扩大适用范围，并提升回测效率（例如，可以在向量化回测中先粗略测试因子的有效性，而后在事件化回测中编写更复杂的策略）。
3. **本地化的“在线”数据库**。本项目提供了一个基于sqlite3的本地数据库，并且设计了一套与tushare数据库对接的数据更新函数，用户只需要点击几个按钮，就可以将tushare数据库中的最新数据下载到本地，实现了鱼（在线数据库的数据及时性）和熊掌（本地数据库的高速、自主可控）的兼得。

# 安装

1. 克隆项目到本地：

​`git clone junnko_backtest`​

2. 安装依赖：

​`pip install -r requirements.txt`​

3. 在项目文件夹中新建一个`token.txt`​，将你的tushare token填入其中
4. 运行`初始化.ipynb`​文件

> 注1：这一步的目的是建立本地数据库，并将tushare数据库的最新数据批量保存到本地，方便后续回测。
>
> 注2：指数日行情（index_daily_price）和指数成分（index_component）的数据量过大，因此请后续在图形化界面中自行选择需要的进行下载。
>
> 注3：由于获取的数据量较大，且tushare的某些接口本身需要较高积分，你也可以选择只运行**建表**部分的代码，后续在图形化界面中下载你想要的数据。

5. 运行项目：双击`run.bat`​文件，浏览器上会自动打开新窗口。

> 也可以打开cmd，定位工作目录到项目文件夹，输入`streamlit run app.py`​。

# 图形界面使用说明

图形界面包含4个标签页：事件化回测、向量化回测、数据检视、调试。

## 事件化回测

请自行探索，图形界面已经足够完善，无需额外说明。

## 向量化回测

* 上传因子文件：parquet格式（使用pandas的to_parquet函数即可），列名为股票代码，行索引为日期（`str`​，`yyyymmdd`​格式）。

## 数据检视

请自行探索，图形界面已经足够完善，无需额外说明。

# 调试

在这里，你可以调用数据获取函数，以查看它们的返回值格式。注意以下两点：

1. 保证你想查看的最终输出结果被赋值到`test_result`​上
2. 保证during_backtest参数为False

# 数据获取函数

为了在编写策略和设计因子时调用本项目的数据获取函数，你首先需要知道，本项目的数据库目前包含以下几个表：

|表名|表含义|对应的tushare文档页|对应的函数|
| -------------------| ----------------------| ---------------------| ------------|
|stock_info|股票基本信息|[Tushare数据](https://tushare.pro/document/2?doc_id=25)||
|index_info|指数基本信息|[Tushare数据](https://tushare.pro/document/2?doc_id=94)<br />||
|stock_daily_price|股票日行情（前复权）|[Tushare数据](https://tushare.pro/document/2?doc_id=146)||
|stock_daily_basic|股票日指标|[Tushare数据](https://tushare.pro/document/2?doc_id=32)||
|index_daily_price|指数日行情|[Tushare数据](https://tushare.pro/document/2?doc_id=146)||
|income|利润表|[Tushare数据](https://tushare.pro/document/2?doc_id=33)||
|balancesheet|资产负债表|[Tushare数据](https://tushare.pro/document/2?doc_id=36)||
|cashflow|现金流量表|[Tushare数据](https://tushare.pro/document/2?doc_id=44)||
|trade_cal|交易日历|[Tushare数据](https://tushare.pro/document/2?doc_id=26)||
|index_component|指数成分股|[Tushare数据](https://tushare.pro/document/2?doc_id=96)||

在调用函数时，参数`table`​的值应从第一列中选择，参数`col`​的值应从第3列的网页中选择。

某些表有单独的数据获取函数（如指数成分股数据，有单独的`get_index_component函数`​），除非你熟悉函数的内部逻辑（参见`junnko_backtest.py`​）和表的结构（参见`初始化.ipynb`​中的建表部分），否则不要使用通用的数据获取函数去获取这些表的数据。

注意：在调用任何数据获取函数之前，最好先在图形界面中将对应数据更新。

## 获取单日数据

​`def get_daily_data(self, col, table, codes, date, during_backtest=True):`​

* col：想获取的字段名
* table：字段所在的表名
* codes：想获取的股票代码，支持以下3种情况

  * 一个`str`​格式的股票代码
  * 一个`list`​，包含多个`str`​格式的股票代码
  * ​`None`​，这样会获取所有股票的数据
* date：想获取的日期，`str`​、`yyyymmdd`​格式

  * 注：如果数据库中不存在与`date`​参数完全一致的数据，则会返回离`date`​最近的数据。
* during_backtest：该参数的作用是防止未来函数，分以下两种情况：

  * 在进行**事件化回测**时，请务必不要指定该参数，或者一律指定为`True`​。
  * 在其他情况下，请务必指定该参数的值为`False`​。
* 返回值：本文档目前不提供返回值说明，请在图形界面中的“调试”页，或者自行在python中实例化一个`Junnko_Backtest`​对象后，运行相关函数，查看返回值。

## 获取多日数据

​`def get_historical_data(self, col, table, code, start_date, end_date, during_backtest=True):`​

* col、table、during_backtest：同上
* code：想获取的股票代码，支持以下2种情况：

  * 一个`str`​格式的股票代码
  * ​`None`​，这样会获取所有股票的数据
* start_date、end_date：数据的开始和结束日期，`str`​、`yyyymmdd`​格式。

## 获取股票基本信息

​`def get_stock_info(self, codes):`​

* codes：同`get_daily_data`​。

## 获取指数基本信息

​`def get_index_info(self, codes):`​

同上。

## 获取交易日历

​`def get_trade_cal(self, exchange, during_backtest=True):`​

* exchange：交易所代码，目前只支持SSE、SZSE
* during_backtest：同上

## 判断股票是否停牌

​`def get_stock_trading_status(self, date, code, during_backtest=True):`​

同上。

目前的判断逻辑似乎不是特别严谨（参见`junnko_backtest.py`​）

## 获取指数成分股

​`def get_index_component(self, index_code, date, during_backtest=True):`​

* index_code：指数代码，`str`​格式
* date，during_backtest：同上

## 获取指数历史成分股

​`def get_historical_index_component(self, index_code, start_date, end_date):`​

参数同上。

这一函数设计的初衷是为了在向量化回测时生成股票池，在事件化回测时建议使用`get_index_component`​函数。

# 交易函数

## 基础交易函数

​`def order(self, code, shares):`​

* code：想交易的股票代码。
* shares：想买入\卖出的股数（卖出用负数表示）

## 调整持仓到指定股数

​`def order_target_shares(self, code, target_shares):`​

* code：同上
* target_shares：想调整到的股数。例如，目前持仓A股票300股，`target_shares=200`​会卖出A股票100股。

## 调整持仓到指定价值

​`def order_target_value(self, code, target_value):`​

* code：同上
* target_value：想调整到的价值。例如，当前持仓A股票1000股，A股票在当天的**开盘价**为1元，`target_value=500`​会卖出A股票500股。

‍
