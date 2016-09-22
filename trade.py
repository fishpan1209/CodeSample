import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from pylab import *
import numpy as np
import math
import LinRegLearner as lrl
import KNNLearner as knn
import BagLearner as bl

from util import get_data, plot_data
from portfolio.analysis import get_portfolio_value, get_portfolio_stats, plot_normalized_data
from marketsim import compute_portvals
from pandas.tseries.offsets import *


# set input pamareters
start_date = dt.datetime(2009,1,31)
end_date = dt.datetime(2009,5,31)
start_val=10000
timerange = pd.date_range(start_date, end_date)
symbols = ['ML4T-399']
close = get_data(symbols, timerange, addSPY=False)

# drop null values
close = close.dropna()
order = pd.DataFrame('NA',index=close.index, columns=['Order'])

#compute rs
delta = close['ML4T-399'].diff()
dUp, dDown = delta.copy(),delta.copy()
dUp[dUp<0]=0
dDown[dDown>0]=0
RolUp = pd.rolling_mean(dUp,20)
RolDown=pd.rolling_mean(dDown,20).abs()
RS = RolUp/RolDown

#compute sma, volatility, std, momentum, rsi features
df=close['ML4T-399'].copy()
daily_ret=df/df.shift(1)-1
close['SMA'] = pd.rolling_mean(close['ML4T-399'],20)
close['volatility']=pd.rolling_std(daily_ret,20)
close['STD']=pd.rolling_std(close['ML4T-399'],20)
close['bb_value']=(close['ML4T-399']-close['SMA'])/(pd.rolling_std(close['ML4T-399'],20)*2)
close['momentum']=close['ML4T-399']/close['ML4T-399'].shift(5)-1
close['RSI']=RS
close['Y']=df.shift(-5)/df-1
close=close.dropna()


# train data
trainX = close[['bb_value','momentum','RSI','STD']]

trainY = close['Y']


#learner = knn.KNNLearner(k=3) # create a KNN learner
        
learner = lrl.LinRegLearner() # create a LinRegLearner
learner.addEvidence(trainX, trainY) # train
Y = learner.query(trainX) # get the predictions

rmse = math.sqrt(((trainY - Y) ** 2).sum()/trainY.shape[0])
print
print "In sample results"
print "RMSE: ", rmse
c = np.corrcoef(Y, y=trainY)
print "corr: ", c[0,1]

# plot pred price
predY = pd.DataFrame(Y,index=trainY.index,columns=['predY'])
predPrice = close['ML4T-399']*(1+predY)
trainPrice = close['ML4T-399']*(1+trainY)
plt.clf()
plt.plot(close.index,close['ML4T-399'])
plt.plot(trainPrice.index,trainPrice)
plt.plot(predPrice.index,predPrice)

plt.ylabel('Price')
plt.xlabel('Date')
plt.title('ML4T-399-linear regression_train')
plt.legend(('ML4T-399','trainPrice','predPrice'),loc='lower right')
savefig('ML4T-399_lrg_FIG1.pdf',format='pdf')


order = pd.DataFrame('NA',index=close.index, columns=['Order'])
close['share']=0
close['holding']=0
close['value']=0
close['cash']=start_val
close['total']=0
short_entry=0
long_entry=0

plt.clf()
plt.plot(close.index,close['ML4T-399'])
plt.ylabel('Price')
plt.xlabel('Date')
plt.title('ML4T-399-linear regression_train')

for i in range(1,close.index.size):
        

        # long entry
        if predY['predY'][i]>=0.01 and long_entry==0:
                plt.axvline(x=close.index[i],c='green')
                close.set_value(close.index[i],'share',close['share'][i]+100)
                spent=close['ML4T-399'][i]*100
                close.set_value(close.index[i],'cash',close['cash'][i-1]-spent)
                order.set_value(close.index[i],'Order','LONG_ENTRY')
                long_entry=i
                """
                print 'Long Entry: ',long_entry
                print close.ix[close.index[i]]
                print '                                     '
                """



        # long exit after 5 days
        elif long_entry>0 and (i-long_entry)>5 and predY['predY'][i]<0:
                plt.axvline(x=close.index[i],c='black')
                close.set_value(close.index[i],'share',close['share'][i]-100)
                gain=close['ML4T-399'][i]*100
                close.set_value(close.index[i],'cash',close['cash'][i-1]+gain)
                order.set_value(close.index[i],'Order','LONG_EXIT')
                long_entry=0
                """
                print 'Long Exit: ',long_entry
                print close.ix[close.index[i]]
                print '                        '
                """



        # short entry
        elif predY['predY'][i]<=-0.01 and short_entry==0 :
                plt.axvline(x=close.index[i],c='red')
                close.set_value(close.index[i],'share',close['share'][i]-100)
                gain=close['ML4T-399'][i]*100
                close.set_value(close.index[i],'cash',close['cash'][i-1]+gain)
                order.set_value(close.index[i],'Order','SHORT_ENTRY')
                short_entry=i
                """
                print 'Short Entry: ',short_entry
                print close.ix[close.index[i]]
                print '                        '
                """

        # SHORT EXIT
        elif short_entry>0 and (i-short_entry>5) and predY['predY'][i]>0:
                plt.axvline(x=close.index[i],c='black')
                close.set_value(close.index[i],'share',close['share'][i]+100)
                spent=close['ML4T-399'][i]*100
                close.set_value(close.index[i],'cash',close['cash'][i-1]-spent)
                order.set_value(close.index[i],'Order','SHORT_EXIT')
                short_entry=0
                """
                print 'Short Exit: ',short_entry
                print close.ix[close.index[i]]
                print '                        '
                """

        else:
            close.set_value(close.index[i],'cash',close['cash'][i-1])




        close['holding']=close['share'].cumsum()
        close['value']=close['holding']*close['ML4T-399']
        close['total']=close['value']+close['cash']
      
close['holding']=close['share'].cumsum()
close['value']=close['holding']*close['ML4T-399']
close['total']=close['value']+close['cash']     
 
order=order[order.Order!='NA']  
#print order 
order.to_csv('ML4T-399_order.csv', sep=',',index_label='Dates')
savefig('ML4T-399_lrg_Strategy.pdf',format='pdf')

portvals=close['total']
cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)
prices_SPX = get_data(['$SPX'], pd.date_range(start_date, end_date))
prices_SPX=prices_SPX.join(close,how='right')
prices_SPX = prices_SPX[['$SPX']]  # remove SPY

portvals_SPX = get_portfolio_value(prices_SPX, [1.0])
cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = get_portfolio_stats(portvals_SPX)

    # Compare portfolio against $SPX
print "Data Range: {} to {}".format(start_date, end_date)
print
print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
print "Sharpe Ratio of $SPX: {}".format(sharpe_ratio_SPX)
print
print "Cumulative Return of Fund: {}".format(cum_ret)
print "Cumulative Return of $SPX: {}".format(cum_ret_SPX)
print
print "Standard Deviation of Fund: {}".format(std_daily_ret)
print "Standard Deviation of $SPX: {}".format(std_daily_ret_SPX)
print
print "Average Daily Return of Fund: {}".format(avg_daily_ret)
print "Average Daily Return of $SPX: {}".format(avg_daily_ret_SPX)
print
print "Final Portfolio Value: {}".format(portvals[-1])

    # Plot computed daily portfolio value
df_temp = pd.concat([portvals, prices_SPX['$SPX']], keys=['Portfolio', '$SPX'], axis=1)
plot_normalized_data(df_temp, title="Daily portfolio value and $SPX")
savefig('ML4T-399_lrg_backtest.pdf',format='pdf')


# test
start_date = dt.datetime(2010,1,31)
end_date = dt.datetime(2010,5,31)
start_val=10000
timerange = pd.date_range(start_date, end_date)
symbols = ['ML4T-399']
close = get_data(symbols, timerange, addSPY=False)

close = close.dropna()
order = pd.DataFrame('NA',index=close.index, columns=['Order'])

delta = close['ML4T-399'].diff()
dUp, dDown = delta.copy(),delta.copy()
dUp[dUp<0]=0
dDown[dDown>0]=0
RolUp = pd.rolling_mean(dUp,20)
RolDown=pd.rolling_mean(dDown,20).abs()
RS = RolUp/RolDown

df=close['ML4T-399'].copy()
daily_ret=df/df.shift(1)-1
close['SMA'] = pd.rolling_mean(close['ML4T-399'],20)
close['volatility']=pd.rolling_std(daily_ret,20)
close['STD']=pd.rolling_std(close['ML4T-399'],20)
close['bb_value']=(close['ML4T-399']-close['SMA'])/(pd.rolling_std(close['ML4T-399'],20)*2)
close['momentum']=close['ML4T-399']/close['ML4T-399'].shift(5)-1
close['RSI']=RS
close['Y']=df.shift(-5)/df-1
close=close.dropna()




testX = close[['bb_value','momentum','RSI','STD']]

testY = close['Y']


#learner = knn.KNNLearner(k=3) # create a KNN learner
        
#learner = lrl.LinRegLearner() # create a LinRegLearner
#learner.addEvidence(trainX, trainY) # train
Y = learner.query(testX) # get the predictions

rmse = math.sqrt(((testY - Y) ** 2).sum()/testY.shape[0])
print
print "Out of sample results"
print "RMSE: ", rmse
c = np.corrcoef(Y, y=testY)
print "corr: ", c[0,1]


predY = pd.DataFrame(Y,index=testY.index,columns=['predY'])
predPrice = close['ML4T-399']*(1+predY)
testPrice = close['ML4T-399']*(1+testY)
plt.clf()
plt.plot(close.index,close['ML4T-399'])
plt.plot(testPrice.index,testPrice)
plt.plot(predPrice.index,predPrice)

plt.ylabel('Price')
plt.xlabel('Date')
plt.title('ML4T-399-linear regression_test')
plt.legend(('ML4T-399','testPrice','predPrice'),loc='lower right')
savefig('ML4T-399_lrg_FIG2.pdf',format='pdf')


order = pd.DataFrame('NA',index=close.index, columns=['Order'])
close['share']=0
close['holding']=0
close['value']=0
close['cash']=start_val
close['total']=0
short_entry=0
long_entry=0

plt.clf()
plt.plot(close.index,close['ML4T-399'])
plt.ylabel('Price')
plt.xlabel('Date')
plt.title('ML4T-399-linear regression_test')

# create trading strategy

for i in range(1,close.index.size):
        

        # long entry
        if predY['predY'][i]>=0.01 and long_entry==0 :
                plt.axvline(x=close.index[i],c='green')
                close.set_value(close.index[i],'share',close['share'][i]+100)
                spent=close['ML4T-399'][i]*100
                close.set_value(close.index[i],'cash',close['cash'][i-1]-spent)
                order.set_value(close.index[i],'Order','LONG_ENTRY')
                long_entry=i
                """
                print 'Long Entry: ',long_entry
                print close.ix[close.index[i]]
                print '                                     '
                """



        # long exit after 5 days
        elif long_entry>0 and (i-long_entry)>5 and predY['predY'][i]<0:
                plt.axvline(x=close.index[i],c='black')
                close.set_value(close.index[i],'share',close['share'][i]-100)
                gain=close['ML4T-399'][i]*100
                close.set_value(close.index[i],'cash',close['cash'][i-1]+gain)
                order.set_value(close.index[i],'Order','LONG_EXIT')
                long_entry=0
                """
                print 'Long Exit: ',long_entry
                print close.ix[close.index[i]]
                print '                        '
                """



        # short entry
        elif predY['predY'][i]<=-0.01 and short_entry==0:
                plt.axvline(x=close.index[i],c='red')
                close.set_value(close.index[i],'share',close['share'][i]-100)
                gain=close['ML4T-399'][i]*100
                close.set_value(close.index[i],'cash',close['cash'][i-1]+gain)
                order.set_value(close.index[i],'Order','SHORT_ENTRY')
                short_entry=i
                """
                print 'Short Entry: ',short_entry
                print close.ix[close.index[i]]
                print '                        '
                """

        # SHORT EXIT
        elif short_entry>0 and (i-short_entry>5) and predY['predY'][i]>0:
                plt.axvline(x=close.index[i],c='black')
                close.set_value(close.index[i],'share',close['share'][i]+100)
                spent=close['ML4T-399'][i]*100
                close.set_value(close.index[i],'cash',close['cash'][i-1]-spent)
                order.set_value(close.index[i],'Order','SHORT_EXIT')
                short_entry=0
                """
                print 'Short Exit: ',short_entry
                print close.ix[close.index[i]]
                print '                        '
                """

        else:
            close.set_value(close.index[i],'cash',close['cash'][i-1])




        close['holding']=close['share'].cumsum()
        close['value']=close['holding']*close['ML4T-399']
        close['total']=close['value']+close['cash']
      
close['holding']=close['share'].cumsum()
close['value']=close['holding']*close['ML4T-399']
close['total']=close['value']+close['cash']     

order=order[order.Order!='NA']  
#print order 
order.to_csv('ML4T-399_order2.csv', sep=',',index_label='Dates')
savefig('ML4T-399_lrg_Strategy2.pdf',format='pdf')

portvals=close['total']
cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)
prices_SPX = get_data(['$SPX'], pd.date_range(start_date, end_date))
prices_SPX=prices_SPX.join(close,how='right')
prices_SPX = prices_SPX[['$SPX']]  # remove SPY
portvals_SPX = get_portfolio_value(prices_SPX, [1.0])
cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = get_portfolio_stats(portvals_SPX)

    # Compare portfolio against $SPX
print "Data Range: {} to {}".format(start_date, end_date)
print
print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
print "Sharpe Ratio of $SPX: {}".format(sharpe_ratio_SPX)
print
print "Cumulative Return of Fund: {}".format(cum_ret)
print "Cumulative Return of $SPX: {}".format(cum_ret_SPX)
print
print "Standard Deviation of Fund: {}".format(std_daily_ret)
print "Standard Deviation of $SPX: {}".format(std_daily_ret_SPX)
print
print "Average Daily Return of Fund: {}".format(avg_daily_ret)
print "Average Daily Return of $SPX: {}".format(avg_daily_ret_SPX)
print
print "Final Portfolio Value: {}".format(portvals[-1])

    # Plot computed daily portfolio value
df_temp = pd.concat([portvals, prices_SPX['$SPX']], keys=['Portfolio', '$SPX'], axis=1)
plot_normalized_data(df_temp, title="Daily portfolio value and $SPX")
savefig('ML4T-399_lrg_backtest2.pdf',format='pdf')
