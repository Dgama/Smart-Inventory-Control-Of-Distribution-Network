import pandas as pd
import numpy as np

sku_demand_distribution = pd.read_csv('sku_demand_distribution.csv')
np.random.seed(3)

days=30
number_of_sample = 5
input_flag=True
dc_list=list(range(6))
sku_list=list(range(1000))

while(input_flag):
        #生成需求
        whether_random=input("Do you want to create a random demand(N for the original demand given by simulation)(Y/N)\n")
        df_demand = sku_demand_distribution.sort_values(['dc_id', 'item_sku_id'])
        para1 = np.asarray([df_demand.loc[df_demand.dc_id == dcid].para1.values for dcid in dc_list])
        para2 = np.asarray([df_demand.loc[df_demand.dc_id == dcid].para2.values for dcid in dc_list])
        if whether_random=='Y':
            print("please wait...")
            demand_sample_all = np.zeros(shape=(number_of_sample, 30, 6, 1000))
            input_flag=False
            for sample in range(0,number_of_sample):
                for day in range(0,days):
                    for dcid in range(0,len(dc_list)):
                        for skuid in range(0,len(sku_list)):
                            x = df_demand.loc[(df_demand.dc_id == dcid) & (df_demand.item_sku_id == skuid + 1)]
                            if (x["dist_type"].values == 'G'):
                                demand_sample_all[sample][day][dcid][skuid]=(np.random.gamma(para1[dcid][skuid],para2[dcid][skuid],1))
                            else:
                                demand_sample_all[sample][day][dcid][skuid] = (np.random.negative_binomial(para1[dcid][skuid],para2[dcid][skuid], 1))
        elif whether_random=='N':
            demand_sample_all = np.random.randint(10, size=(number_of_sample, days, len(dc_list), len(sku_list)))
            input_flag=False
        else:
            print("No,not like this,please enter again")

with open('demand.txt','w') as outfile:
    for sp in range(0,len(demand_sample_all)):
        for daily in demand_sample_all[sp]:
            np.savetxt(outfile,daily)


