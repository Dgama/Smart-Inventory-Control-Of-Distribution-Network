#!/usr/bin/env python
# -*- coding:UTF-8 -*-
'''
第七小组：高照、胡誉闻、杨祉煜、张乘源、宋图乾
代码：胡誉闻
'''

# import all modules been used
import pandas as pd
import numpy as np
from gurobipy import *
from scipy.stats import gamma,nbinom
import time

class UserPolicy:
    def __init__(self, initial_inventory, inventory_replenishment, sku_demand_distribution, sku_cost):
        self.inv = [initial_inventory]
        self.replenish = inventory_replenishment
        self.distribution = sku_demand_distribution
        self.cost = sku_cost
        self.sku_limit = np.asarray([200, 200, 200, 200, 200])
        self.extra_shipping_cost_per_unit = 0.01
        self.capacity_limit = np.asarray([3200, 1600, 1200, 3600, 1600])
        self.abandon_rate =np.asarray([1./100, 7./100, 10./100, 9./100, 8./100])

        #自己定义的用到的值
        self.abandoned_num=[]
        self.stock_out_deduct_abandon=[]
        self.stock_out_deduct_abandon_and_SR=[]
        self.q=0.01#履约成本
        self.bigM=1000000
        self.arrayinv=[]#array形式的库存
        self.dc_list=list(range(6))
        self.sku_list=list(range(1000))
        self.days=30
        self.total_cost=0
        self.ABC_boundary=[100,400,1000]#ABC个数划分
        self.beta_ABC=[0.7,0.7,0.7]#ABC三个等级货物beta划分值
        self.gamma_ABC=[0.95,0.95,0.95]#ABC三个等级货物gamma划分值


        print("Initializing UserPolicy,Please wait..")

        #将pandas转化为array
        self.arraylize(self.cost,self.replenish)

        #计算分布的系数
        (self.df_dist,self.para1,self.para2)=self.caculate_paras(self.distribution)

        #区分ABC货物
        self.sku_type=self.sku_ABC(self.cost,self.ABC_boundary)

        #得到每个商品在不同dc对应的Z(beta),Z(gama)
        (self.reorder_point,self.inventory_level)=self.caculate_reorder_point_and_inventory_level(self.df_dist,self.para1,self.para2,self.beta_ABC,self.gamma_ABC,self.sku_type)

        #计算各个物品补货的天数
        self.replenish_date_of_sku=self.caculate_replenish_date_of_sku(self.replenish)

        #计算每天的保留量
        self.RDC_reservation=self.caculate_rdc_reservation_of_each_day(self.df_dist,self.para1,self.para2,self.replenish_date_of_sku)
        print("Now,Solving ...")

    def daily_decision(self,t):
        '''
        daily decision of inventory allocation
        input values:
            t: decision date
        return values:
            allocation decision, 2-D numpy array, shape (5,1000), type integer
        '''

        (available_inventory,demand_of_today)=self.caculate_available_inventory_and_demand(self.arrayinv[t-1],self.RDC_reservation[t-1],self.reorder_point,self.inventory_level)
        transshipment_decision=self.optimize_model(t,available_inventory,demand_of_today)
        return transshipment_decision

    def info_update(self,end_day_inventory,t):
        '''
        input values: inventory information at the end of day t
        '''
        self.inv.append(end_day_inventory)
        self.transform_and_add_inv_to_array(end_day_inventory)

    def arraylize(self,sku_cost,replenish):
        """转化各种pandas为array"""
        # 初始化初始库存到array中
        self.transform_and_add_inv_to_array(self.inv[0])

        # 初始化缺货成本存到array中
        df_cost = sku_cost.sort_values('item_sku_id', ascending=True)
        self.arraycost = df_cost.stockout_cost.values

        # 初始化更新到array中
        df_rep = pd.DataFrame([[d, skuid] for d in range(1, self.days + 1) for skuid in self.sku_list],
                              columns=['date', 'item_sku_id'])
        df_rep = pd.merge(df_rep, replenish[['item_sku_id', 'date', 'replenish_quantity']],
                          on=['date', 'item_sku_id'], how='left')
        df_rep.fillna(value=0, inplace=True)
        self.arrayreplenish = np.asarray(
            [df_rep.loc[df_rep.date == d].replenish_quantity.values for d in range(1, self.days + 1)], dtype=int)

    def caculate_paras(self,distribution):
        """计算分布的系数，方便后面使用"""
        df_dist = distribution.sort_values(['dc_id', 'item_sku_id'])
        para1 = np.asarray([df_dist.loc[df_dist.dc_id == dcid].para1.values for dcid in self.dc_list])
        para2 = np.asarray([df_dist.loc[df_dist.dc_id == dcid].para2.values for dcid in self.dc_list])

        return df_dist,para1,para2

    def transform_and_add_inv_to_array(self,inventory):
        '''transform initial_inventory into numpy array, shape (6,1000) '''
        df_inv = inventory.sort_values(['dc_id', 'item_sku_id'])
        self.arrayinv.append(np.asarray([df_inv.loc[df_inv.dc_id == dcid].stock_quantity.values for dcid in self.dc_list]))

    def sku_ABC(self,sku_cost,ABC_boundary):
        """传入有序sku成本表，返回有序sku分类表,0,1,2,分别代表ABC"""
        ABC_cost = sku_cost.sort_values('item_sku_id', ascending=True)
        sku_type = ABC_cost.stockout_cost.values
        sku_ranking_index=sku_type[::-1].argsort()
        for a in range(0,ABC_boundary[0]):
            sku_type[sku_ranking_index[a]]=0
        for b in range(ABC_boundary[0],ABC_boundary[1]):
            sku_type[sku_ranking_index[b]]=1
        for c in range(ABC_boundary[1],ABC_boundary[2]):
            sku_type[sku_ranking_index[c]]=2
        sku_type=sku_type.astype(np.int16)
        return sku_type

    def caculate_reorder_point_and_inventory_level(self,df_dist,para1,para2,betas,gammas,type):
        """计算每个产品，每个地方需求分布的对应分位数值"""
        # reorder_point 和inventory_level 都是一个(6,1000)
        reorder_point=np.zeros((6,1000))
        inventory_level = np.zeros((6, 1000))
        for i in range(6):
            for j in range(1000):
                x = df_dist.loc[(df_dist.dc_id == i) & (df_dist.item_sku_id == j + 1)]
                if (x["dist_type"].values == 'G'):
                    reorder_point[i][j]=np.around(gamma.isf(1-betas[type[j]],para1[i][j],scale=para2[i][j]))
                    inventory_level[i][j]=np.around(gamma.isf(1-gammas[type[j]],para1[i][j],scale=para2[i][j]))
                else:
                    reorder_point[i][j] = np.around(nbinom.isf(1 - betas[type[j]], para1[i][j],para2[i][j]))
                    inventory_level[i][j] = np.around(nbinom.isf(1 - gammas[type[j]], para1[i][j],para2[i][j]))
        return reorder_point,inventory_level

    def caculate_replenish_date_of_sku(self,replenish):
        """计算不同货物的补货天数，每个货物的补货天数不一定相同"""

        ABC_rep = pd.DataFrame([[d, skuid] for d in range(1, self.days + 1) for skuid in self.sku_list],
                              columns=['date', 'item_sku_id'])
        ABC_rep = pd.merge(ABC_rep, replenish[['item_sku_id', 'date', 'replenish_quantity']],
                          on=['date', 'item_sku_id'], how='left')
        ABC_rep.fillna(value=0, inplace=True)
        ABCreplenish = np.asarray(
            [ABC_rep.loc[ABC_rep.date == d].replenish_quantity.values for d in range(1, self.days + 1)], dtype=int)
        date_of_replenish=[[0,30] for i in self.sku_list]

        for day in range(30):
            for sku in self.sku_list:
                if ABCreplenish[day][sku]!=0:
                    date_of_replenish[sku].insert(-1,day+1)
        return date_of_replenish

    def caculate_rdc_reservation_of_each_day(self,df_reserve,para1_RDC,para2_RDC,replenish_date):
        """计算每天RDC保留量"""

        reservation_rate=np.zeros((30,1000))
        reservation_quantity_of_each_today=np.zeros((30,1000))
        cumulative_reservation_quantity=np.zeros((30,1000))

        for skuid in self.sku_list:
            for times in range(0,len(replenish_date[skuid])-1):
                for day in range(replenish_date[skuid][times],replenish_date[skuid][times+1]):

                    reservation_rate[day][skuid]=(replenish_date[skuid][times+1]-day)*0.01+0.5
                    x = df_reserve.loc[(df_reserve.dc_id == 0) & (df_reserve.item_sku_id == skuid + 1)]
                    if (x["dist_type"].values == 'G'):
                        reservation_quantity_of_each_today[day][skuid]=np.around(gamma.isf(1-reservation_rate[day][skuid],para1_RDC[0][skuid],scale=para2_RDC[0][skuid]))
                    else:
                        reservation_quantity_of_each_today[day][skuid] = np.around(nbinom.isf(1 - reservation_rate[day][skuid], para1_RDC[0][skuid],para2_RDC[0][skuid]))

                for day in range(replenish_date[skuid][times], replenish_date[skuid][times + 1]):
                    cumulative_reservation_quantity[day][skuid]=sum(reservation_quantity_of_each_today.T[skuid][day:replenish_date[skuid][times+1]])

        return cumulative_reservation_quantity

    def create_demand(self):
        """生成需求"""
        demand_of_today = np.zeros((6, 1000))
        for i in range(6):
            for j in range(1000):
                x = self.df_dist.loc[(self.df_dist.dc_id == i) & (self.df_dist.item_sku_id == j + 1)]
                if (x["dist_type"].values == 'G'):
                    demand_of_today[i][j] = np.around(self.para1[i][j] * self.para2[i][j])
                else:
                    demand_of_today[i][j] = np.around(self.para1[i][j] * (1 - self.para2[i][j]) / self.para2[i][j])
        return demand_of_today

    def caculate_available_inventory_and_demand(self,inventory_of_t_1,RDC_reservation,reorder_point,inventory_level):
        """计算当日可用的调拨量，以及当日的需求"""
        available_inventory=np.zeros((1,1000)).astype(int)
        demand=np.zeros((6,1000)).astype(int)

        for skuid in self.sku_list:
            available_inventory[0][skuid]=max(inventory_of_t_1[0][skuid]-RDC_reservation[skuid],0)
            for dcid in self.dc_list:
                if inventory_of_t_1[dcid][skuid]<4*reorder_point[dcid][skuid]:
                    demand[dcid][skuid]=np.round(4*inventory_level[dcid][skuid]-inventory_of_t_1[dcid][skuid])
        return available_inventory,demand

    def optimize_model(self,t,available_inventory,real_demand_of_t):

        print("Solving for day:",t)
        optimization_starttime=time.time()
        """创建规划模型"""
        m=Model("optimize_model")
        """变量创建"""

        #创建调拨变量(以及upperbound)
        upperbound=[available_inventory[0][:] for i in range(5)]
        omega=m.addVars(5,1000,lb=0,ub=upperbound,vtype=GRB.INTEGER,name="omega")

        #创建0,1用于约束2的变量
        y_c2=m.addVars(5,1000,lb=0,vtype=GRB.BINARY,name="y_c2")

        #对于成本c2,c3,需要缺货、RDC索取变量群，成本变量群以及一个约束变量群z

        stock_out_deduct_abandon_and_SR_of_t=m.addVars(5,1000,lb=0,vtype=GRB.INTEGER,name='stock_out_deduct_abandon_and_SR_of_t')
        cost1=m.addVars(6,1000,lb=0,name="cost1")
        cost2=m.addVars(1,1000,lb=0,name="cost2")
        cost3=m.addVars(1,1000,lb=0,name="cost3")

        #总成本变量，用于后期计算
        total_cost1=m.addVar(lb=0,name="total_cost1")
        total_cost2 = m.addVar(lb=0, name="total_cost2")
        total_cost3 = m.addVar(lb=0, name="total_cost3")


        """约束条件"""
        #约束条件1，保证RDC每件产品库存>=0
        for i in range(1000):
            m.addConstr(omega.sum('*', i) <= max(available_inventory[0][i], 0),"c1_" + str(i + 1))


        #约束条件2：种类限制,,ifx>0.y=1 0<=-x+My<=M-1
        for i in range(5):
             for j in range(1000):
                 m.addConstr(self.bigM*y_c2[i,j]-omega[i,j]>=0)
                 m.addConstr(self.bigM*y_c2[i,j]-omega[i,j]<=self.bigM-1)
             m.addConstr(y_c2.sum(i,"*")<=self.sku_limit[i])


        #约束条件3：个数限制
        for i in range(5):
            m.addConstr(omega.sum(i, '*') <= self.capacity_limit[i], "c3_" + str(i + 1))

        """c1成本计算"""
        abandoned_num_of_t=np.zeros((6,1000))
        stock_out_deduct_abandon_of_t=np.zeros((6,1000))

            #先单独计算RDC的成本
        for i in range(1000):
            if self.arrayinv[t-1][0][i]<=real_demand_of_t[0][i]:
                abandoned_num_of_t[0][i]=real_demand_of_t[0][i]-self.arrayinv[t-1][0][i]
                m.addConstr(cost1[0,i]==(self.arraycost[i]*abandoned_num_of_t[0][i]))

            #再计算FDC成本
        for i in range(1,6):
            for j in range(1000):
                if self.arrayinv[t-1][i][j] <= real_demand_of_t[i][j]:
                    abandoned_num_of_t[i][j]=np.ceil(self.abandon_rate[i-1]*(real_demand_of_t[i][j]-self.arrayinv[t-1][i][j] ))
                    stock_out_deduct_abandon_of_t[i][j]=(real_demand_of_t[i][j]-self.arrayinv[t-1][i][j])-abandoned_num_of_t[i][j]
                    m.addConstr(cost1[i,j]==self.arraycost[j] *abandoned_num_of_t[i][j])
                    z=m.addVar(lb=-GRB.INFINITY)#保证能运行添加变量
                    m.addConstr(z==stock_out_deduct_abandon_of_t[i][j]-omega[i-1,j])
                    m.addConstr(stock_out_deduct_abandon_and_SR_of_t[i-1,j]==max_(z,0))


        self.abandoned_num.append(abandoned_num_of_t)
        self.stock_out_deduct_abandon.append(stock_out_deduct_abandon_of_t)


        """c2,c3计算"""
        for i in range(1000):
            excess_RDC_inventory=float(self.arrayinv[t-1][0][i]-real_demand_of_t[0][i])
            z=m.addVar(lb=-GRB.INFINITY)  # 保证能运行添加变量
            m.addConstr(z == self.arraycost[i]*(stock_out_deduct_abandon_and_SR_of_t.sum('*',i)-excess_RDC_inventory))
            m.addConstr(cost2[0,i]==max_(z,0))
            if excess_RDC_inventory>=0:
                z=m.addVar(lb=-GRB.INFINITY)  # 保证能运行添加变量
                m.addConstr(z==self.q*stock_out_deduct_abandon_and_SR_of_t.sum('*',i))
                m.addConstr(cost3[0,i] == min_(self.q*excess_RDC_inventory,z) )
            else:
                m.addConstr(cost3[0,i]==0)

        #总成本计算
        m.addConstr(total_cost1==cost1.sum())
        m.addConstr(total_cost2==cost2.sum())
        m.addConstr(total_cost3==cost3.sum())

        #设置目标函数，并求解
        m.setObjective(total_cost1+total_cost2+total_cost3,GRB.MINIMIZE)
        m.optimize()

        transshipment_decision = np.zeros((5, 1000)).astype(int)

        #将结果转存
        if m.status == GRB.status.OPTIMAL:
            self.total_cost+=m.objVal

            for i in range(5):
                for j in range(1000):
                    transshipment_decision[i][j]=omega[i,j].x
            return transshipment_decision

        elif m.status != GRB.Status.INFEASIBLE:
            print('Optimization was stopped with status %d' % m.status)
        else:
            print('')
            print('Model is infeasible')
            m.computeIIS()
            m.write("model.ilp")
            print("IIS written to file 'model.ilp'")
            exit(0)
        if(m.objVal<=0):
            exit(0)




