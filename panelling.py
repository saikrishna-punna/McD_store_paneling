import pandas as pd
import numpy as np
import os
import yaml
import traceback
from FE import dummy_fe
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from models import Lme
import sp_utils as ut
pd.options.mode.chained_assignment = None  # default='warn'

class clustering:
    
    def __init__(self):

        self.config = ut.yaml_to_dict('sp_default.yaml')

        self.pmix_path = self.config['store_paneling']['paths']['store_pmix_path'].format(**self.config['store_paneling']['paths'])
        self.gc_path = self.config['store_paneling']['paths']['store_gc_path'].format(**self.config['store_paneling']['paths'])
        # self.storeinfo_path = self.config['store_paneling']['paths']['storeinfo'].format(**self.config['store_paneling']['paths'])

        self.baseprice = self.config['store_paneling']['col_names']['baseprice']
        self.days_open = self.config['store_paneling']['col_names']['days_open']
        self.guestcount = self.config['store_paneling']['col_names']['gc']
        self.item_id = self.config['store_paneling']['col_names']['item_id']
        self.sales = self.config['store_paneling']['col_names']['sales']
        self.store_id = self.config['store_paneling']['col_names']['store_id']
        self.unit_sales = self.config['store_paneling']['col_names']['unit_sales']
        self.week_id = self.config['store_paneling']['col_names']['week_id']


        self.start_date = pd.to_datetime(self.config['store_paneling']['inputs']['start_date'])
        self.end_date = pd.to_datetime(self.config['store_paneling']['inputs']['end_date'])

        self.stores_list = os.listdir(self.pmix_path)
        if self.config['store_paneling']['paths']['storesList']:
            stores = ut.load_flat_file(self.config['store_paneling']['paths']['storesList']).values
            self.stores_list = [file_ for file_ in self.stores_list if any([store in file_ for store in stores])]
               
        self.stores_list_gc = os.listdir(self.gc_path)
        if self.config['store_paneling']['paths']['storesList']:
            stores = ut.load_flat_file(self.config['store_paneling']['paths']['storesList']).values
            self.stores_list_gc = [file_ for file_ in self.stores_list_gc if any([store in file_ for store in stores])]

        self.process_data()
        # self.storeinfofeaturesdf=ut.storeinfogenerate(self.config['store_paneling']['paths']['storeinfo'])
        # self.storeinfofeaturesdf.to_excel(os.path.join(self.config['store_paneling']['features']['store_info_features'],'Storeinfoprocessed.xlsx'),index=False)

    def process_data(self):
        try:
            self.pmix = pd.concat([ut.load_flat_file(os.path.join(self.pmix_path, file_)) for file_ in self.stores_list])
            self.gc = pd.concat([ut.load_flat_file(os.path.join(self.gc_path, file_)) for file_ in self.stores_list_gc])

            self.pmix[self.week_id]=pd.to_datetime(self.pmix[self.week_id])
            self.gc[self.week_id]=pd.to_datetime(self.gc[self.week_id])
            self.pmix = self.pmix[(self.pmix[self.week_id]>=self.start_date) & (self.pmix[self.week_id]<=self.end_date)]
            self.gc = self.gc[(self.gc[self.week_id]>=self.start_date) & (self.gc[self.week_id]<=self.end_date)]

            stores_final = list(set(self.pmix[self.store_id].unique())-set(self.gc[self.store_id].unique()))

            print("num of stores in pmix: ", self.pmix[self.store_id].nunique())
            print("num of stores in gc: ", self.gc[self.store_id].nunique())
            print("num of common stores: ", len(stores_final))

            self.pmix = self.pmix[self.pmix[self.store_id].isin(stores_final)]
            self.gc = self.gc[self.gc[self.store_id].isin(stores_final)]


            # TODO: Update it with FE API
            if self.config['store_paneling']['inputs']['use_elasticity_as_feature']:
                # agg = {
                #     self.unit_sales: np.sum,
                #     self.sales: np.sum,
                # }
                # self.pmix_for_elasticities = self.pmix.pivot_table(index=[self.store_id, self.week_id], values=list(agg.keys()), aggfunc=agg).reset_index()
                # self.gc_for_elasticites = self.gc[[self.store_id, self.week_id, self.guestcount]].copy()
                # self.wap_for_elasticities = self.pmix.groupby(self.store_id).apply(self.rolling_wap)

                fe_for_elasticity = dummy_fe()
                self.fe_for_elasticity = fe_for_elasticity[fe_for_elasticity[self.store_id].isin(stores_final)]
                self.fe_for_elasticity.drop(columns=['WEIGHTED_PRICE_STORE', 'UNNAMED_0'], inplace=True)
                self.fe_for_elasticity.rename(columns={'WEIGHTED_PRICE_STORE':'WAP'}, inplace=True)
        except:
            print(traceback.format_exc())


    def get_elasticities(self, y_var='units', test_weeks = 20):
        # log transform price var
        self.fe_for_elasticity[self.baseprice] = np.log(self.fe_for_elasticity[self.baseprice])
        self.fe_for_elasticity = self.fe_for_elasticity.sort_vales([self.store_id, self.week_id]).reset_index(drop=True)
        
        self.fe_for_elasticity['index_'] = self.fe_for_elasticity[self.store_id].astype(str) + "_" + self.fe_for_elasticity[self.week_id].astype(str)
        self.fe_for_elasticity = self.fe_for_elasticity.set_index('index_')
        self.fe_for_elasticity.drop(columns=self.week_id, inplace=True)
        
        # Test train split by number of test weeks
        test = self.fe_for_elasticity.group_by([self.store_id, self.week_id]).tail(test_weeks).reset_index()
        train = self.fe_for_elasticity.iloc[:,:(-1*test_weeks)]

        lme_args = {}
        # log transform y_var and drop other y column
        if y_var=='units':
            self.fe_for_elasticity[self.unit_sales] = np.log(self.fe_for_elasticity[self.unit_sales])
            self.fe_for_elasticity.drop(columns=[self.guestcount], inplace=True)
            lme_args['y_col'] = self.unit_sales

        if y_var=='gc':
            self.fe_for_elasticity[self.guestcount] = np.log(self.fe_for_elasticity[self.guestcount])
            self.fe_for_elasticity.drop(columns=[self.unit_sales], inplace=True)
            lme_args['y_col'] = self.guestcount


        lme_args['group_col'] = self.config['store_paneling']['elasticity_model']['group_col']
        lme_args['random_effects'] = self.config['store_paneling']['elasticity_model']['random_effects']
        lme_args['fixed_effects'] = [col for col in self.fe_for_elasticity.column if col not in lme_args['group_col']+lme_args['y_col']]

        # Scaling the features
        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(train)
        scaled_test = scaler.transform(test)

        # y_col           : column name of dependent variable (str)
        # group_col       : column name containing grouping variable for random fixed_effects (str)
        # fixed_effects   : list of features to be treated as fixed effects
        # random_effects  : list of features to be treated as random effects

        LME = Lme(scaled_train, lme_args, track_tag='')
        pred_df = LME.predict(test_data)
        


ins = clustering()
print(ins)



    # def rolling_wap(self, df, rolling_weeks):
    #     df.sort_values(by=self.week_id, inplace=True)
    #     itemPrices = df.pivot_table(index=[self.store_id, self.week_id], columns=self.item_id, values=self.baseprice)
    #     itemUnits = df.pivot_table(index=[self.store_id, self.week_id], columns=self.item_id, values=self.unit_sales)[itemPrices.columns]
    #     itemBaseSales = itemPrices.multiply(itemUnits)
    #     itemBaseSalesRolling = itemBaseSales.shift(1).rolling(rolling_weeks, min_periods=1).sum()
    #     itemUnitsRolling = itemUnits.shift(1).rolling(rolling_weeks, min_periods=1).sum()

    #     numerator = itemPrices.multiply(itemUnitsRolling)
    #     numerator = numerator.fillna(itemBaseSalesRolling)

    #     wap = numerator.sum(axis=1).divide(itemUnitsRolling.sum(
    #         axis=1)).reset_index().set_index([self.store_id, self.week_id])
    #     wap = wap.dropna().reset_index()
    #     wap.columns = [self.store_id, self.week_id]+['WAP']




    # TODO: after getting the elasticites add store level aggregation to read_data

    # def aggregate_data_store_lvl(self, pmix_df, gc_df, metric, num_years):
        
    #     if metric == self.gc:
    #         return gc_df.groupby(self.store_id)[self.gc].mean()
    #     else:
    #         gc_weeks = gc_df.groupby(self.store_id)[self.week_id].unique().to_dict()
    #         if metric == self.sales:
    #             df = pmix_df.groupby()
        



