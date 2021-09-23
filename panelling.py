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
        self.store_info_path = self.config['store_paneling']['paths']['storeinfo'].format(**self.config['store_paneling']['paths'])
        # self.storeinfo_path = self.config['store_paneling']['paths']['storeinfo'].format(**self.config['store_paneling']['paths'])

        self.ac = self.config['store_paneling']['col_names']['ac']
        self.baseprice = self.config['store_paneling']['col_names']['baseprice']
        self.days_open = self.config['store_paneling']['col_names']['days_open']
        self.guestcount = self.config['store_paneling']['col_names']['gc']
        self.item_id = self.config['store_paneling']['col_names']['item_id']
        self.sales = self.config['store_paneling']['col_names']['sales']
        self.store_id = self.config['store_paneling']['col_names']['store_id']
        self.unit_sales = self.config['store_paneling']['col_names']['unit_sales']
        self.upt = self.config['store_paneling']['col_names']['upt']
        self.wap = self.config['store_paneling']['col_names']['WAP']
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
                fe_for_elasticity = dummy_fe()
                self.fe_for_elasticity = fe_for_elasticity[fe_for_elasticity[self.store_id].isin(stores_final)]
                self.fe_for_elasticity.drop(columns=['WEIGHTED_PRICE_STORE', 'UNNAMED_0'], inplace=True)
                self.fe_for_elasticity.rename(columns={'WEIGHTED_PRICE_STORE':'WAP'}, inplace=True)
            
            if self.config['store_paneling']['inputs']['run_gmm'] or self.config['store_paneling']['inputs']['run_rf']:
                # import store info file and format required columns
                storeinfo = ut.load_flat_file(self.store_info_path, sheet_name='storeinfo')
                num_cols = ut.load_flat_file(self.store_info_path, sheet_name='Numeric columns to consider').values
                cat_cols = ut.load_flat_file(self.store_info_path, sheet_name='NonNumeric columns to consider').values
                storeinfo = storeinfo[[self.store_id]+[num_cols]+[cat_cols]]
                storeinfo = storeinfo.set_index(self.store_id)
                storeinfo[cat_cols] = storeinfo[cat_cols].astype(str)
                storeinfo[num_cols] = pd.to_numeric(storeinfo[cat_cols], errors='coerce')
                fe_for_gmm = storeinfo.copy()
                # fe_for_gmm[''] = 
     

        except:
            print(traceback.format_exc())


    def get_elasticities(self, y_var='units', test_weeks = 20):
        """get_elasticities 

        To get the demand or gc elasticities for all the stores.

        Parameters
        ----------
        y_var : str, optional
            "units" for demand elasticity and "gc" for GC elasticity, by default 'units'
        test_weeks : int, optional
            number of test weeks, by default 20
        """
        try:
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
            # scaler = MinMaxScaler()
            # scaled_train = scaler.fit_transform(train)
            # scaled_test = scaler.transform(test)

            # y_col           : column name of dependent variable (str)
            # group_col       : column name containing grouping variable for random fixed_effects (str)
            # fixed_effects   : list of features to be treated as fixed effects
            # random_effects  : list of features to be treated as random effects

            LME = Lme(train, lme_args, track_tag='')
            res = LME.fit()
            params = LME.get_params()
            pred_df = LME.predict(test)

            return res, params, pred_df
        except:
            print(traceback.format_exc())
    


    def aggregate_data_store_lvl(self, metric, num_years):
        
        start_year = self.pmix_df[self.week_id].dt.year.max() - num_years
        self.pmix_df = self.pmix_df[self.pmix_df[self.week_id].dt.year>=start_year]
        self.gc_df = self.gc_df[self.gc_df[self.week_id].dt.year>=start_year]

        gc_values = self.gc_df.groupby(self.store_id)[self.gc]
        if metric == 'gc':
            return gc_values
        else:
            df = self.gc_df.groupby(self.store_id)[self.gc]
            gc_weeks = self.gc_df.groupby(self.store_id)[self.week_id].unique().to_dict()
            df = self.pmix_df.groupby([self.store_id])[self.sales, self.unit_sales].sum().reset_index()
            df['num_weeks'] = df[self.store_id].map(gc_weeks)
            df[self.sales] = df[self.metric]/df['num_weeks']
            df[self.unit_sales] = df[self.unit_sales]/df['num_weeks']
            df = df.merge(gc_values, on=self.store_id, how='left')
            if metric == 'sales':
                return df[[self.store_id, self.sales]]
            if metric == 'units':
                return df[[self.store_id, self.unit_sales]]
            if metric == 'AC':
                df[self.ac] = df[self.sales]/df[self.guestcount]
                return df[[self.store_id, self.ac]]
            if metric == 'UPT':
                df[self.ac] = df[self.unit_sales]/df[self.guestcount]
                return df[[self.store_id, self.upt]]     
            if metric == 'WAP':
                df[self.ac] = df[self.sales]/df[self.unit_sales]
                return df[[self.store_id, self.upt]]     

        





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




        

inst = clustering()
inst.process_data()
print('')


