import pandas as pd
import numpy as np
import os
import yaml

import sp_utils as ut


class clustering:
    
    def __init__(self):

        self.config = ut.yaml_to_dict('sp_default.yaml')
        
        self.pmix_path = self.config['store_paneling']['paths']['store_pmix_path']
        self.gc_path = self.config['store_paneling']['paths']['store_gc_path']
        self.storeinfo_path = self.config['store_paneling']['paths']['storeinfo']

        self.baseprice = self.config['store_paneling']['col_names']['baseprice']
        self.days_open = self.config['store_paneling']['col_names']['days_open']
        self.gc = self.config['store_paneling']['col_names']['gc']
        self.item_id = self.config['store_paneling']['col_names']['item_id']
        self.sales = self.config['store_paneling']['col_names']['sales']
        self.store_id = self.config['store_paneling']['col_names']['store_id']
        self.unit_sales = self.config['store_paneling']['col_names']['unit_sales']
        self.week_id = self.config['store_paneling']['col_names']['week_id']

        self.start_date = pd.to_datetime(self.config['store_paneling']['inputs']['start_date'])
        self.end_date = pd.to_datetime(self.config['store_paneling']['inputs']['end_date'])

        self.stores_list = os.listdir(self.pmix_path)
        if self.config['paths']['storeList']:
            stores = ut.load_flat_file(self.config['paths']['storeList']).values
            self.stores_list = [file_ for file_ in self.stores_list if any([store in file_ for store in stores])]

        self.read_data()



    def read_data(self):

        self.pmix = pd.concat([ut.load_flat_file(os.path.join(self.pmix_path, file_)) for file_ in self.store_files_list])
        self.gc = pd.concat([ut.load_flat_file(os.path.join(self.gc_path, file_)) for file_ in self.store_files_list])

        self.pmix = self.pmix[(self.pmix[self.week_id]>=self.start_date) & (self.pmix[self.week_id]<=self.end_date)]
        self.gc = self.gc[(self.gc[self.week_id]>=self.start_date) & (self.gc[self.week_id]<=self.end_date)]

        stores_final = list(set(self.pmix[self.store_id].nunique())-set(self.gc[self.store_id].nunique()))

        print("num of stores in pmix: ", self.pmix[self.store_id].nunique())
        print("num of stores in gc: ", self.gc[self.store_id].nunique())
        print("num of common stores: ", len(stores_final))

        self.pmix = self.pmix[self.pmix[self.store_id].isin(stores_final)]
        self.gc = self.gc[self.gc[self.store_id].isin(stores_final)]

        agg = {
            self.unit_sales: np.sum,
            self.sales: np.sum,
        }
        self.pmix_for_elasticities = self.pmix.pivot_table(index=[self.store_id, self.week_id], values=list(agg.keys()), aggfunc=agg).reset_index()
        self.gc_for_elasticites = self.gc[[self.store_id, self.week_id, self.gc]].copy()

        


    # TODO: after getting the elasticites add store level aggregation to read_data

    # def aggregate_data_store_lvl(self, pmix_df, gc_df, metric, num_years):
        
    #     if metric == self.gc:
    #         return gc_df.groupby(self.store_id)[self.gc].mean()
    #     else:
    #         gc_weeks = gc_df.groupby(self.store_id)[self.week_id].unique().to_dict()
    #         if metric == self.sales:
    #             df = pmix_df.groupby()
        

        


