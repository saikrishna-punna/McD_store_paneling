import pandas as pd
import numpy as np
import os
import yaml
import traceback
from FE import dummy_fe, neighbour_store_count
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


from models import Lme, gmm
import sp_utils as ut
import warnings
warnings.filterwarnings("ignore")

# pd.options.mode.chained_assignment = None  # default='warn'

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
        self.wap = self.config['store_paneling']['col_names']['wap']
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

            stores_final = list(set(self.pmix[self.store_id].unique()).intersection(set(self.gc[self.store_id].unique())))

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
                num_cols = ut.load_flat_file(self.store_info_path, sheet_name='Numeric columns to consider')['COLUMNS_TO_CONSIDER']
                num_cols = ut.clean_col_names(pd.Series(num_cols)).to_list()
                cat_cols = ut.load_flat_file(self.store_info_path, sheet_name='NonNumeric columns to consider')['COLUMNS_TO_CONSIDER']
                cat_cols = ut.clean_col_names(pd.Series(cat_cols)).to_list()

                storeinfo = ut.load_flat_file(self.store_info_path, sheet_name='storeinfo')[[self.store_id]+num_cols+cat_cols]
                storeinfo = storeinfo.set_index(self.store_id)
                storeinfo[cat_cols] = ut.change_dtypes(storeinfo[cat_cols], 'string')
                storeinfo[num_cols] = ut.change_dtypes(storeinfo[num_cols], 'numeric')
                fe_for_gmm = storeinfo.reset_index().copy()
                fe_for_gmm['NUM_STORES_IN_5KM'] = fe_for_gmm[self.store_id].map(neighbour_store_count(fe_for_gmm.set_index(self.store_id), 5))
                fe_for_gmm.drop(columns=['LATITUDE', 'LONGITUDE'], inplace=True)

                fe_for_gmm = fe_for_gmm[fe_for_gmm[self.store_id].isin(stores_final)]
                fe_for_gmm[self.guestcount] = fe_for_gmm[self.store_id].map(self.aggregate_data_store_lvl('gc', 1))
                fe_for_gmm[self.unit_sales] = fe_for_gmm[self.store_id].map(self.aggregate_data_store_lvl('units', 1))
                fe_for_gmm[self.sales] = fe_for_gmm[self.store_id].map(self.aggregate_data_store_lvl('sales', 1))
                fe_for_gmm[self.ac] = fe_for_gmm[self.store_id].map(self.aggregate_data_store_lvl('AC', 1))
                fe_for_gmm[self.wap] = fe_for_gmm[self.store_id].map(self.aggregate_data_store_lvl('WAP', 1))
                fe_for_gmm[self.upt] = fe_for_gmm[self.store_id].map(self.aggregate_data_store_lvl('UPT', 1))
                
                self.fe_for_gmm = fe_for_gmm.set_index(self.store_id)
                print("Created Features for GMM")
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
        Returns
        -------
        Dict
            Store level wap coefficients

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
        """aggregate_data_store_lvl 

        Aggregate (mean) the given metric in the given num_years

        Parameters
        ----------
        metric : str
            options: 'gc', 'sales', 'units', 'AC', 'UPT', 'WAP'
        num_years : int
            To filter the latest n number of years

        Returns
        -------
        dict
            store level metric value
        """
        pmix_df = self.pmix.copy()
        gc_df = self.gc.copy()
        start_year = pmix_df[self.week_id].dt.year.max() - num_years
        pmix_df = pmix_df[pmix_df[self.week_id].dt.year>=start_year]
        gc_df = gc_df[gc_df[self.week_id].dt.year>=start_year]

        gc_values = gc_df.groupby(self.store_id)[self.guestcount].mean().reset_index()
        if metric == 'gc':
            return dict(gc_values.values)
        else:
            df = gc_df.groupby(self.store_id)[self.guestcount].mean().reset_index()
            gc_weeks = gc_df.groupby(self.store_id)[self.week_id].nunique().to_dict()
            df = pmix_df.groupby([self.store_id])[self.sales, self.unit_sales].sum().reset_index()
            df['num_weeks'] = df[self.store_id].map(gc_weeks)
            df[self.sales] = df[self.sales]/df['num_weeks']
            df[self.unit_sales] = df[self.unit_sales]/df['num_weeks']
            df = df.merge(gc_values, on=self.store_id, how='left')
            if metric == 'sales':
                return dict(df[[self.store_id, self.sales]].values)
            if metric == 'units':
                return dict(df[[self.store_id, self.unit_sales]].values)
            if metric == 'AC':
                df[self.ac] = df[self.sales]/df[self.guestcount]
                return dict(df[[self.store_id, self.ac]].values)
            if metric == 'UPT':
                df[self.upt] = df[self.unit_sales]/df[self.guestcount]
                return dict(df[[self.store_id, self.upt]].values)
            if metric == 'WAP':
                df[self.wap] = df[self.sales]/df[self.unit_sales]
                return dict(df[[self.store_id, self.wap]].values)

        
    def elbow(self, max_clusters):

        X = pd.get_dummies(self.fe_for_gmm.dropna(axis=1))
        wcss = []
        for i in range(1,max_clusters):
            kmean = KMeans(n_clusters=i,init="k-means++")
            kmean.fit_predict(X)
            wcss.append(kmean.inertia_)

        plt.plot(range(1,max_clusters),wcss)
        plt.title('Elbow Curve')
        plt.xlabel("No of Clusters")
        plt.ylabel("WCSS")
        plt.show()


    def get_clusters(self):
        try:
            num_clusters = [2]  # Get it from elbow function

            data = pd.get_dummies(self.fe_for_gmm.dropna(axis=1))

            # Scale featurs to convert to normal dist.
            scaler = StandardScaler()
            scaler.fit_transform(data)
            self.final_dict = {}
            for clusters in num_clusters:
                gmm_args = {}
                gmm_args['num_clusters'] = clusters
                gmm_args['covariance_type'] = 'full'
                gmm_args['max_iter'] = 100
                gmm_args['init_params'] = 'kmeans'
                
                GMM = gmm(data, gmm_args)
                GMM.fit()
                labels = GMM.predict()
                gmm_summary = GMM.get_model_summary()
                
                cluster_dict = {}
                cluster_dict['Summary'] = gmm_summary
                cluster_dict['Labels'] = labels
                
                self.final_dict[clusters] = cluster_dict
        
            for clusters in num_clusters:
                print(self.final_dict[clusters]['Summary'])
        except:
            print(traceback.format_exc())

