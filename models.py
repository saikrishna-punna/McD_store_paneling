import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import traceback

class Lme():
    """
    Lme serves as an api for linear mixed effects modeling (internally uses statsmodels mixedlm)
    provides an interface  to fit, predict and aso to get formatted version of random and fixed betas
    """

    def __init__(self, train_data, lme_args, track_tag):
        """
        initializes the class attributes
        Parameters
        ----------
        train_data : DataFrame
            pandas dataframe containing train data
        lme_args : dict
            dictionary containing arguments to configure lme run
            y_col           : column name of dependent variable (str)
            group_col       : column name containing grouping variable for random fixed_effects (str)
            fixed_effects   : list of features to be treated as fixed effects
            random_effects  : list of features to be treated as random effects
        track_tag : str
            a string in the following format (to log/track the combination of the invoking job):
            format: "item_panel_rolling" (ex: 4314_2_20-02-2021)
        Attributes
        ----------
        track_tag : str
        y_col : str
        group_col : str
        fixed_effects : list
        random_effects  : list
        train_data : dataframe
        md : statsmodel-model
            initialised model object (will be used for training and predictions by other class functions)
        param_gen_flag : boolean
            indicates if model parameters are already generated and stored (helps avoid redundent function calls)
        
        """
        try:
            self.track_tag = track_tag
            self.y_col = lme_args["y_col"]
            self.group_col = lme_args["group_col"]
            self.fixed_effects = lme_args["fixed_effects"]
            self.random_effects = lme_args["random_effects"]
            self.train_data = train_data

            self.param_gen_flag = 0

            endog = self.train_data[self.y_col]
            exog = self.train_data[self.fixed_effects]
            exog_re = self.train_data[self.random_effects]
            groups = self.train_data[self.group_col]

            self.md = sm.MixedLM(endog=endog,
                                groups=groups,
                                exog=exog,
                                exog_re=exog_re)
        except:
            print(traceback.format_exc())

    def fit(self):
        """
        fit is statsmodels_model.fit()
        fits a linear mixed effect model on a predefiend model (class atribute)
        """
        try:
            self.mdf = self.md.fit()
        except:
            print(traceback.format_exc())

    def gen_params(self):
        """
        gen_params fetches model params and stores in class attributed for further reference
        executes only if attribute param_gen_flag is 0 (1 indicates that parameters already updated in class attributes)
        Attributes
        ----------
        fixed_params : dict
            trained parameters of fixed fixed_effects
        random params: dict
            trained parameters of random_effects 
        """
        try:
            if not self.param_gen_flag:
                self.fixed_params = self.mdf.params
                self.random_params = self.mdf.random_effects
                self.param_gen_flag = 1
        except:
            print(traceback.format_exc())

    def get_params(self):
        """
        get_params formats the generated params to a consumable format and returns
        Returns
        -------
        fixed_param_df : DataFrame
            DataFrame containing fixed_effect feature wise learnt coefficient
        random_param_df: DataFrame
            DataFrame containing group x random_effect feature wise learnt coefficient
        """
        try:
            self.gen_params()
            # add code for correct formating align with bayesian params
            random_param_df = pd.DataFrame(self.random_params).T.reset_index()
            random_param_df = random_param_df.rename(columns={"index": self.group_col})
            fixed_param_df = pd.DataFrame(self.fixed_params).reset_index()
            fixed_param_df.columns = ["feature", "param"]
            return fixed_param_df, random_param_df
        except:
            print(traceback.format_exc())

    def make_pred_col(self, col, track_Tag):
        """
        make_pred_col extends a column name with postfix to make a coresponding column name for predictions
        postfix string is taken from utils/names.py 
        purpose of this function is to standardise the column names representing predictions (by avoiding random string constructions)
        
        Parameters
        ----------
        col : string
            actual column name of feature to be predicted
        track_tag : string
            an identifier used to log/track the info about of the invoking job
        Returns
        -------
        string
            corresponding prediction column name
        """
        try:
            return col + '_PRED'
        except:
            print(traceback.format_exc())
    
    def predict(self, test_data=pd.DataFrame()):
        """
        predict used trained LME parameters from astats models model to generate predictions on the given dataset
        Parameters
        ----------
        test_data : DataFrame, optional 
            predictions performed on train data if this parameter is not specified or emty
        Returns
        -------
        DataFrame
            group x week level actuals and predictions of dependent variable
            
        """

        try:
            self.gen_params()
            if not test_data.empty:
                data = test_data.copy()
            else:
                data = self.train_data.copy()

            # vectorize this
            pred_yhat = []
            groups_in_order = data[self.group_col].drop_duplicates().to_list()
            for group_id in data[self.group_col].unique():
                group_df = data[data[self.group_col] == group_id]  # change to names.
                group_yhat = np.zeros(len(group_df))
                for fe in self.fixed_effects:
                    coef = self.fixed_params[fe]
                    group_yhat = group_yhat + group_df[fe].values * coef
                for re in self.random_effects:
                    coef = self.random_params[group_id][re]
                    group_yhat = group_yhat + group_df[re].values * coef
                group_pred_df = pd.DataFrame(group_yhat)
                group_pred_df[1] = group_id
                group_pred_df[2] = group_df[self.y_col].values
                group_pred_df[3] = group_df['WK_END_DT'].values  # change to names.
                pred_yhat.append(group_pred_df)

            pred_yhat_df = pd.concat(pred_yhat)
            pred_yhat_df.columns = [self.make_pred_col(self.y_col, self.track_tag), self.group_col, self.y_col, 'WK_END_DT']

            return pred_yhat_df
        except:
            print(traceback.format_exc())
