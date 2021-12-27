"""
Description
-----------

    This module is linked to the Arvato Project Workbook (jupyterlab notebook). Many explanation is given in the workbook. 


"""
import pandas as pd
import numpy as np



class DataCleaner:
    """
    Description
    -----------

    This class will provide some functionality to execute some basic cleanining 
    a arvato data set
    """
    
                                    
    @property
    def df_metadata(self):
        return self.__df_metadata
    
    @df_metadata.setter
    def df_metadata(self, val):
        self.__df_metadata = val    
    
    def __init__(self, drop_nan_threshold:float, df_metadata:pd.DataFrame):
        """
        Description
        -----------

        inits the class.
        
        Parameters
        ----------
            df_metadata: pd.DataFrame
                pandas dataframe with the loaded data from file "DIAS Attributes - Values 2017.xlsx". Containing
                information about the attribute values.
        """
        assert (drop_nan_threshold >0) and (drop_nan_threshold <= 1.0)
        self.drop_nan_threshold = drop_nan_threshold
        self.df_metadata = df_metadata

        # replace the metadata attribute column ending "_RZ" by "" in order to match the dataset column names
        self.df_metadata['Attribute'] = self.df_metadata['Attribute'].str.replace('_RZ','')

    
    def transform(self, df:pd.DataFrame, drop_duplicates:bool=False, build_kind_features:bool=True, drop_cols:bool=True)->pd.DataFrame:
        """
        Description
        -----------
                
        executes the data transformation (cleaning)

        Parameters
        ----------

            df : pd.DataFrame
                the dataframe that is to be cleaned

        """                      
        df = self.__drop_customer_columns(df)
        df = self.__handle_data_load_errors(df)
        
        if drop_duplicates:
            df.drop_duplicates(inplace=True)
            
        df = self.__fix_year_columns(df)
        df = self.__mark_nans(df)
        
        if build_kind_features:
            df = self.__build_features_chidren(df, drop_childcols=False)
            
        df = self.__catvars_to_dummies(df)        
        df = self.__catvars_to_binary(df)          
                
        if drop_cols:                                                
            df = self.__drop_columns(df)        
            
        return df
            
        
    def fit (self, df:pd.DataFrame)->pd.DataFrame:
        """
        Description
        -----------

        prepare data for transformation
        """

        pass
    
    def __fix_year_columns(self, df:pd.DataFrame) ->pd.DataFrame:
        """
        Description
        ------------
        
            converts year columns to int
        """        
        cols = ['MIN_GEBAEUDEJAHR','EINGEZOGENAM_HH_JAHR','GEBURTSJAHR']
        print(f'fixing year columns: {cols}')
        for col in cols:
            df[col].fillna(df[col].median(), inplace=True)
            df[col].astype('int')
        
        
        return df
    
    

        
    def __drop_customer_columns (self, df:pd.DataFrame, columns_to_drop:bool=None)->pd.DataFrame:
        """
        drop additional coloumns of the customer dataset
        """
        cols = ['CUSTOMER_GROUP', 'ONLINE_PURCHASE', 'PRODUCT_GROUP']        
        
        if cols[0] in df.columns:
            print(f'Dropping customer dataset cols: {cols}')
            df = self.__drop_columns(df, cols)
            
        return df
        
        
    
    def __handle_data_load_errors(self, df:pd.DataFrame) ->pd.DataFrame:
        """
        handles the errors fo columns 18 and 19 of dtype float that contain two 18,19 
        """
        cols_to_fix = {'CAMEO_DEUG_2015':'X', 'CAMEO_INTL_2015':'XX'}

        print(f'fixing load errors {cols_to_fix}')

        for col, val in cols_to_fix.items():
            n = df.loc[df[col] == val].shape[0]
            df.loc[df[col] == val, col] = np.NaN
            df.loc[:,col] = df.loc[:,col].astype('float')

            print(f'fixed column {col} - records fixed: {n}')
        
        return df


    def __drop_columns(self, df:pd.DataFrame, columns_to_drop:list=None, drop_nan_threshold:float=None)->pd.DataFrame:
        """
        Description
        -----------
        
            LP_STATUS_GROB: drop this as LP_STATUS_FEIN contains the same information more detailed
            LP_FAMILIE_GROB : analogue to LP_STATUS_GROB
            D19_VERSAND_ANZ_24: drop
            EINGEFUEGT_AM : just timestamp information when the record has been created
            LP_LEBENSPHASE_FEIN: drop - we keep just LP_LEBENSPHASE_GROB

        """
        # if columns to drop have been defined then use them 
        # else execute the default cleaning        
        if columns_to_drop:            
            cols_to_drop = columns_to_drop
        else: 
            # default set of columns to drop
            cols_to_drop = ['EINGEFUEGT_AM']
            
            # drop because of very high correlation to other columns (>=0.9).
            cols_toomuchcorrelation = ['CAMEO_DEU_2015','LP_STATUS_GROB','LP_FAMILIE_GROB','D19_VERSAND_ANZ_24','LP_LEBENSPHASE_FEIN', 
                                       'ANZ_STATISTISCHE_HAUSHALTE', 'CAMEO_INTL_2015', 'D19_VERSAND_ONLINE_DATUM', 'KBA13_HALTER_66',
                                       'KBA13_HERST_SONST',  'LP_LEBENSPHASE_GROB',
                                       'PLZ8_BAUMAX', 'PLZ8_GBZ', 'PLZ8_HHZ',
                                       'D19_GESAMT_ANZ_24', 'D19_VERSAND_ANZ_12',  'D19_VERSAND_DATUM',   'KBA05_KRSHERST2', 
                                       'KBA05_KRSHERST3', 'KBA05_SEG9', 'KBA13_KMH_250','PLZ8_ANTG1', 'PLZ8_ANTG3', 'PLZ8_ANTG4',
                                      'D19_VERSAND_ONLINE_QUOTE_12','GEBURTSJAHR']
            cols_toomuch_neg_correlation = ['KOMBIALTER', 'ORTSGR_KLS9']
             
            
            if drop_nan_threshold:
                drop_level = 0.25

                num_of_records = df_azdias_cleaned.shape[0]
                s_missing_data_pct = df.isnull().sum(axis=0) / num_of_records 

                columns_to_drop = df.sort_values(ascending=False)
                columns_to_drop = columns_to_drop[columns_to_drop>drop_level].index            
                cols_toomanynulls = columns_to_drop.sort_values().tolist()
            else:            
                # drop because of too many NULL values (>25%)
                cols_toomanynulls =  ['AGER_TYP',
                                     'ALTERSKATEGORIE_FEIN',
                                     'ALTER_HH',
                                     'ALTER_KIND1',
                                     'ALTER_KIND2',
                                     'ALTER_KIND3',
                                     'ALTER_KIND4',
                                     'D19_BANKEN_ONLINE_QUOTE_12',
                                     'D19_GESAMT_ONLINE_QUOTE_12',
                                     'D19_KONSUMTYP',
                                     'D19_LETZTER_KAUF_BRANCHE',
                                     'D19_LOTTO',
                                     'D19_SOZIALES',
                                     'D19_TELKO_ONLINE_QUOTE_12',
                                     'D19_VERSAND_ONLINE_QUOTE_12',
                                     'D19_VERSI_ONLINE_QUOTE_12',
                                     'EXTSEL992',
                                     'GEBURTSJAHR',
                                     'KBA05_AUTOQUOT',
                                     'KBA05_BAUMAX',
                                     'KBA05_SEG6',
                                     'KKK',
                                     'KK_KUNDENTYP',
                                     'REGIOTYP',
                                     'TITEL_KZ']

            print()
            print('drop columns')
            print('Drop columns with too many Nulls: ', cols_toomanynulls)
            print('Drop columns with too much positive correlation: ', cols_toomanynulls)
            print('Drop columns with too much negative correlation: ', cols_toomuch_neg_correlation)
            print('Drop columns due too irrelevant data: ',cols_to_drop)
            cols_to_drop = cols_to_drop + cols_toomuchcorrelation + cols_toomanynulls + cols_toomuch_neg_correlation
        
        print('')
        print('- - '*25)
        print(f'dropping columns: {cols_to_drop}')                

        try:
            df.drop(labels=cols_to_drop, axis=1, inplace=True)   
        except KeyError as ex_keyerror:
            print(f'CATCHED EXCEPTION: KeyError: you tried to drop non existing columns: {cols_to_drop}')
            print(f'Failed columns: {ex_keyerror.args}')

        return df     

    def __catvars_to_dummies(self, df:pd.DataFrame)->pd.DataFrame:
        """
        Description
        -----------

        handles categorical variables. This will generate one hot encodings for the defined columns
        """
        #'CAMEO_DEU_2015' will be dropped - ignore this
        # D19_LETZTER_KAUF_BRANCHE-> will be deleted 
        cat_cols = []

        print('')
        print('- - '*25)
        print('creating one hot encoding columns for: ')
        for col in cat_cols:
            print(f'\t{col}')

        if cat_cols:
            # create one hot encodings using pandas get_dummies function
            df_dummies = pd.get_dummies(df[cat_cols], prefix=cat_cols, drop_first=True).astype('int64')
            df = pd.concat([df, df_dummies], axis=1)

            # drop original columns
            df.drop(cat_cols, axis=1, inplace=True)        

        return df

    def __catvars_to_binary(self, df:pd.DataFrame)->pd.DataFrame:
        """
        Description
        -----------

        """
        cat_cols = {'OST_WEST_KZ':{'W':0,'O':1}}

        print('convert to binary: ')
        for col, dict_map in cat_cols.items():
            print(f'\tcolumn: {col} - Mapping: {dict_map}')
            df.loc[:,col] = df.loc[:,col].map(dict_map)

        return df



    def __mark_nans(self, df:pd.DataFrame)->pd.DataFrame:
        """
        Description
        -----------

        replaces all unkown values by np.NAN so that the pandas NAN functions can be used.

        Parameters
        ----------

            df : pd.DataFrame
                pandas DataFrame that is to be cleaned. Frame is expected to have columns as AZDIAS or CUSTOMERS                
        """       
        print('')
        print('- - '*25)
        print('replace unkown values by NaNs: ') 
        unknown_val_set = self.df_metadata.copy()
        
        # select all rows that contain the term "unknown" 
        unknown_val_set = unknown_val_set[(unknown_val_set['Meaning'].str.contains('unknown'))]
        unknown_val_set['value_list']  = unknown_val_set['Value'].str.split(',')
        
        #with progressbar.ProgressBar(max_value=unknown_val_set.index.shape[0]) as bar:
        cnt = 0
        max_value=unknown_val_set.index.shape[0]
        for idx in unknown_val_set.index:
            col  = unknown_val_set.loc[idx,'Attribute']
            vals = unknown_val_set.loc[idx,'value_list']
            # str convert to integers
            vals = list(map(int,vals))
            if col in df:
                df.loc[df[col].isin(vals),col] = np.NaN

            cnt += 1
            if (cnt == max_value) or (cnt % (max_value // 10)==0):
                print(f'\tProcessed columns\r{cnt:4} of {max_value}', end='\r')
        
        
        #fix CAMEO_DEU_2015 XX will be dropped
        print('')
        print('Fix col CAMEO_DEU_2015: replace XX by NaN')
        df.loc[df['CAMEO_DEU_2015']=='XX','CAMEO_DEU_2015'] = np.NaN
        print()
        
        # fix for LP_LEBENSPHASE_GROB','LP_FAMILIE_FEIN => 0 is not described. We handle it as unknown == missing
        cols = ['LP_LEBENSPHASE_GROB','LP_FAMILIE_FEIN','GEBURTSJAHR']
        print(f'replace 0 by NaNs for : {cols}')
        df.replace({'LP_LEBENSPHASE_GROB':0 ,'LP_FAMILIE_FEIN':0, 'GEBURTSJAHR':0}, np.NaN, inplace=True)
        
        return df
    
    def __build_features_chidren(self, df:pd.DataFrame, drop_childcols:bool = True)->pd.DataFrame:
        """
        Description
        -----------
        
        This function will build some features based on the given input data

        * Children and Teens: 
            * Children:= number of children younger or equal than 10
            * Teens   := number of children older or equal than 10

        Parameters
        ----------
            df : pd.DataFrame
                pandas DataFrame that is to be cleaned. Frame is expected to have columns as AZDIAS or CUSTOMERS                
        """
        
        # num of children > 0
        df['d_HAS_CHILDREN'] = 0
        # younger than or equal 10
        df['d_HAS_CHILDREN_YTE10'] = 0

        df.loc[df['ANZ_KINDER'] > 0, 'd_HAS_CHILDREN'] = 1

        # mask to filter rows that have at least one record
        mask = df[['ALTER_KIND1','ALTER_KIND2','ALTER_KIND3','ALTER_KIND4']].max(axis=1) < 11
        df.loc[mask, 'd_HAS_CHILDREN_YTE10'] = 1
        
        child_cols = ['ANZ_KINDER','ALTER_KIND1','ALTER_KIND2','ALTER_KIND3','ALTER_KIND4']
        
        if drop_childcols:
            df.drop(child_cols, axis='columns', inplace=True)
            
        return df
        


    def __calc_children_features(self, s):
        """
        Description
        -----------
            uses features 'ALTER_KIND1','ALTER_KIND2','ALTER_KIND3','ALTER_KIND4', 'ANZ_KINDER' to reduce them to 
            'd_HAS_CHILDREN', 'd_HAS_CHILDREN_YTE10'


            * d_HAS_CHILDREN_YTE10 if person has children ANZ_KINDER>0
            * d_HAS_CHILDREN if person has at least one children <= 10            

        Parameters
        ----------
            s : pd.Series
                series of a particular DataFrame row containing at least these columns
                'ALTER_KIND1','ALTER_KIND2','ALTER_KIND3','ALTER_KIND4', 'ANZ_KINDER', 'd_HAS_CHILDREN', 'd_HAS_CHILDREN_YTE10'
        """        
        yte_10 = (s[['ALTER_KIND1','ALTER_KIND2','ALTER_KIND3','ALTER_KIND4']] <= 10).sum()
        

        s['d_HAS_CHILDREN'] = s['ANZ_KINDER']>0
        s['d_HAS_CHILDREN_YTE10']  = yte_10>0
        
        return s

    def __calc_child_and_teens(self, s):
        """
        Description
        -----------

        counts the number of children less 10 and greater equal than 10. I assume that for more than 5 children
        all children > 4 are older than 10. Based on the analysis this is in general true

        Parameters
        ----------
            s : pd.Series
                series of a particular DataFrame row containing at least these columns
                'ALTER_KIND1','ALTER_KIND2','ALTER_KIND3','ALTER_KIND4', 'ANZ_KINDER', 'd_NUM_CHILDREN_LESS_10', 'd_NUM_CHILDREN_GTE_10'
        """        
        less_10 = (s[['ALTER_KIND1','ALTER_KIND2','ALTER_KIND3','ALTER_KIND4']] < 10).sum()
        gte_10 = s['ANZ_KINDER'] - less_10

        s['d_NUM_CHILDREN_LESS_10'] = less_10
        s['d_NUM_CHILDREN_GTE_10']  = gte_10
        
        return s


        
        
        
        

class FeatureBuilder:
    """
    Description
    -----------

    executes some data transformations on a arvato dataset and creates some new features
    """
    
    def __init__(self):
        pass

    def transform(self, df:pd.DataFrame)->pd.DataFrame:
        """
        Description
        -----------

        executes the data transformation 

        Parameters
        ----------
            df : pd.DataFrame
                pandas DataFrame that is to be cleaned. Frame is expected to have columns as AZDIAS or CUSTOMERS                


        """
        self.__build_features_chidren(df)

        return df
    
    def fit (self, df:pd.DataFrame)->pd.DataFrame:
        """
        Description
        -----------
        
        prepare data for transformation
        """
        pass

    def __build_features_chidren(self, df:pd.DataFrame, drop_childcols:bool = True)->pd.DataFrame:
        """
        Description
        -----------
        
        This function will build some features based on the given input data

        * Children and Teens: 
            * Children:= number of children younger or equal than 10
            * Teens   := number of children older or equal than 10

        Parameters
        ----------
            df : pd.DataFrame
                pandas DataFrame that is to be cleaned. Frame is expected to have columns as AZDIAS or CUSTOMERS                
        """
        
        # num of children > 0
        df['d_HAS_CHILDREN'] = 0
        # younger than or equal 10
        df['d_HAS_CHILDREN_YTE10'] = 0

        df.loc[df['ANZ_KINDER'] > 0, 'd_HAS_CHILDREN'] = 1

        # mask to filter rows that have at least one record
        mask = df[['ALTER_KIND1','ALTER_KIND2','ALTER_KIND3','ALTER_KIND4']].max(axis=1) < 11
        df.loc[mask, 'd_HAS_CHILDREN_YTE10'] = 1
        
        child_cols = ['ANZ_KINDER','ALTER_KIND1','ALTER_KIND2','ALTER_KIND3','ALTER_KIND4']
        
        if drop_childcols:
            df.drop(child_cols, axis='columns', inplace=True)
            
        return df
        


    def __calc_children_features(self, s):
        """
        Description
        -----------
            uses features 'ALTER_KIND1','ALTER_KIND2','ALTER_KIND3','ALTER_KIND4', 'ANZ_KINDER' to reduce them to 
            'd_HAS_CHILDREN', 'd_HAS_CHILDREN_YTE10'


            * d_HAS_CHILDREN_YTE10 if person has children ANZ_KINDER>0
            * d_HAS_CHILDREN if person has at least one children <= 10            

        Parameters
        ----------
            s : pd.Series
                series of a particular DataFrame row containing at least these columns
                'ALTER_KIND1','ALTER_KIND2','ALTER_KIND3','ALTER_KIND4', 'ANZ_KINDER', 'd_HAS_CHILDREN', 'd_HAS_CHILDREN_YTE10'
        """        
        yte_10 = (s[['ALTER_KIND1','ALTER_KIND2','ALTER_KIND3','ALTER_KIND4']] <= 10).sum()
        

        s['d_HAS_CHILDREN'] = s['ANZ_KINDER']>0
        s['d_HAS_CHILDREN_YTE10']  = yte_10>0
        
        return s

    def __calc_child_and_teens(self, s):
        """
        Description
        -----------

        counts the number of children less 10 and greater equal than 10. I assume that for more than 5 children
        all children > 4 are older than 10. Based on the analysis this is in general true

        Parameters
        ----------
            s : pd.Series
                series of a particular DataFrame row containing at least these columns
                'ALTER_KIND1','ALTER_KIND2','ALTER_KIND3','ALTER_KIND4', 'ANZ_KINDER', 'd_NUM_CHILDREN_LESS_10', 'd_NUM_CHILDREN_GTE_10'
        """        
        less_10 = (s[['ALTER_KIND1','ALTER_KIND2','ALTER_KIND3','ALTER_KIND4']] < 10).sum()
        gte_10 = s['ANZ_KINDER'] - less_10

        s['d_NUM_CHILDREN_LESS_10'] = less_10
        s['d_NUM_CHILDREN_GTE_10']  = gte_10
        
        return s
    