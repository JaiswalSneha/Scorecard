# SFTL
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import streamlit as st
from sklearn.metrics import roc_auc_score
# st.set_page_config(layout="wide")

st.markdown("## 📊 Credit Risk Scorecard Model")

st.markdown("""
Developed an end-to-end **credit scorecard model** using application data, incorporating:

- WOE-IV based feature engineering  
- Fine and coarse binning  
- Logistic Regression for Probability of Default (PD)  

The model is further enhanced using **Reject Inference (Weighted Augmentation)**  
to estimate risk for rejected applicants.

### 🎯 Key Outputs:
- Probability of Default (PD)  
- Credit Scorecard  
- Model Performance (AUC, KS)  
- Business insights (Approval Rate vs Bad Rate trade-off)
""")

st.markdown("---")
st.markdown("## 📁 Dataset Source")

st.markdown("""
💳 **Credit Card Default – Reject Inference Project**

This dataset contains real-world inspired credit application data used for:
- Credit risk modeling  
- Scorecard development  
- Reject inference techniques  

[🔗 Click here to view dataset on Kaggle](https://www.kaggle.com/code/shraddhacodes/credit-card-default-reject-inference-project/input)
""")


st.markdown(
    """
    <style>
    .stApp {
        background-color: #E3F2FD;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(
    # page_title="Credit Scorecard Dashboard",
    # page_title="OM",
    page_icon="💳",
    layout="wide"
)


# =========================================================================================
# =================================== APPROVED_DATA =======================================
# =========================================================================================


df = pd.read_excel('approved.xlsx')
# df['amt'] = np.log(df['amt'])
df['amt'] = np.log(df['amt'].clip(lower=1))

# st.title('Finally Completed ! Thank You God for the 80%')
# st.title('💳 Credit Scorecard Project')

st.markdown('---')
st.header('Approved Loans')
st.markdown('---')
st.markdown('### Shape of the Approved dataset')
st.write(f'Number of Columns in Approved dataset : {df.shape[1]}')
st.write(f'Number of Rows in Approved dataset : {df.shape[0]}')
st.markdown('---')
st.markdown('### Sample Approved Loans')
st.text('With all the columns = Behavioral + Application')
st.dataframe(df)

st.markdown('---')
st.header('List of variables present')
st.text(df.columns)
df = df[['rej', 'tax_code', 'amt', 'acc_amt', 'annual_income_amt', 'education','emp_yr_cnt', 'gender', 'marital_status', 'residence','employment_status_cd','tgt_var','appl_pa_leg_judg_flg','appl_pa_bnkr_sts_cd','appl_pa_mnts_flg']]
st.markdown('---')
col1, col2 = st.columns(2)
with col1:
    st.markdown('#### Behavioral Variables')
    behavioral = ['acc_1_30dlq_lst_3m_cnt','acc_31_60dlq_lst_3m_cnt','acc_61_90dlq_lst_3m_cnt','acc_91_120dlq_lst_3m_cnt','acc_dld_pay_lst_3m_cnt','acc_1_30dlq_lst_3m_amt','tot_outstanding_31_60_day_amt','acc_61_90dlq_lst_3m_amt','outcome_cd','sln_dr_trns_lst_3m_cnt','acc_91_120dlq_lst_3m_amt','appl_scr_no','acc_appl_pcl_val_amt','appl_pcl_typ_cd','appl_pa_hhd_inc_amt','appl_pa_lqd_ast_amt','appl_pa_rest_amt','appl_pa_ast_oth_amt','appl_pa_lbl_rest_amt','appl_pa_leg_judg_flg','appl_pa_bnkr_sts_cd','appl_pa_mnts_flg','appl_appt_max_age_no','appl_appt_max_lbl_amt','appl_outcm_cd','appl_pa_bur1_bnkp_cnt','appl_pa_bur2_bnkp_cnt','appl_pa_bur1_curr_lmt_amt']
    st.text(behavioral)
with col2:
    st.markdown('#### Application variables')
    st.text(df.columns)

st.info("💡 Keeping only the Application variables as Behavioral variables aren’t used in application scorecards because they are only observed after the account is active, so they aren’t available at the time of application and using them would cause data leakage.")
st.markdown('---')

# ======================================Default column ========================

st.header('Target Variable')
st.text("Renaming the Target Variable from 'tgt_var' to 'Default'")
df = df.rename(columns = {'tgt_var':'Default'})

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('#### Unique Target Status')
    st.dataframe(df['Default'].unique())
    st.markdown('#### Count of Defaults(1) and Non-Defaults(0)')
    st.dataframe(df['Default'].value_counts())
with col2:
    counts = df['Default'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(counts, labels = ['Non-Default','Default'], autopct='%1.1f%%')
    st.markdown('### % of Defaults and Non-Defaults')
    st.pyplot(fig)
with col3:
# Nulls check
    st.markdown('#### Is Null')
    st.text('Null values in the data')
    nulls_df = df.isnull().sum().reset_index().rename(columns = {'index':'column_name',0:'nulls'})
    # st.dataframe(nulls_df)
    # st.text('Columns with null values')
    st.dataframe(nulls_df[nulls_df['nulls']>0].sort_values('nulls',ascending=False))
st.markdown('---')





# # =================== missing values====================================
miss = nulls_df[nulls_df['nulls']>0]['column_name'].values

st.header('Filling the Missing Values')
st.text('Relacing the nulls in Numerical columns with Mean of that column')
st.text("Relacing the nulls in Categorical columns with 'Missing' value")
num_cols = df.drop(columns = ['Default','rej']).select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()


for i in miss:
    df[f'{i}_is_missing'] = df[i].isna().astype(int)
    if i in num_cols:
        df[i] = df[i].fillna(df[i].mean())
    else:
        df[i] = df[i].fillna('Missing')

st.markdown('---')
# # =================== EDA ====================================

st.header('Exploratory Data Analysis')

#---------- MIN MAX MEAN ----------
st.markdown('### Min-Mean-Max')
st.dataframe(df.describe().loc[['min','mean','max']])
st.markdown('---')

st.markdown('#### rej')
st.text(f"Unique rej values in the dataset : {df['rej'].value_counts().shape[0]}")
st.write('Looks like ID column ->> Dropping - of no use')
df = df.drop(columns=['rej'])


col1, col2,col3 ,col4 = st.columns(4)
with col1:
    st.markdown('#### Tax Code')
    st.text('(tax_code)')
    fig, ax = plt.subplots()
    ax.bar(df['tax_code'].value_counts().index,df['tax_code'].value_counts())
    st.pyplot(fig)
with col2:
    st.markdown('#### Employment Year Count')
    st.text('(emp_yr_cnt)')
    fig, ax = plt.subplots()
    ax.bar(df['emp_yr_cnt'].value_counts().index,df['emp_yr_cnt'].value_counts())
    st.pyplot(fig)


st.markdown('---')
# without log amount and amount_icome
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('#### Amount')
    st.text('(amt)')
    fig, ax = plt.subplots()
    ax.hist(np.exp(df['amt']).dropna())
    st.pyplot(fig)
with col2:
    st.markdown(' ###')
    st.markdown(' #####')
    fig, ax = plt.subplots()
    ax.boxplot(np.exp(df['amt']).dropna())
    st.pyplot(fig)
with col3:
    st.markdown('#### Annual Income Amount')
    st.text('(annual_income_amt)')
    fig, ax = plt.subplots()
    ax.hist(df['annual_income_amt'])
    st.pyplot(fig)
with col4:
    st.markdown(' ###')
    st.markdown(' #####')
    fig, ax = plt.subplots()
    ax.boxplot(df['annual_income_amt'].dropna())
    st.pyplot(fig)

# with log amount and amount_icome
st.info('Taking Log as the values are skewed')
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('#### Amount')
    fig, ax = plt.subplots()
    ax.hist(df['amt'])
    st.pyplot(fig)
with col2:
    st.markdown(' ###')
    fig, ax = plt.subplots()
    ax.boxplot(df['amt'].dropna())
    st.pyplot(fig)
with col3:
    st.markdown('#### Annual Income Amount')
    fig, ax = plt.subplots()
    ax.hist(np.log(df['annual_income_amt'].dropna()))
    st.pyplot(fig)
with col4:
    st.markdown(' ###')
    fig, ax = plt.subplots()
    ax.boxplot(np.log(df['annual_income_amt'].dropna()))
    st.pyplot(fig)
st.markdown('---')


# df['annual_income_amt'] = np.log(df['annual_income_amt'])
df['annual_income_amt'] = np.log(df['annual_income_amt'].clip(lower=1))

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('#### Education')
    st.text('(education)')
    fig, ax = plt.subplots()
    ax.bar(df['education'].value_counts().index,df['education'].value_counts())
    st.pyplot(fig)
with col2:
    st.markdown('#### Gender')
    st.text('(gender)')
    fig, ax = plt.subplots()
    ax.bar(df['gender'].value_counts().index,df['gender'].value_counts())
    st.pyplot(fig)
with col3:
    st.markdown('#### Marital Status')
    st.text('(marital_status)')
    fig, ax = plt.subplots()
    ax.bar(df['marital_status'].value_counts().index,df['marital_status'].value_counts())
    st.pyplot(fig)
with col4:
    st.markdown('#### Residence')
    st.text('(residence)')
    fig, ax = plt.subplots()
    ax.bar(df['residence'].value_counts().index,df['residence'].value_counts())
    st.pyplot(fig)


#---------------------- appl_pa_bnkr_sts_cd, appl_pa_leg_judg_flg, appl_pa_mnts_flg----------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('#### Bankruptcy status code')
    st.text('(appl_pa_bnkr_sts_cd)')
    fig, ax = plt.subplots()
    ax.bar(df['appl_pa_bnkr_sts_cd'].value_counts().index,df['appl_pa_bnkr_sts_cd'].value_counts())
    st.pyplot(fig)
with col2:
    st.markdown('#### Legal judgment flag')
    st.text('(appl_pa_leg_judg_flg)')
    fig, ax = plt.subplots()
    ax.bar(df['appl_pa_leg_judg_flg'].value_counts().index,df['appl_pa_leg_judg_flg'].value_counts())
    st.pyplot(fig)
with col3:
    st.markdown('#### Monetary default/obligation flag')
    st.text('(appl_pa_mnts_flg)')
    fig, ax = plt.subplots()
    ax.bar(df['appl_pa_mnts_flg'].value_counts().index,df['appl_pa_mnts_flg'].value_counts())
    st.pyplot(fig)


st.markdown('---')
st.info('Correlation is checked after WOE transformation and IV-based variable selection, as WOE ensures linearity and makes correlation meaningful for logistic regression. Highly correlated variables are pruned to avoid multicollinearity')
st.markdown('---')
# # ========================== Creating bins =============================
import woe as woe

st.header('Creating Fine bins and Missing Columns for variables for Null data')

def fine_binning(df, num_vars=[], cat_vars=[]):
    df_binned = df.copy()
    
    for col in num_vars:
        df_binned[f'{col}_fine_bin'] = pd.qcut(df_binned[col], q=10, duplicates='drop')
        
    for col in cat_vars:
        bin_col = col + "_fine_bin"
        df_binned[bin_col] = df_binned[col].fillna('Missing')
    
    return df_binned


st.markdown('#### Numerical Columns')
st.text(num_cols)
st.markdown('#### Categorical Columns')
st.text(cat_cols)

df_final = fine_binning(df,num_cols,cat_cols)

# ============================================== Safe Display =========================================
def safe_display(df):
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, pd.Interval)).any():
            df[col] = df[col].astype(str)
    return df
# =====================================================================================================


st.dataframe(safe_display(df_final))

st.info('Fine Binning (using pd.qcut): It splits the numeric column into 10 equal-sized bins based on quantiles, creating a new categorical column showing which range each value falls into')

st.markdown('---')


# #------------------------ Missing Values column -------------------------

st.header('Flagging the Missing Values')
st.markdown('#### Created Missing value columns')
df_final_miss = [col for col in df_final.columns if col.endswith('is_missing')]
st.text(df_final_miss)

st.markdown('### Example')
col1 , col2, col3, col4 = st.columns(4)
with col1:
    st.dataframe(df_final[df_final['amt_is_missing']==1][['amt','amt_is_missing']])
with col2:
    st.dataframe(df_final[df_final['annual_income_amt_is_missing']==1][['annual_income_amt','annual_income_amt_is_missing']])
with col3:
    st.dataframe(df_final[df_final['gender_is_missing']==1][['gender','gender_is_missing']])
with col4:
    st.dataframe(df_final[df_final['marital_status_is_missing']==1][['marital_status','marital_status_is_missing']])
st.markdown('---')


#----------------------------------------update later---------------------------------
st.header('Individual Bins - Unique Values')
fine_bins = [col for col in df_final.columns if col.endswith('fine_bin')]

st.text('Total Columns with Bins:')
st.text(len(fine_bins))

col11, col112, col113, col14, col15 = st.columns(5)
columns_to_choose = [col11, col112, col113, col14,col15, col11, col112, col113, col14,col15,col11, col112, col113, col14,col15,]

k=0

for col in fine_bins:
    with columns_to_choose[k]:
        st.text(col)
        st.dataframe(df_final[col].unique())
    k=k+1

st.markdown('---')
# ============================ WOE_mapping ============================================

def calculate_woe_adj(df_copy, feature, target, epsilon=0.3):
    df = df_copy.copy()
    temp = df.groupby(feature)[target].agg(['count','sum'])
    temp.columns = ['total','bad']
    temp['good'] = temp['total'] - temp['bad']
    temp['good_adj'] = temp['good'] + epsilon
    temp['bad_adj'] = temp['bad'] + epsilon
    total_good = temp['good_adj'].sum()
    total_bad = temp['bad_adj'].sum()
    temp['%good'] = temp['good_adj'] / total_good
    temp['%bad'] = temp['bad_adj'] / total_bad
    temp['woe'] = np.log(temp['%good'] / temp['%bad'])
    temp['iv'] = (temp['%good'] - temp['%bad']) * temp['woe']
    return temp.reset_index()

st.header('Columns in df_final data = Missing + Fine bins')
st.text('Columns incorporated till now')
st.text(df_final.columns)

df_fine_bin_col = [col for col in df_final.columns if col.endswith('fine_bin')]
# st.text(df_fine_bin_col)
target = df['Default']


st.markdown('---')

woe_dict = {}
st.header('Calculating WOE on Fine binned columns')
st.info('WOE (Weight of Evidence) is calculated as the natural log of the ratio of the proportion of goods to bads in each bin, i.e., WOE = ln(%good / %bad), with a small epsilon=0.3 added to avoid division by zero.')
for col in df_fine_bin_col:
    woe_dict[col] = calculate_woe_adj(df_final,col,'Default')
st.markdown('---')

st.header('Sample WOE and IV')
st.markdown('#### Amt')
st.text(f"Minimum value of amount {df_final['amt'].min()}")
st.text(f"Maximum value of amount {df_final['amt'].max()}")
st.dataframe(woe_dict['amt_fine_bin'].sort_values('amt_fine_bin'))


# # ================================== IV ================================================

st.markdown('---')
st.header('Information Value')
st.info('Information Value (IV) is a measure of how well a feature separates goods (non-defaults) from bads (defaults)')
st.markdown('#### IV based on Fine Bins')


iv_df = pd.DataFrame()
iv_list = []
iv_col = []

for col in woe_dict.keys():
    iv = woe_dict[col]['iv'].sum()
    iv_col.append(col.replace('_fine_bin',''))
    iv_list.append(iv)

iv_df['Variable'] = iv_col
iv_df['IV'] = iv_list

col1, col2 = st.columns(2)
with col1:
    st.dataframe(iv_df.sort_values('IV', ascending=False))

st.markdown('---')


#================================== Coarse Bin =========================================

st.header('Coarse Binning')

def convert_first_col_interval_and_sort(df):
    df_sorted = df.copy()
    col = df_sorted.columns[0]

    # Convert strings to Interval
    def str_to_interval(val):
        if isinstance(val, str) and val.startswith('(') and val.endswith(']'):
            left, right = val[1:-1].split(',')
            return pd.Interval(float(left.strip()), float(right.strip()), closed='right')
        return val  # keep 'Missing' or other strings

    df_sorted[col] = df_sorted[col].apply(str_to_interval)

    # Sort unique intervals numerically
    intervals_sorted = sorted([i for i in df_sorted[col].unique() if isinstance(i, pd.Interval)], key=lambda x: x.left)
    other_cats = [i for i in df_sorted[col].unique() if not isinstance(i, pd.Interval)]
    categories = intervals_sorted + other_cats

    # Convert to ordered categorical for proper sorting
    df_sorted[col] = pd.Categorical(df_sorted[col], categories=categories, ordered=True)

    # Sort dataframe by this column
    df_sorted = df_sorted.sort_values(by=col).reset_index(drop=True)
    
    return df_sorted

coarse_bins_df_create = {}
cols_to_do_coarse_bins_on = list(woe_dict.keys())
k=0
# copy of df_final for coarse binning
coarse_df = df_final.copy()

dummy_values = {'tax_code_fine_bin':'0,1,4,6','amt_fine_bin':'0,1,5,10','annual_income_amt_fine_bin':'0,1,7,10','emp_yr_cnt_fine_bin':'0,1,4,8,10','education_fine_bin':"GRA:HGR,PGR:UGR",'residence_fine_bin':"OTH:OWN,OTH:REN",'acc_amt_fine_bin':'0,1,5,6,10','gender_fine_bin':'','marital_status':'','employment_status_cd_fine_bin':'','appl_pa_leg_judg_flg_fine_bin':'','marital_status_fine_bin':'','appl_pa_bnkr_sts_cd_fine_bin':'','appl_pa_mnts_flg_fine_bin':''}

for col in cols_to_do_coarse_bins_on:
    # st.text(col)
    col1, col2,col3 = st.columns(3)
    with col1:
        st.header(col.replace('_fine_bin',''))
        fig, ax = plt.subplots(figsize=(4, 4))
        plot_df = convert_first_col_interval_and_sort(woe_dict[col][[col,'bad','good','woe']])
        ax.plot(plot_df[col].astype(str),plot_df['woe'], marker='o')
        ax.set_xticklabels(plot_df[col].astype(str), rotation=20)
        st.pyplot(fig)
        if col.replace('_fine_bin','') in cat_cols:
            user_input = st.text_input(f"Enter bins to merge as comma separated Green:D Green, Blue:D Blue {col.replace('_fine_bin','')}",value=dummy_values[col])
        else:
            user_input = st.text_input(f"Enter bins to merge as comma separated for {col.replace('_fine_bin','')}",value=dummy_values[col])
    with col2:
        st.markdown('### Fine Binning')
        st.dataframe(safe_display(convert_first_col_interval_and_sort(woe_dict[col][[col,'bad','good','woe']])))
    with col3:
        if col.replace('_fine_bin','') in cat_cols:
            # st.text('Category')
            if ':' in user_input:
                values_list = [x.strip() for x in user_input.split(",")]
                # st.text(values_list)
                coarse_df[col.replace('_fine_bin','_coarse_bin')] = coarse_df[col]
                for grp in user_input.split(","):
                    # st.text(f'In group {grp}')
                    new_cats, old_cats = grp.split(":")
                    new_cats = new_cats.strip()
                    old_list = [x.strip() for x in old_cats.split(",")]

                    # st.text(f"New Bin: {new_cats}, Old Bins: {old_list}")

                    for old in old_list:
                        coarse_df[col.replace('_fine_bin','_coarse_bin')] = coarse_df[col.replace('_fine_bin','_coarse_bin')].replace(old, new_cats)
            else:
                coarse_df[col.replace('_fine_bin','_coarse_bin')] = coarse_df[col]

            coarse_bins_df_create[col.replace('_fine_bin','_coarse_bin')] = calculate_woe_adj(coarse_df,col.replace('_fine_bin','_coarse_bin'),'Default')
            st.markdown('#### Coarse binning')
            st.dataframe(coarse_bins_df_create[col.replace('_fine_bin','_coarse_bin')])
        else:
            # st.text('Numerical')
            if ',' in user_input:
                st.header("Coarse binning")
                values_list = [int(x.strip()) for x in user_input.split(",")]
                temp_intervals = pd.qcut(coarse_df[col.replace('_fine_bin','')], q=10, duplicates='drop')
                bin_left_values = [i.left for i in temp_intervals.cat.categories]
                bin_left_values.append(coarse_df[col.replace('_fine_bin','')].max()) 
                # st.text(bin_left_values)
                # st.text([bin_left_values[i] for i in values_list])
                coarse_df[f"{col.replace('_fine_bin','_coarse_bin')}"] = pd.cut(coarse_df[col.replace('_fine_bin','')],bins = [bin_left_values[i] for i in values_list], include_lowest=True)
                coarse_bins_df_create[col.replace('_fine_bin','_coarse_bin')] = calculate_woe_adj(coarse_df,col.replace('_fine_bin','_coarse_bin'),'Default')
                st.dataframe(coarse_bins_df_create[col.replace('_fine_bin','_coarse_bin')])
            else:
                coarse_df[f"{col.replace('_fine_bin','_coarse_bin')}"] = coarse_df[col] 
                coarse_bins_df_create[col.replace('_fine_bin','_coarse_bin')] = calculate_woe_adj(coarse_df,col.replace('_fine_bin','_coarse_bin'),'Default')
        
        # Plotting Corase Bin Graph
        fig, ax = plt.subplots(figsize=(3, 3))
        variable_name = col.replace('_fine_bin','_coarse_bin')
        ax.plot(coarse_bins_df_create[variable_name][variable_name].astype(str),coarse_bins_df_create[variable_name]['woe'], marker='o')
        ax.set_xticklabels(coarse_bins_df_create[variable_name][variable_name].astype(str), rotation=20)
        st.pyplot(fig) 
                
    st.markdown('---')

 
st.header('Before Coarse Binning')
st.text(coarse_bins_df_create.keys())

# --------------------------------------------------------------------------------
for col in [c for c in df_final.columns if c.endswith('missing')]:
    coarse_bins_df_create[col] = calculate_woe_adj(coarse_df,col,'Default')
    # coarse_df[f'{col}_woe'] = pd.cut(coarse_df[col],bins = [1,0], include_lowest=True)
# --------------------------------------------------------------------------------

st.header('After Coarse Binning')
st.text('Considering the missing columns as well for Coarse Binning - not createing bin column but will later calculate WOE')
st.text(coarse_bins_df_create.keys())

st.info('Missing values aren’t split into fine bins because they have no numeric distribution—so they are kept as a single coarse (categorical) bin to capture their overall risk.')



# DISPLAY
st.header('Checking for Null values in Coarse bin we just created')
coarse_bin_columns = [col for col in coarse_df.columns if ((col.endswith('_coarse_bin')) | col.endswith('_missing')) ]
col1, col2 = st.columns(2)
with col1:
    st.dataframe(coarse_df[coarse_bin_columns].isnull().sum())
with col2:
    st.text('Total Nulls in all the columns in data')
    st.text(coarse_df[coarse_bin_columns].isnull().sum().sum())

st.markdown('---')

st.header('Data after Coarse Binning')
st.text('Coarse Binning column gets added in the Dataset')    
st.dataframe(coarse_df[sorted(coarse_df.columns)])


# # ================================== IV ================================================

st.markdown('---')
st.header('Information Value')
st.markdown('#### IV based on Coarse Bins')

iv_df = pd.DataFrame()
iv_list = []
iv_col = []

for col in coarse_bins_df_create.keys():
    iv = coarse_bins_df_create[col]['iv'].sum()
    iv_col.append(col.replace('_coarse_bin',''))
    iv_list.append(iv)

iv_df['Variable'] = iv_col
iv_df['IV'] = iv_list

col1, col2 = st.columns(2)
with col1:
    st.dataframe(iv_df.sort_values('IV', ascending=False))

st.markdown('---')


# =============================== Important ===================================

drop_fine_bin = [col for col in coarse_df.columns if not col.endswith('_fine_bin')]
df_final = coarse_df.copy()


# =============================== # suggest for columns to keep / drop - """CHECKBOX""" ===================================


all_columns = df_final.columns.tolist()

all_columns = [col for col in all_columns if not col.endswith('_fine_bin')]
all_columns = [col for col in all_columns if not col.endswith('_coarse_bin')]
# all_columns = [col for col in all_columns if not col.endswith('_missing')]
all_columns = [col for col in all_columns if not col.endswith('Default')]

default_columns = ['amt', 'acc_amt', 'residence', 'annual_income_amt_is_missing']

st.header("Select Columns to Keep on the basis of the IV calculated using Coarse Bins")
selected_cols = []
for col in sorted(all_columns):
    # If column is in default_columns, checkbox is checked
    if st.checkbox(col, value=(col in default_columns)):
        selected_cols.append(col)

st.markdown('---')
st.markdown("### Columns you selected to keep:")
st.text(selected_cols)

# final list for rejected data

final_list = [col.replace('_is_missing','') for col in selected_cols]



add_columns = []
for i in selected_cols:
    if i.endswith('_missing'):
        pass
    else:
        add_columns.append(f'{i}_coarse_bin')

selected_cols = selected_cols + add_columns

st.text('List of Final columns in the dataset along with their Coarse bins')
st.write('For Missing columns we dont have a separate Coarse bin, it is itself the Coarse bin')
st.text(selected_cols)


# df_final_before_selection = df_final.copy()

# Create a new dataframe with only selected columns
df_final = df_final[selected_cols]
st.dataframe(df_final)
st.markdown('---')

# ================================ MAP WOE ===============================================


coarse_df_woe_map = df_final.copy()

st.header('Shape of the DataFrame - Before Merging for WOE')
st.text(coarse_df_woe_map.shape)
# st.text(coarse_df_woe_map.columns)
# st.text(type(coarse_bins_df_create))
# st.text(coarse_bins_df_create.keys())

# ------------------- Trial and error ----------------------------

for col in coarse_bins_df_create.keys():
    # st.text(col)
    # st.dataframe(coarse_bins_df_create[col][[col,'woe']])
    for col_df in coarse_df_woe_map.columns:
        if col_df.startswith(col):
            coarse_df_woe_map = coarse_df_woe_map.merge(coarse_bins_df_create[col][[col,'woe']], how='left').rename(columns={'woe': f"{col.replace('_coarse_bin','')}_woe"})
        else:
            continue

# for col in coarse_bins_df_create.keys():
#     merge_col = col
#     woe_map = coarse_bins_df_create[col][[col, 'woe']]
    
#     coarse_df_woe_map = coarse_df_woe_map.merge(
#         woe_map,
#         on=merge_col,
#         how='left'
#     ).rename(columns={'woe': f"{col.replace('_coarse_bin','')}_woe"})




st.dataframe(coarse_df_woe_map)

m=0
for i in selected_cols:
    if i.endswith('missing') or i.endswith('_coarse_bin'):
        m=m+1
st.text(f'Number of columns for which WOE should be mapped : {m}')

st.header('Shape of the DataFrame - After Merging for WOE')
st.text(coarse_df_woe_map.shape)

df_final = coarse_df_woe_map.copy()

st.markdown('---')

# # ================================== Correlation =========================================

st.header('Calculating the correlation between variables')
st.text('On approved dataset')

woe_cols = [col for col in df_final.columns if col.endswith('_woe')]

import seaborn as sns
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    sns.heatmap(df_final[woe_cols].corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)
with col2:
    st.dataframe(df_final[woe_cols].corr())

# ########################################################################################
# st.header('Checking for nulls')
# st.dataframe(df_final.isnull().sum())

# st.dataframe(df_final[df_final['amt_woe'].isnull()])
# ########################################################################################

df_final['Default'] =df['Default']


# ### ================================ Logistic Regression ========================================


st.markdown('---')
st.header('Logistic Regression - using Exhaustive Model Search')

# ********************************************************************************************************************

import time

start_time = time.time()

from itertools import combinations
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

woe_miss_cols = [col for col in df_final.columns if (col.endswith('_woe'))]

# Example dataset
X = df_final[woe_miss_cols]
y = df_final['Default']

# Set up model
model = LogisticRegression(max_iter=1000)

best_score = 0
best_features = None

# Exhaustive search over feature subsets
for k in range(1, len(X.columns)+1):  # subset sizes 1 to n
    for subset in combinations(X.columns, k):
        X_subset = X[list(subset)]
        score = cross_val_score(model, X_subset, y, cv=5, scoring='roc_auc').mean()
        if score > best_score:
            best_score = score
            best_features = subset

st.header(f"Best AUC:")
st.text(best_score)
st.header(f"Best Features: ")
st.text(best_features)
st.markdown('---')

st.text(f'Total Time taken : {time.time()- start_time}')

# ****************************************************ifFinal Columns list****************************************************************

# if st.button('Use Best Features calculated using Exhaustive Model search'):
#     final_columns_list = best_features
# else:
#     final_columns_list = woe_cols


# ********************************************************************************************************************

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

X_app = df_final[woe_cols]
y_app = df_final['Default']

X_train_app, X_test_app, y_train_app, y_test_app = train_test_split(X_app, y_app, test_size=0.3, random_state=42)

# st.header('Header in Order')
# st.text(X_train_app.columns)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_app, y_train_app)

# Train AUC
app_train_pred = model.predict_proba(X_train_app)[:,1]
app_train_auc = roc_auc_score(y_train_app, app_train_pred)

# Test AUC
app_test_pred = model.predict_proba(X_test_app)[:,1]
app_test_auc = roc_auc_score(y_test_app, app_test_pred)

st.header('Applying Logistic Regression')
st.markdown("#### Model Coefficients")
st.text([(i, j) for i, j in zip(X_train_app.columns, model.coef_[0])])
st.markdown("#### Approved Data - Train AUC")
st.text(app_train_auc)
st.markdown("#### Approved Data - Test AUC")
st.text(app_test_auc)


# =========================================================================================
# =================================== REJECTED_Data ============================================
# =========================================================================================

rej = pd.read_excel('rejected.xlsx')
rej['amt'] = np.log(rej['amt'])
rej['amt'] = np.log(rej['amt'].clip(lower=1))


st.markdown('---')
st.header('Rejected Loans')
st.markdown('---')
st.write(f'Number of Columns in Rejected dataset : {rej.shape[1]}')
st.write(f'Number of Rows in Rejected dataset : {rej.shape[0]}')
st.markdown('---')
st.markdown('### Sample Rejected Loans')
st.dataframe(rej)

st.markdown('---')
st.header('List of variables present in Rejection data')
st.text(rej.columns)

st.markdown('---')
st.header('Keeping Only the Columns present in the Approved dataset')
final_list = [col for col in final_list if not col.endswith('bin')]
final_list = [col for col in final_list if not col.endswith('missing')]
# final_list = final_list + ['tgt_var']

st.header('Rejected loans columns')
st.text(list(set(final_list)))

rej = rej[list(set(final_list))]

st.dataframe(rej.head())



# Nulls check
st.markdown('---')
st.header('Is Null')
col1, col2 = st.columns(2)
with col1:
    st.markdown('#### Rejected Loans')
    nulls_rej = rej.isnull().sum().reset_index().rename(columns = {'index':'column_name',0:'nulls'})
    st.dataframe(nulls_rej[nulls_rej['nulls']>0].sort_values('column_name',ascending=True))
with col2:
    st.markdown('#### Approved Loans')
    # nulls_rej = rej.isnull().sum().reset_index().rename(columns = {'index':'column_name',0:'nulls'})
    st.dataframe(nulls_df[nulls_df['nulls']>0].sort_values('column_name',ascending=True))
st.markdown('---')

# ================== missing values imputation =============================


num_cols_rej = rej.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_rej = rej.select_dtypes(include=['object', 'category']).columns.tolist()

st.text('Columns in rejection')

nulls_df_rej = rej.isnull().sum().reset_index().rename(columns = {'index':'column_name',0:'nulls'})
miss_rej = nulls_df_rej[nulls_df_rej['nulls']>0]['column_name'].values

st.header('missing values imputation')
for i in miss_rej:
    # st.text(i)
    rej[f'{i}_is_missing'] = rej[i].isna().astype(int)
    if i in num_cols_rej:
        rej[i] = rej[i].fillna(rej[i].mean())
    else:
        rej[i] = rej[i].fillna('Missing')


st.dataframe(rej)

#---------- MIN MAX MEAN ----------
st.markdown('### Min-Mean-Max')
st.markdown('#### Rejected Loans')
st.dataframe(rej[list(set(final_list))].describe().loc[['min','mean','max']])
st.markdown('#### Approved Loans')
st.dataframe(df[list(set(final_list))].describe().loc[['min','mean','max']])
st.markdown('---')

if 'amt' in final_list:
    st.header('Amount')
    col1, col2,col3, col4 = st.columns(4)
    with col1:
        st.markdown('### Rejected')
        fig, ax = plt.subplots()
        ax.hist(rej['amt'])
        st.pyplot(fig)
    with col2:
        st.markdown('### Approved')
        fig, ax = plt.subplots()
        ax.hist(df['amt'])
        st.pyplot(fig)
    with col3:
        st.markdown('### Rejected')
        fig, ax = plt.subplots()
        ax.boxplot(rej['amt'].dropna())
        st.pyplot(fig)
    with col4:
        st.markdown('### Approved')
        fig, ax = plt.subplots()
        ax.boxplot(df['amt'].dropna())
        st.pyplot(fig)
    st.markdown('---')

if 'annual_income_amt' in final_list:
    st.header('Annual Income Amount')
    col1, col2,col3, col4 = st.columns(4)
    with col1:
        st.markdown('### Rejected')
        fig, ax = plt.subplots()
        ax.hist(rej['annual_income_amt'])
        st.pyplot(fig)
    with col2:
        st.markdown('### Approved')
        fig, ax = plt.subplots()
        ax.hist(df['annual_income_amt'])
        st.pyplot(fig)
    with col3:
        st.markdown('### Rejected')
        fig, ax = plt.subplots()
        ax.boxplot(rej['annual_income_amt'].dropna())
        st.pyplot(fig)
    with col4:
        st.markdown('### Approved')
        fig, ax = plt.subplots()
        ax.boxplot(df['annual_income_amt'].dropna())
        st.pyplot(fig)
    st.markdown('---')

rej['annual_income_amt'] = np.log(rej['annual_income_amt'].clip(lower=1))

# ++++++++++++++++++++++++++++++++ WOE REJECTED DATA ++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

st.header('Sample WOE collection')
st.dataframe(coarse_bins_df_create['amt_coarse_bin'])
# st.text(coarse_bins_df_create['amt_coarse_bin']['amt_coarse_bin'].dtype)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ---------------------------------------------- TRIAL ----------------------------------------------------------------------------
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def map_to_coarse_bins(df, col, coarse_bins_df, is_categorical=False):

    df = df.copy()
    if col.endswith('missing'):
        coarse_col = col
    else:
        coarse_col = col + "_coarse_bin"
    woe_col = col + "_woe"

    # ---------------------------
    # Categorical column
    # ---------------------------
    if is_categorical:
        # Fill missing
        df[coarse_col] = df[col].fillna('Missing').astype(str)

        # Known bins from coarse table
        known_bins = coarse_bins_df[coarse_col].astype(str).unique().tolist()

        # Assign 'Others' to any value not in known bins
        df[coarse_col] = df[coarse_col].apply(lambda x: x if x in known_bins else 'Others')

    # ---------------------------
    # Numerical column
    # ---------------------------
    else:
        # Assign Interval bins using pd.cut with the edges from the coarse_bins_df
        bins = list(coarse_bins_df[coarse_col].apply(lambda x: x.left)) + [coarse_bins_df[coarse_col].iloc[-1].right]
        labels = coarse_bins_df[coarse_col].astype(str).tolist()

        df[coarse_col] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)

        # Assign 'Others' to values outside range
        df[coarse_col] = df[coarse_col].cat.add_categories('Others')
        df.loc[df[col] < bins[0], coarse_col] = 'Others'
        df.loc[df[col] > bins[-1], coarse_col] = 'Others'
        df[coarse_col] = df[coarse_col].astype(str)

    # ---------------------------
    # Map WOE
    # ---------------------------
    woe_map = dict(zip(coarse_bins_df[coarse_col].astype(str), coarse_bins_df['woe']))
    woe_map['Others'] = coarse_bins_df['woe'].mean()
    df[woe_col] = df[coarse_col].map(lambda x: woe_map.get(x, 0))

    return df



temp = rej.copy()

# st.header('HIIIIIIIIIIIIIIIIIIIIIi')
# st.text(final_list)
# st.text(rej.columns)

st.markdown('#### The dictionary storing all the Caorse Bins and WOE and IV used later for mapping')
st.text(coarse_bins_df_create.keys())

num_cols_rej = rej.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_rej = rej.select_dtypes(include=['object', 'category']).columns.tolist()

for col in rej.columns:
    # st.text(f'columnam e {col}')
    if col in num_cols_rej:
        # st.text('in numbers')
        temp = map_to_coarse_bins(temp, col, coarse_bins_df_create[f'{col}_coarse_bin'], is_categorical=False)
    else:
        if col.endswith('_is_missing'):
            # st.text('in cate')
            temp = map_to_coarse_bins(temp, col, coarse_bins_df_create[col], is_categorical=True)
        else:
            # st.text('in text')
            temp = map_to_coarse_bins(temp, col, coarse_bins_df_create[f'{col}_coarse_bin'], is_categorical=True)

st.dataframe(temp[sorted(temp.columns)])

st.header('Checking for nulls')
st.dataframe(temp.isnull().sum())

rej_final = temp.copy()



### ================================ Logistic Regression ========================================

woe_cols = [col for col in rej_final.columns if col.endswith('_woe')]

rej_columns_lr = X_app.columns

X_rej = rej_final[rej_columns_lr]

rej_pred = model.predict_proba(X_rej)[:,1]

rej_final = rej_final[rej_columns_lr]


st.markdown('---')
st.header('Rejection dataset final')
st.dataframe(rej_final)
st.markdown('---')

# ====================================================================================================================================
# ============================== Credit SCore ============================================
# ====================================================================================================================================

df_final['pred_prob'] = model.predict_proba(df_final[X_app.columns])[:,1]
rej_final['pred_prob'] = rej_pred


st.header('Score')
st.markdown('---')

PDO = 10
base_score = 500
good = (df_final['Default'] == 0).sum()
bad =  (df_final['Default'] == 1).sum()

odds = good / bad

st.markdown("#### Points to Double the Odds :")
st.text(PDO)
st.markdown("#### Base Score :")
st.text(base_score)
st.markdown("#### Population odds:")
st.text(odds)


st.info('Odds is calculated basis the approved dataset by counting the Goods and Bads')
# odds = 2   # 1:20 odds

factor = PDO / np.log(2)
offset = base_score - factor * np.log(odds)

    # col1, col2 = st.columns(2)
    # with col1:
    #     st.header('Approved Dataset')
    #     st.dataframe(df_final)
    #     st.text('Columns in Approved Dataset')
    #     st.text(df_final.columns.sort_values())Keeping Only the Columns present in the Approved dataset

    # with col2:
    #     st.header('Rejected Dataset')
    #     st.dataframe(rej_final)
    #     st.text('Columns in Rejected Dataset')
    #     st.text(rej_final.columns.sort_values())


 # creating scores in the dataset
df_final['score'] = offset - factor * np.log(df_final['pred_prob'] / (1 - df_final['pred_prob']))
rej_final['score'] = offset - factor * np.log(rej_final['pred_prob'] / (1 - rej_final['pred_prob']))

st.markdown('---')

# ----++++++++++++++++++++ Plot Scores distribution +++++++++++++++++++++++++++++++++++++++++++
col1, col2 = st.columns(2)

with col1:
    st.header('Approved Scores')
    fig, ax = plt.subplots()
    ax.hist(df_final['score'], bins=50, edgecolor='k')
    ax.set_title("Score Distribution")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    st.pyplot(fig)

with col2:
    st.header('Rejected Scores')
    fig, ax = plt.subplots()
    ax.hist(rej_final['score'], bins=50, edgecolor='k')
    ax.set_title("Score Distribution")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    st.pyplot(fig)

st.markdown('---')
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from sklearn.metrics import roc_curve

# Calculate ROC
col1, col2, col3 = st.columns(3)

with col1:
    st.header('ROC Curve')
    st.header('Receiver Operating Characteristic')
    st.text('A curve that shows model performance across all thresholds.')
    fpr, tpr, thresholds = roc_curve(df_final['Default'], df_final['pred_prob'])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label="Model")
    ax.plot([0, 1], [0, 1], '--', label="Random")  # baseline
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)
with col2:
    st.header('KS Statistic')
    st.header('Kolmogorov–Smirnov')
    st.text('Maximum separation between good and bad distributions')
    ks_stat = max(tpr - fpr)
    st.text(ks_stat)
with col3:
    auc = roc_auc_score(df_final['Default'], df_final['pred_prob'])
    st.header('AUC')
    st.header('Area Under ROC Curve')
    st.text('A single number summary of the ROC curve')
    st.text(auc)

st.markdown('---')

# df_sorted = df_final.sort_values('pred_prob_comb', ascending=True)
# df_sorted['cum_good'] = (1 - df_sorted['Default']) * df_sorted['Weight']
# df_sorted['cum_bad'] = df_sorted['Default'] * df_sorted['Weight']
# df_sorted['cum_good_pct'] = df_sorted['cum_good'].cumsum() / df_sorted['cum_good'].sum()
# df_sorted['cum_bad_pct'] = df_sorted['cum_bad'].cumsum() / df_sorted['cum_bad'].sum()
# ks_weighted = max(abs(df_sorted['cum_good_pct'] - df_sorted['cum_bad_pct']))
# st.text(f"KS statistic ***NEW*** using weights:{ ks_weighted}")


# st.header('Data with Scores')
# st.dataframe(df_final)

# st.text('Min Score')
# st.text(df_final['score'].min())
# st.text('Max Score')
# st.text(df_final['score'].max())



# ================================ REJECT INFERENCING ==============================

#----------------Penalizing Factor----------------------

# woe_cols = [col for col in df_final.columns if col.endswith('_woe')]
# st.text(woe_cols)
# st.dataframe(df_final[woe_cols])
# st.dataframe(df_final)



# # bins creation

df_final['score_bin'] = pd.qcut(df_final['score'], q=10, duplicates='drop')

bin_edges = df_final['score_bin'].cat.categories
edges = [i.left for i in bin_edges] + [bin_edges[-1].right]
rej_final['score_bin'] = pd.cut(rej_final['score'], bins=edges, include_lowest=True)

# st.dataframe(df_final[['Default','score','score_bin']])

# aug_df = df_final.groupby('score_bin')['Default'].agg(['count','sum']).reset_index()
# aug_df = aug_df.rename(columns = {'count':'Accepts','sum':'Accepted_Bads'})
# aug_df['prob_bin_str'] = aug_df['score_bin'].astype(str)


st.header('Augmented DF')
st.text('Accepts = Count of Approved loans lying in the Score range')
st.text('Rejects = Count of Rejected loans lying in the Score range')
aug_df = pd.DataFrame()
aug_df = df_final.groupby('score_bin')['score'].count().reset_index().rename(columns = {'score':'Accepts'})
aug_df = aug_df.merge(rej_final.groupby('score_bin')['score'].count().reset_index().rename(columns = {'score':'Rejects'}),how='left')
aug_df['Rejects'] = aug_df['Rejects'].fillna(0)

# Convert interval objects to strings
score_bin_str = aug_df['score_bin'].astype(str)
aug_df.insert(0, 'Score_Bands', score_bin_str) 
st.dataframe(aug_df.drop(columns = 'score_bin'))
st.markdown('---')

st.header('Augmentation Factor = Accepts + Rejects / Accepts ')
aug_df['Aug_fact'] = (aug_df['Accepts']+ aug_df['Rejects'])/ aug_df['Accepts']
st.dataframe(aug_df.drop(columns = 'score_bin'))
st.markdown('---')

st.header('Actual Goods & Actual Bads')
accepted_goods = df_final[df_final['Default']==0].groupby('score_bin')['score'].count().reset_index().rename(columns = {'score':'Accepted_Goods'})
accepted_bads = df_final[df_final['Default']==1].groupby('score_bin')['score'].count().reset_index().rename(columns = {'score':'Accepted_Bads'})

aug_df = aug_df.merge(accepted_goods,how='left')
aug_df = aug_df.merge(accepted_bads,how='left')

st.dataframe(aug_df.drop(columns = 'score_bin'))
st.markdown('---')

st.header('Augmented Goods = Accepted Goods * Augmentation Factor')
st.header('Augmented Bads = Accepted Bads * Augmentation Factor')
aug_df['Aug_Goods'] = aug_df['Aug_fact'] * aug_df['Accepted_Goods']
aug_df['Aug_Bads'] = aug_df['Aug_fact'] * aug_df['Accepted_Bads']
st.dataframe(aug_df.drop(columns = 'score_bin'))
st.markdown('---')

st.header('Rejected Goods = Augmented Goods - Accepted Goods')
st.header('Rejected Bads = Augmented Bads - Accepted Bads')
aug_df['Rej_Goods'] = aug_df['Aug_Goods'] - aug_df['Accepted_Goods']
aug_df['Rej_Bads'] = aug_df['Aug_Bads'] - aug_df['Accepted_Bads']
st.dataframe(aug_df.drop(columns = 'score_bin'))
st.markdown('---')

accepted_odds = aug_df['Accepted_Goods'].sum() / aug_df['Accepted_Bads'].sum()
st.header('Accepted Goods')
st.markdown('#### Accepted Odds = Accepted Goods / Accepted Bads')
st.text(accepted_odds)
rejected_odds = aug_df['Rej_Goods'].sum() / aug_df['Rej_Bads'].sum()
st.header('Rejected Goods')
st.markdown('#### Rejected Odds = Rejected Goods / Rejected Bads')
st.text(rejected_odds)
st.markdown('---')

st.header('Penalizing Factor = Accepted Odds / Rejected Odds')
penalizing_factor = accepted_odds/rejected_odds
st.text(penalizing_factor)
st.markdown('---')

# # ============================== Creating weighted data ===================================

st.header('Creating Instances of the Rejected data with Penalizing Weights')
df_bad = rej_final.copy()

# Create the "Bad" instances
st.markdown('#### Rejected Data with Default = 1 & Weight = Predicted Probabilty * Penalizing Factor')
df_bad['Default'] = 1
df_bad['Weight'] = df_bad['pred_prob'] * penalizing_factor

# Create the "Good" instances
st.markdown('#### Rejected Data with Non Default = 0 & Weight = (1- Predicted Probabilty) * Penalizing Factor')
df_good = rej_final.copy()
df_good['Default'] = 0
df_good['Weight'] = (1 - df_good['pred_prob']) * penalizing_factor

# Combine them
st.markdown('#### Accepted data with weights = 1')
df_final['Weight'] = 1
# df_final['Default'] = df['Default']
st.markdown('---')

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('##### Columns in Rejected Data with Default = 1')
    st.text(f'df_bad = {df_bad.shape}')
    st.text(df_bad.columns.sort_values())
with col2:
    st.markdown('##### Columns in Rejected Data with Non Default = 0')
    st.text(f'df_good = {df_good.shape}')
    st.text(df_good.columns.sort_values())
with col3:
    df_final_comb = df_final[df_bad.columns]
    st.markdown('##### Columns in Approved data')
    st.text(f'df_final = {df_final_comb.shape}')
    st.text(df_final_comb.columns.sort_values())
st.markdown('---')

st.header('Shape of the Combined data - df_fuzzy')
df_join_rej = pd.concat([df_bad, df_good])
df_fuzzy = pd.concat([df_join_rej, df_final_comb])
st.text(f'combined = {df_fuzzy.shape}')
st.text(df_fuzzy.columns.sort_values())
st.markdown('---')


st.header('Checking for NULLs')
st.dataframe(df_fuzzy.isnull().sum())

st.markdown('---')
st.header('Combined dataset')
st.dataframe(df_fuzzy)


# # # ================================ Logistic Regression ========================================

st.markdown('---')
st.header("Logistic Regression for Overall loans")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

X_fuzzy = df_fuzzy[X_app.columns]
y_fuzzy = df_fuzzy['Default']
w_fuzzy = df_fuzzy['Weight']

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_fuzzy, y_fuzzy, w_fuzzy, test_size=0.3, random_state=42
)

model_final = LogisticRegression(max_iter=1000)
model_final.fit(X_train, y_train, sample_weight=w_train)

# Evaluate
y_train_pred = model_final.predict_proba(X_train)[:,1]
y_test_pred = model_final.predict_proba(X_test)[:,1]

train_auc = roc_auc_score(y_train, y_train_pred, sample_weight=w_train)
test_auc = roc_auc_score(y_test, y_test_pred, sample_weight=w_test)

st.text(f"Train AUC:,{train_auc}")
st.text(f"Test AUC:, {test_auc}")

df_fuzzy['pred_prob_comb'] = model_final.predict_proba(X_fuzzy)[:,1]

st.markdown('---')


st.header('Recomputing the Scores and the score bins')
df_fuzzy['score'] = offset - factor * np.log(df_fuzzy['pred_prob_comb'] / (1 - df_fuzzy['pred_prob_comb']))
df_fuzzy['score_bin'] = pd.qcut(df_fuzzy['score'], q=10, duplicates='drop')


st.markdown('---')
st.header('Checking for NULLs')
st.dataframe(df_fuzzy.isnull().sum())
st.markdown('---')

# =========================================================================================
# =================================== MORE_MONEY===========================================
# =========================================================================================

st.header('Acceptance Rate = Approved loans / Combined data Weights')
acceptance_rate = (df.shape[0]/(df_fuzzy['Weight'].sum()))*100
st.text(acceptance_rate)

st.header('Bad Rate = Bads in Approved Loans / Total Approved loans')
bads_app = df[df['Default']==1].shape[0]
bad_rate = (bads_app/df.shape[0])*100

st.text(bad_rate)

# st.header('Good Rate')
# st.text(100-bad_rate)

# st.markdown('---')

# # ============================== Freezing Acceptance rate ================================

score_mm = []
weight_mm = []
nearest = []

st.markdown('---')

for i in np.arange(df_fuzzy['score'].min(), df_fuzzy['score'].max(),0.1):
    score_mm.append(i)
    score_grt_weight = df_fuzzy[df_fuzzy['score']>=i]['Weight'].sum()
    overall_weight = df_fuzzy['Weight'].sum()
    weight_mm.append((score_grt_weight/overall_weight)*100)
    nearest.append(abs(acceptance_rate-((score_grt_weight/overall_weight)*100)))

mm_df_ar_cnst = pd.DataFrame()
mm_df_ar_cnst['Score'] = score_mm
mm_df_ar_cnst['Approval_Rate'] = weight_mm
# mm_df_ar_cnst['Closest_To_Acceptance_Rate'] = nearest

best_index = np.argmin(nearest)
best_score_threshold = score_mm[best_index]

st.header(' Same Acceptance Rate')
st.dataframe(mm_df_ar_cnst)
# st.text(best_index)
st.header(f'Nearest Score to the Acceptance Rate of earlier Acceptance rate')
st.text(best_score_threshold)

den_bad = df_fuzzy[df_fuzzy['score']>best_score_threshold]['Weight'].sum()
num_bad = df_fuzzy[(df_fuzzy['score']>best_score_threshold) & (df_fuzzy['Default']==1)]['Weight'].sum()
bad_rate_later = (num_bad/den_bad)*100
st.header('Bad Rate keeping Acceptance rate constant')
st.text(bad_rate_later)

if bad_rate_later<bad_rate:
    st.info(f'Earlier Bad rate {bad_rate} . Bad Rate keeping Acceptance rate constant {bad_rate_later}. The Bad Rate has now decreased.')


# # ============================== Freezing Bad rate ================================

score_mm = []
weight_mm = []
nearest = []

for i in np.arange(df_fuzzy['score'].min(), df_fuzzy['score'].max(),0.5):
    score_mm.append(i)
    num_bad = df_fuzzy[(df_fuzzy['score']>i) & (df_fuzzy['Default']==1)]['Weight'].sum()
    den_bad = df_fuzzy[df_fuzzy['score']>i]['Weight'].sum()
    weight_mm.append((num_bad/den_bad)*100)
    nearest.append(abs(bad_rate-((num_bad/den_bad)*100)))

mm_df_br_cnst = pd.DataFrame()
mm_df_br_cnst['Score'] = score_mm
mm_df_br_cnst['Approval_Rate'] = weight_mm

best_index = np.argmin(nearest)
best_score_threshold = score_mm[best_index]

st.header('Same Bad Rate')
st.dataframe(mm_df_br_cnst)
st.header(f'Nearest Score to the Bad Rate of {bad_rate}')
st.text(best_score_threshold)

score_grt_weight = df_fuzzy[df_fuzzy['score']>=best_score_threshold]['Weight'].sum()
overall_weight = df_fuzzy['Weight'].sum()
st.header('Acceptance Rate keeping the Bad rate constant')
st.text((score_grt_weight/overall_weight)*100)

# ===================================================================================================================================================
# ================================================= Cut OFF score ================================================
# ===================================================================================================================================================

# Example DataFrame
# df_final = pd.DataFrame({'pred_prob': [...], 'Default': [...]})

# Bin the scores into deciles
# df_final['score_bin'] = pd.qcut(df_final['pred_prob'], 10)

# Group by score_bin and Default
score_counts = df_fuzzy.groupby(['score_bin', 'Default']).size().unstack(fill_value=0)

# Plot
st.markdown('---')

st.header('Interpreting the cut off Score')
col1,col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(score_counts.index.astype(str), score_counts[0], marker='o', label='Non-Defaulters (0)', color='green')
    ax.plot(score_counts.index.astype(str), score_counts[1], marker='o', label='Defaulters (1)', color='red')

    ax.set_title('Score-wise Defaulters vs Non-Defaulters')
    ax.set_xlabel('Score Bins')
    ax.set_ylabel('Number of Customers')
    ax.legend()
    plt.xticks(rotation=45)

    st.pyplot(fig)