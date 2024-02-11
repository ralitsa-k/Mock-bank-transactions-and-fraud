# %% Preparing the absolute path
import sys 
sys.path.insert(1, "C:/Users/Ralitsa/OneDrive - Sopra Steria/SynData/synthtrials/SyntheticData/Twitter/")
sys.path.insert(1, "C:/Users/Ralitsa/OneDrive - Sopra Steria/SynData/synthtrials/SyntheticData/UseCase/")

# %% uploading packages

import pandas as pd
import matplotlib as plt
import numpy as np
from functions import data_util
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from functions.vae import get_vae
from functions.wgangp import get_wgan_gp
from sklearn.utils import shuffle
from tensorflow import keras
import tensorflow as tf
from functions import get_path
import seaborn as sns
from scipy import stats
from plotnine import *
curr_path = get_path.get_path_f('Twitter')


# %%
tf.random.set_seed(
    321
)
df = pd.read_csv(curr_path + 'SyntheticFraud/OutputData/data_with_classified_scam.csv', index_col=0)
df.head()

# %%
#df_drop = df.drop(columns=['transaction_id','bank','case_value'])
# Decide to keep case value
df_drop = df.loc[:,['Descriptions', 'Amount', 'Category', 'date', 'customer_id', 'type',
       'is_scam_transaction', 'fraud_type', 'case_id', 'transaction_id',
       'month', 'customer_scammed', 'In_or_Out', 'bank_to', 'bank_from']]
df_drop.head()



# %%
nocid_vae_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
nocid_vae = get_vae(len(df_drop.columns),nocid_vae_optimizer,latent_dim=6, lam=0.008)
nocid_vae_hist = nocid_vae.fit(np.array(df_drop), epochs = 5, batch_size = 64)
np.save(curr_path+'/SynthethicFraud/VAE/nocid_vae_hist.npy',nocid_vae_hist.history)
nocid_vae.save_weights(curr_path+'//SynthethicFraud/VAE//nocid_vae_weights')

# %%
noise = tf.random.normal(shape = (len(df_fe_nocid.index), 6), seed = 123)
nocid_vae_gen_raw = nocid_vae.decoder.predict(noise)


# %%
nocid_vae_gen_df_raw = pd.DataFrame(nocid_vae_gen_raw, columns=df_fe_nocid.columns)
nocid_vae_gen_df_raw.to_csv(curr_path+'/UseCase/nocid_vae_gen_df_raw.csv')
nocid_vae_gen_df_pro = data_util.data_reverse_transform(df_fe_nocid,nocid_vae_gen_df_raw)
nocid_vae_gen_df_pro.describe()

# %%
df_fe_nocid.describe()

# %%
def fix_mult_class(df:pd.DataFrame, df_pre:pd.DataFrame, return_err = True):
    if 'case_id' in df.columns:
        df_pre_type = df_pre.drop(columns=['transaction_value','case_value','number_of_transactions_by_case','case_id'])
        df_type = df.drop(columns=['transaction_value','case_value','number_of_transactions_by_case','case_id'])
    else:
        df_pre_type = df_pre.drop(columns=['transaction_value','case_value','number_of_transactions_by_case'])
        df_type = df.drop(columns=['transaction_value','case_value','number_of_transactions_by_case'])
    df_new_type = df_type.copy()
    df_rowsum = (df_type).sum(axis=1)
    df_notone = (df_rowsum != 1).astype(int)
    notone_cnt = df_notone.sum()
    for i in range(len(df.index)):
        if df_notone[i] == 1:
            max_index = np.argmax(np.array(df_pre_type.iloc[i]))
            new_col = np.zeros(7)
            
            new_col[max_index] = 1
            df_new_type.iloc[i] = new_col
    if 'case_id' in df.columns:
        new_df = pd.concat([df[['transaction_value','case_value','number_of_transactions_by_case','case_id']], df_new_type],axis=1)
    else:
        new_df = pd.concat([df[['transaction_value','case_value','number_of_transactions_by_case']], df_new_type],axis=1)
    if return_err:
        df_zero = (df_rowsum == 0).astype(int)
        zero_cnt = df_zero.sum()
        twoplus_count = notone_cnt - zero_cnt
        err_list = [notone_cnt, zero_cnt, twoplus_count]
        return new_df, err_list
    else:
        return new_df        

# %%
nocid_vae_gen_df_fin, nocid_vae_err = fix_mult_class(nocid_vae_gen_df_pro,nocid_vae_gen_df_raw)
nocid_vae_err_df = pd.DataFrame(nocid_vae_err)
nocid_vae_err_df.to_csv(curr_path+'/UseCase/nocid_vae_err_df.csv')
nocid_vae_gen_df_fin.to_csv(curr_path+'/UseCase/nocid_vae_gen_df_fin.csv')

# %%
nocid_vae_gen_df_fin.describe()

# %%
vae_gen_notone = (nocid_vae_gen_df_fin.drop(columns=['transaction_value','case_value','number_of_transactions_by_case'])).sum(axis=1)
vae_gen_notone = (vae_gen_notone != 1).astype(int)
vae_gen_notone.sum()

# %%
vae_nocid_merged_df = pd.concat([df_fe_nocid, nocid_vae_gen_df_fin], axis=0)
vae_nocid_merged_df.describe()

# %%
vae_nocid_merged_scaled = StandardScaler().fit_transform(np.array(vae_nocid_merged_df))
vae_nocid_tsne = TSNE(random_state=123).fit_transform(vae_nocid_merged_scaled)
vae_nocid_tsne_orig = vae_nocid_tsne[:len(df_fe_nocid)]
vae_nocid_tsne_gen = vae_nocid_tsne[len(df_fe_nocid):]
np.save(curr_path+'//UseCase/vae_nocid_tsne_orig.npy',vae_nocid_tsne_orig)
np.save(curr_path+'//UseCase/vae_nocid_tsne_gen.npy',vae_nocid_tsne_gen)

# %%
cid_vae_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
cid_vae = get_vae(len(df_fe_cid.columns),cid_vae_optimizer,latent_dim=7, lam=0.003)
cid_vae_hist = cid_vae.fit(np.array(df_fe_cid_scaled), epochs = 5, batch_size = 64)
np.save(curr_path+'//UseCase/cid_vae_hist.npy',nocid_vae_hist.history)
nocid_vae.save_weights(curr_path+'//UseCase/cid_vae_weights')

# %%
noise = tf.random.normal(shape = (len(df_fe_cid.index), 7), seed = 123)
cid_vae_gen_raw = cid_vae.decoder.predict(noise)
cid_vae_gen_df_raw = pd.DataFrame(cid_vae_gen_raw, columns=df_fe_cid.columns)
cid_vae_gen_df_pro = data_util.data_reverse_transform(df_fe_cid,cid_vae_gen_df_raw)
cid_vae_gen_df_raw.to_csv(curr_path+'//UseCase/cid_vae_gen_df_raw.csv')
cid_vae_gen_df_fin, cid_vae_err = fix_mult_class(cid_vae_gen_df_pro,cid_vae_gen_df_raw)
cid_vae_err_df = pd.DataFrame(cid_vae_err)
cid_vae_err_df.to_csv(curr_path+'//UseCase/cid_vae_err_df.csv')
cid_vae_gen_df_fin.to_csv(curr_path+'//UseCase/cid_vae_gen_df_fin.csv')
cid_vae_gen_df_fin.describe()

# %%
vae_cid_merged_df = pd.concat([df_fe_cid, cid_vae_gen_df_fin], axis=0)
vae_cid_merged_scaled = StandardScaler().fit_transform(np.array(vae_cid_merged_df))
vae_cid_tsne = TSNE(random_state=123).fit_transform(vae_cid_merged_scaled)
vae_cid_tsne_orig = vae_cid_tsne[:len(df_fe_cid)]
vae_cid_tsne_gen = vae_cid_tsne[len(df_fe_cid):]
np.save(curr_path+'/UseCase/vae_cid_tsne_orig.npy',vae_cid_tsne_orig)
np.save(curr_path+'/UseCase/vae_cid_tsne_gen.npy',vae_cid_tsne_gen)

# %%
nocid_gan_g_optimizer = keras.optimizers.Adam(
    learning_rate=0.0001, beta_1=0.5, beta_2=0.9
)
nocid_gan_c_optimizer = keras.optimizers.Adam(
    learning_rate=0.0001, beta_1=0.5, beta_2=0.9
)
nocid_gan = get_wgan_gp(len(df_fe_nocid.columns),nocid_gan_c_optimizer, nocid_gan_g_optimizer)
nocid_gan_hist = nocid_gan.fit(np.array(df_fe_nocid_scaled), epochs = 20, batch_size = 64)
np.save(curr_path+'/UseCase/nocid_gan_hist.npy',nocid_gan_hist.history)
nocid_gan.save_weights(curr_path+'/UseCase/nocid_gan_weights')

# %%
noise = tf.random.normal(shape = (len(df_fe_nocid.index), 32), seed = 123)
nocid_gan_gen_raw = nocid_gan.generator.predict(noise)

# %%
nocid_gan_gen_df_raw = pd.DataFrame(nocid_gan_gen_raw, columns=df_fe_nocid.columns)
nocid_gan_gen_df_raw.to_csv(curr_path+'/UseCase/nocid_gan_gen_df_raw.csv')
nocid_gan_gen_df_pro = data_util.data_reverse_transform(df_fe_nocid,nocid_gan_gen_df_raw)
nocid_gan_gen_df_fin, nocid_gan_err = fix_mult_class(nocid_gan_gen_df_pro,nocid_gan_gen_df_raw)
nocid_gan_err_df = pd.DataFrame(nocid_gan_err)
nocid_gan_err_df.to_csv(curr_path+'/UseCase/nocid_gan_err_df.csv')
nocid_gan_gen_df_fin.to_csv(curr_path+'/UseCase/nocid_gan_gen_df_fin.csv')

# %%
gan_nocid_merged_df = pd.concat([df_fe_nocid, nocid_gan_gen_df_fin], axis=0)
gan_nocid_merged_scaled = StandardScaler().fit_transform(np.array(gan_nocid_merged_df))
gan_nocid_tsne = TSNE(random_state=123).fit_transform(gan_nocid_merged_scaled)
gan_nocid_tsne_orig = gan_nocid_tsne[:len(df_fe_nocid)]
gan_nocid_tsne_gen = gan_nocid_tsne[len(df_fe_nocid):]
np.save(curr_path+'/UseCase/gan_nocid_tsne_orig.npy',gan_nocid_tsne_orig)
np.save(curr_path+'/UseCase/gan_nocid_tsne_gen.npy',gan_nocid_tsne_gen)

# %%
cid_gan_g_optimizer = keras.optimizers.Adam(
    learning_rate=0.0001, beta_1=0.5, beta_2=0.9
)
cid_gan_c_optimizer = keras.optimizers.Adam(
    learning_rate=0.0001, beta_1=0.5, beta_2=0.9
)
cid_gan = get_wgan_gp(len(df_fe_cid.columns),cid_gan_c_optimizer, cid_gan_g_optimizer)
cid_gan_hist = cid_gan.fit(np.array(df_fe_cid_scaled), epochs = 20, batch_size = 64)
np.save(curr_path+'/UseCase/cid_gan_hist.npy',cid_gan_hist.history)
cid_gan.save_weights(curr_path+'/UseCase/cid_gan_weights')

# %%
noise = tf.random.normal(shape = (len(df_fe_nocid.index), 32), seed = 123)
cid_gan_gen_raw = cid_gan.generator.predict(noise)
cid_gan_gen_df_raw = pd.DataFrame(cid_gan_gen_raw, columns=df_fe_cid.columns)
cid_gan_gen_df_raw.to_csv(curr_path+'/UseCase/cid_gan_gen_df_raw.csv')
cid_gan_gen_df_pro = data_util.data_reverse_transform(df_fe_cid,cid_gan_gen_df_raw)
cid_gan_gen_df_fin, cid_gan_err = fix_mult_class(cid_gan_gen_df_pro,cid_gan_gen_df_raw)
cid_gan_err_df = pd.DataFrame(cid_gan_err)
cid_gan_err_df.to_csv(curr_path+'/UseCase/cid_gan_err_df.csv')
cid_gan_gen_df_fin.to_csv(curr_path+'/UseCase/cid_gan_gen_df_fin.csv')

# %%
gan_cid_merged_df = pd.concat([df_fe_cid, cid_gan_gen_df_fin], axis=0)
gan_cid_merged_scaled = StandardScaler().fit_transform(np.array(gan_cid_merged_df))
gan_cid_tsne = TSNE(random_state=123).fit_transform(gan_cid_merged_scaled)
gan_cid_tsne_orig = gan_cid_tsne[:len(df_fe_cid)]
gan_cid_tsne_gen = gan_cid_tsne[len(df_fe_cid):]
np.save(curr_path+'/UseCase/gan_cid_tsne_orig.npy',gan_cid_tsne_orig)
np.save(curr_path+'/UseCase/gan_cid_tsne_gen.npy',gan_cid_tsne_gen)


