import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv("medical_examination.csv")

# 2
df['overweight']=np.where(((df['weight']/((df['height']/100)**2)) > 25),1,0)

# 3
df[['cholesterol','gluc']]=df[['cholesterol','gluc']].applymap(lambda x: 0 if x == 1 else 1 )


# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df , id_vars=['cardio'] , value_vars=['cholesterol','gluc','smoke','alco','active','overweight'], var_name='variable', value_name='count')


    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'count']).size().reset_index(name='total')

    

    # 7



    # 8
    fig = sns.catplot(data=df_cat,x='variable' , y='total', hue='count' ,col = 'cardio' , kind='bar' )
    axes = fig.axes.flatten()
    for ax in axes:
        ax.set_xlabel("variable")
        ax.set_ylabel("total")

    # 9
    fig.savefig('catplot.png')
    return fig.figure



# 10
def draw_heat_map():
    # 11
    df_heat = df.loc[(df['ap_lo'] <= df['ap_hi']) & 
                     (df['height'] >= df['height'].quantile(0.025)) & 
                     (df['height'] <= df['height'].quantile(0.975)) & 
                     (df['weight'] >= df['weight'].quantile(0.025)) & 
                     (df['weight'] <= df['weight'].quantile(0.975)) ]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr , dtype=np.bool_))



    # 14
    fig, ax = plt.subplots(figsize=(10,10))

    # 15

    sns.heatmap(corr , annot=True ,cmap='coolwarm' , mask=mask,linewidths=0.5 ,fmt=".1f", ax=ax)

    # 16
    fig.savefig('heatmap.png')
    return fig
