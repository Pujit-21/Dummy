#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
from bokeh.io import output_notebook
from bokeh.plotting import figure,show
from bokeh.layouts import gridplot
from bokeh.models import LinearColorMapper, BasicTicker, PrintfTickFormatter, ColorBar
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[2]:


frame = pd.read_csv(r'C:\Users\pujit\Desktop\titanic.csv')
frame_corr = frame.corr()


# In[3]:


frame.head()


# In[4]:


output_notebook()


# In[5]:


df = pd.DataFrame(frame_corr.stack(),columns = ["corr"])
df


# In[6]:


variables = list(frame_corr.columns)
TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"
p = figure(title = "Correlation",x_axis_location="above", plot_width=900, plot_height=400,x_range= variables,
           y_range=variables,tools=TOOLS, toolbar_location='below',
           tooltips=[('Variable', '@X_t" Y: " @Y_t'), ('corr','@corr')])
colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
mapper = LinearColorMapper(palette=colors, low=-1, high=1)
df = df.reset_index()
df.columns = ["X_t","Y_t","corr"]
p.rect(x="X_t", y="Y_t", width=0.9, height=0.9,
       source=df,
       fill_color={'field': 'corr', 'transform': mapper},
       line_color=None)

color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt",
                     ticker=BasicTicker(desired_num_ticks=len(colors)),
                     label_standoff=6, border_line_color=None, location=(0, 0))
p.add_layout(color_bar, 'right')


# In[7]:


show(p)


# In[22]:


frame = frame.select_dtypes(include=[np.number])
columns = frame.columns
frame = frame[columns]
total_variable = list(frame.columns)
colour = ["yellow","brown","red","green","blue"]
total_plots = []
count = 0
print("click on legend to hide that legend")
for i in frame.columns:
    remaning = set(total_variable)-set([i])
#     print(remaning,set(total_variable))
    xy = list(remaning)
    fig = figure(plot_width=800, plot_height=250)
    fig.title.text = i
    for j,colours in zip(xy,colour):
        fig.scatter(frame[j],frame[i],alpha=0.8, legend=j,line_color = colours)
    fig.legend.location = "top_right"
    fig.legend.click_policy="hide"
    total_plots.append(fig)
grid = gridplot(total_plots, ncols=2, plot_width=650, plot_height=500)
show(grid)


# In[9]:


frame_corr


# In[24]:


columns = frame.columns
Y = frame["Survived"]
X = frame[columns[1:]] 
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.33, random_state=42)


# In[25]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
Y_pred = random_forest.predict(x_test)
random_forest.score(x_train,y_train)
res = round(random_forest.score(x_train, y_train)*100, 2)
res


# In[ ]:





# In[ ]:





# In[ ]:




