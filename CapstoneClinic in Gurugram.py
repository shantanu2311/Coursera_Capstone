#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[364]:


import requests # library to handle requests
import pandas as pd # library for data analsysis
import numpy as np # library to handle data in a vectorized manner

get_ipython().system('pip install geopy ')
from geopy.geocoders import Nominatim # module to convert an address into latitude and longitude values


# In[365]:


import urllib.request
url = "https://finkode.com/hr/gurgaon.html"
page = urllib.request.urlopen(url)

from bs4 import BeautifulSoup 
soup = BeautifulSoup(page, "html5lib")


# In[367]:


tables=soup.find_all("table") 

for table in tables:
    head = table.find_all('caption')
    headings = [h2.text.strip() for h2 in head]
    if headings == ['List of Post Offices/ Pincodes in areas under Gurgaon district, Haryana']:
        break
#table


# In[147]:


A=[]
B=[]

for row in table.findAll('tr'):
    cells=row.findAll('td')
    if len(cells)==2:
        A.append(cells[0].findAll(text=True))
        B.append(cells[1].findAll(text=True))

df=pd.DataFrame(A,columns=['Area'])
df['Pincode']=B
df.head()


# In[148]:


df['Pincode'] = df['Pincode'].str[0]
df


# In[149]:


#df['Pincode'] = df['Pincode'].astype(str)+',India'
df.head()


# In[130]:


get_ipython().system('pip install pgeocode')


# In[150]:


import pgeocode
lis  = df.loc[:,"Pincode"]
lis2 = lis.tolist()
nominatim = pgeocode.Nominatim('IN') #Country code


# In[151]:


df2 = nominatim.query_postal_code(lis2)
df2


# In[157]:


df2.rename(columns = {'postal_code':'Pincode'},inplace=True)
df3= pd.merge(df,df2,on ='Pincode', how ='inner')
df3.head()


# In[158]:


df4 = df3.filter(['Area','Pincode','community_name','latitude','longitude'], axis=1)
df4.head()


# In[159]:


df5= df4.drop_duplicates(subset='Area', keep='first')
df4


# In[160]:


df5=df5.dropna()
df5 = df5.reset_index(drop=True)
df5


# In[161]:


df6=df5.sort_values("Pincode")
df6 = df6.reset_index(drop=True)
df6.head(89)


# In[162]:


df7=df6.iloc[5:35,]
df7


# In[171]:


CLIENT_ID = 'SVXBQLDXOARVTQR2OG3JQFHBQK3PAVHOTN4TBLLMT1TC1HIG' # your Foursquare ID
CLIENT_SECRET = 'PCK52X4UC0TBKAAVPITUDTWPCCTVPB0INCUWS2HQA2RA1UM0' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

LIMIT = 15 # limit of number of venues returned by Foursquare API
radius = 3000 #3Km radius


# In[197]:


# tranforming json file into a pandas dataframe library
from pandas.io.json import json_normalize
search_query_list = ['Clinic','Hospital']
print(type(search_query_list))
venues_count=[]
lat=[]
long=[]
LIMIT = 300
for ind in df6.index:
    latitude = df6['latitude'][ind]
    longitude = df6['longitude'][ind]
    for ind2 in search_query_list:
        search_query=ind2
        url = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&v={}&query={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION, search_query, radius, LIMIT)
        results = requests.get(url).json()
        venues = results['response']['venues']
        dataframe = json_normalize(venues)
        val_count = len(dataframe.index)
            venues_count.append(val_count)
        lat.append(latitude)
        long.append(longitude)


# In[311]:


num_count = np.array(venues_count)
headings_queries = search_query_list
reshaped_output = num_count.reshape(len(df6),len(headings_queries))

#print(reshaped)
foursquare_output = pd.DataFrame(reshaped_output, columns=headings_queries)
#lat
#foursquare_output['latitude']=lat
#foursquare_output['longitude']=long
df_dum=foursquare_output


# In[310]:


print(type(foursquare_output))
print(foursquare_output.shape)
df6.head()


# In[285]:


num_count1 = np.array(lat)
reshaped_output1 = num_count1.reshape(len(df6),len(headings_queries))
#reshaped_output1 
num_count2 = np.array(long)
reshaped_output2 = num_count2.reshape(len(df6),len(headings_queries))
#reshaped_output2


# In[286]:


foursquare_output1 = pd.DataFrame(reshaped_output1)
foursquare_output2 = pd.DataFrame(reshaped_output2)
#foursquare_output['latitude'] = reshaped_output1.tolist()
#foursquare_output['longitude'] = reshaped_output2.tolist()


# In[287]:


#foursquare_output=pd.concat([foursquare_output,foursquare_output1,foursquare_output2],axis=1) 
#foursquare_output=foursquare_output.drop(foursquare_output.iloc[:,3:4], axis=1)
name_list = ['Clinic','Hospital','latitude','longitude']
foursquare_output.columns= name_list
foursquare_output.head()


# In[288]:


df7=pd.merge(foursquare_output,df6,on ='latitude', how ='inner')
df7.shape


# In[291]:


#df7.groupby('community_name').count()
df8=df7.drop_duplicates(subset='community_name', keep='first')
#df8.rename(columns={'longitude_y':'longitude'},inplace=True)
df8.head()


# In[296]:


df8 = df8.reset_index(drop=True)
df8.shape
df9=df8.drop(["longitude_x"],axis=1)
df9.head()


# In[318]:


df9=df9[['community_name','Pincode','Area','Clinic', 'Hospital','latitude','longitude']]
df9=df9.reset_index(drop=True)
df10=df9


# In[305]:


# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

get_ipython().system('pip install folium')
import folium # map rendering library

address = 'Gurugram'
geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
#print('The geograpical coordinate of Gurgaon are {}, {}.'.format(latitude, longitude))

# create map of Gurgaon using latitude and longitude values
map_Gurgaon = folium.Map(location=[latitude, longitude], zoom_start=14)

map_Gurgaon


# In[312]:


# add markers to map
for lat, lng, community in zip(df9['latitude'], df9['longitude'], df9['community_name']):
    label = '{}'.format(community)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='red',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_Gurgaon)  
    
map_Gurgaon


# In[324]:


df9.head()


# In[346]:


# set number of clusters
kclusters =4
df10=df9[['Clinic','Hospital']]
# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(df10)

# check cluster labels generated for each row in the dataframe
kmeans.labels_


# In[347]:


# add clustering labels
df10.insert(0, 'Cluster Labels', kmeans.labels_)
df10.head()


# In[357]:


#df10=df10.merge(df9, how='outer')
#df11=df10.drop_duplicates(subset='community_name', keep='first')
df11.sort_values(by=['Cluster Labels'], inplace=True)
df11=df11.reset_index(drop=True)
df11


# In[362]:


x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(df11['latitude'], df11['longitude'], df11['Area'], df11['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_Gurgaon)
       
map_Gurgaon

