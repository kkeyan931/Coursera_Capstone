import pandas as pd
import numpy as np
import folium
import matplotlib.cm as cm
import matplotlib.colors as colors

from sklearn.cluster import KMeans

import requests
import json
from pandas.io.json import json_normalize


url="https://en.wikipedia.org/wiki/List_of_districts_of_Tamil_Nadu"

contents=pd.read_html(url)


columns=["No","District","Code","Capital","Date_Of_Formation","Split_From","Area","Population","Population_Density","Taluks","Map"]

df=contents[1]
df.columns=columns


df.drop(columns=["No","Capital","Map"],inplace=True)
df.Taluks=[i.split(" ") for i in df.Taluks] 
df.Area=df.Area.astype(str)
Area=df.Area.str.split("[",1).str[0].str.strip()
Area=df1.str.replace(",","").str.strip()
df.Area=Area
df.Area=df.Area.astype(float)


df.dtypes


df.head()


from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim


longitude = [] 
latitude = [] 
   
# function to find the coordinate 
# of a given city  
def findGeocode(city): 
       
    # try and catch is used to overcome 
    # the exception thrown by geolocator 
    # using geocodertimedout   
    try: 
        geolocator = Nominatim(user_agent="Data-Analysis") 
          
        return geolocator.geocode(city) 
      
    except GeocoderTimedOut: 
          
        return findGeocode(city)     

for i in (df["District"] +" , Tamil Nadu"): 
      
    if findGeocode(i) get_ipython().getoutput("= None: ")
           
        loc = findGeocode(i)
        latitude.append(loc.latitude) 
        longitude.append(loc.longitude) 
        
    else: 
        latitude.append(np.nan) 
        longitude.append(np.nan) 


df['Latitude']=latitude
df['Longitude']=longitude


df.head()


df.info()


df.to_csv("Tamilnadu.csv")


address = 'Tamil Nadu, India'

geolocator = Nominatim(user_agent="Data Analysis")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Tamil Nadu are {}, {}.'.format(latitude, longitude))


map_TamilNadu = folium.Map(location=[latitude, longitude], zoom_start=6.4)

# add markers to map
for lat, lng, label in zip(df['Latitude'], df['Longitude'], df['District']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=True).add_to(map_TamilNadu)  
    
map_TamilNadu


CLIENT_ID = '0URO1O34DDEKYCW0LGTEM0FYCND1KJO2EUIFKDRSGDRW3ANX' # your Foursquare ID
CLIENT_SECRET = 'KWQZIUBNSY0M1TUCU3TBYFWDXEEZUVO5WUAKUIMXBGDSQLQP' # your Foursquare Secret
VERSION = '20200605' # Foursquare API version
LIMIT = 100 # A default Foursquare API limit value

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


def getNearbyVenues(names, latitudes, longitudes, radius=5000):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['District', 
                  'District Latitude', 
                  'District Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


Tamilnadu_venues=getNearbyVenues(df["District"],
                                 df["Latitude"],
                                 df["Longitude"])


print(Tamilnadu_venues.shape)
Tamilnadu_venues.head()


Tamilnadu_venues.groupby('District').count()


Tamilnadu_onehot = pd.get_dummies(Tamilnadu_venues[['Venue Category']], prefix="", prefix_sep="")

Tamilnadu_onehot['District'] = Tamilnadu_venues['District'] 

fixed_columns = [Tamilnadu_onehot.columns[-1]] + list(Tamilnadu_onehot.columns[:-1])
Tamilnadu_onehot = Tamilnadu_onehot[fixed_columns]


Tamilnadu_onehot.head()


Tamilnadu_grouped = Tamilnadu_onehot.groupby('District').mean().reset_index()
Tamilnadu_grouped.head()


Tamilnadu_grouped.head()


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['District']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
District_venues_sorted = pd.DataFrame(columns=columns)
District_venues_sorted['District'] = Tamilnadu_grouped['District']

for ind in np.arange(Tamilnadu_grouped.shape[0]):
    District_venues_sorted.iloc[ind, 1:] = return_most_common_venues(Tamilnadu_grouped.iloc[ind, :], num_top_venues)

District_venues_sorted.head()


# set number of clusters
kclusters = 5

Tamilnadu_grouped_clustering = Tamilnadu_grouped.drop('District', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(Tamilnadu_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# add clustering labels
District_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

Tamilnadu_merged = df

# merge manhattan_grouped with manhattan_data to add latitude/longitude for each neighborhood
Tamilnadu_merged = Tamilnadu_merged.join(District_venues_sorted.set_index('District'), on='District')

Tamilnadu_merged.head() # check the last columns!


Tamilnadu_merged.shape


Tamilnadu_merged['Cluster Labels']=Tamilnadu_merged['Cluster Labels'].fillna(5)
Tamilnadu_merged['Cluster Labels']=Tamilnadu_merged['Cluster Labels'].astype(int)


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=6)

# set color scheme for the clusters
x = np.arange(kclusters+1)
ys = [i + x + (i*x)**2 for i in range(kclusters+1)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(Tamilnadu_merged['Latitude'], Tamilnadu_merged['Longitude'], Tamilnadu_merged['District'], Tamilnadu_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=2).add_to(map_clusters)
       
map_clusters


Tamilnadu_merged.loc[Tamilnadu_merged['Cluster Labels'] == 0, Tamilnadu_merged.columns[[0] + list(range(5, Tamilnadu_merged.shape[1]))]]


Tamilnadu_merged.loc[Tamilnadu_merged['Cluster Labels'] == 1, Tamilnadu_merged.columns[[0] + list(range(5, Tamilnadu_merged.shape[1]))]]


Tamilnadu_merged.loc[Tamilnadu_merged['Cluster Labels'] == 2, Tamilnadu_merged.columns[[0] + list(range(5, Tamilnadu_merged.shape[1]))]]


Tamilnadu_merged.loc[Tamilnadu_merged['Cluster Labels'] == 3, Tamilnadu_merged.columns[[0] + list(range(5, Tamilnadu_merged.shape[1]))]]


Tamilnadu_merged.loc[Tamilnadu_merged['Cluster Labels'] == 4, Tamilnadu_merged.columns[[0] + list(range(5, Tamilnadu_merged.shape[1]))]]


Tamilnadu_merged.loc[Tamilnadu_merged['Cluster Labels'] == 5, Tamilnadu_merged.columns[[0] + list(range(5, Tamilnadu_merged.shape[1]))]]



