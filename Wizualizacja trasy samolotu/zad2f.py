#Filip Szafran
#325705
#nr 13

from blh2xyz import blh2xyz
import numpy as np
from read_flightradar import read_flightradar
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import pykdtree
from scipy.spatial import cKDTree
import folium as fl

# def haversine(fi1,fi2,lam1,lam2):
#     R=6371e3
#     fi1=np.deg2rad(fi1)
#     fi2=np.deg2rad(fi2)
#     lam1=np.deg2rad(lam1)
#     lam2=np.deg2rad(lam2)
#     dlam=lam2-lam1
#     dfi=fi2-fi1
#     a=np.sin(dfi/2)**2+np.cos(fi1)*np.cos(fi2)*np.sin(dlam/2)**2
#     c=2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
#     d=R*c
#     return d/1000

def Rneu(phi,lam):
    return np.array([[-np.sin(phi)*np.cos(lam),-np.sin(lam),np.cos(phi)*np.cos(lam)],
            [-np.sin(phi)*np.sin(lam),np.cos(lam),np.cos(phi)*np.sin(lam)],
            [np.cos(phi),0,np.sin(phi)]])
def timestamp2minutes(timestamp):
    return timestamp/60

file = 'C:/Users/filo1/Desktop/Geodesis_highersis/cw2/lot13.csv'
dane=read_flightradar(file)

to= dane[0,0]
flh = dane[:,[7,8,9]]
flh[:,2]= flh[:,2]*0.3048 + 135.40
flh_lotnisko = flh[0,:]
xyz_lotnisko=blh2xyz(flh_lotnisko[0],flh_lotnisko[1], flh_lotnisko[2])   
azymuty =[]
odl=[]

R=Rneu(np.deg2rad(flh_lotnisko[0]),np.deg2rad(flh_lotnisko[1]))
height = []
for i in range(len(flh)):
    fi, lam, h = flh[i] 
    xyz=blh2xyz(fi,lam,h) 
    xsl = np.array(xyz) - np.array(xyz_lotnisko) 
    RT= np.transpose(R)
    neu= np.dot(RT,xsl)
    d= (np.sqrt(neu[0]**2+neu[1]**2+neu[2]**2)/1000) 
    az = np.arctan2(neu[1],neu[0])
    eh=np.arcsin(neu[2]/np.sqrt(neu[0]**2+neu[1]**2+neu[2]**2))
    height.append(eh)
    odl.append(d) 
    azymuty.append(az) 


request = cimgt.OSM()
# fi1,lam1,_=flh[0]
# fi2,lam2,_=flh[-1]
# distance=haversine(fi1,fi2,lam1,lam2)
# print (distance)

def mapa():
    global request, flh
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=request.crs)
    extent = [-10, 23, 45, 53]
    ax = plt.axes(projection=request.crs)
    ax.set_extent(extent)
    ax.set_global()
    ax.add_image(request, 5)
    ax.stock_img()
    ax.coastlines()
    for i in range(len(height)-1):
        if height[i]>0 or i<=32:
            ax.plot([flh[i][1], flh[i+1][1]], [flh[i][0], flh[i+1][0]], transform=ccrs.Geodetic(), color='g')
        else:
            ax.plot([flh[i][1], flh[i+1][1]], [flh[i][0], flh[i+1][0]], transform=ccrs.Geodetic(), color='r')
    ax.plot([flh[0][1], flh[-1][1]], [flh[0][0], flh[-1][0]], transform=ccrs.PlateCarree(), color='black')
    plt.show()

#mapa()

def lin_wys_samolotu():
    h = np.arange(0,timestamp2minutes(dane[-1,0]-to)+30,30)
    fig= plt.figure(figsize=(10, 5))
    ax=fig.add_subplot(1,1,1)
    ax.set_title("Zależność wysokości od czasu")
    ax.set_xlabel("Czas (min)")
    ax.set_ylabel("Wysokość (m)")
    ax.set_xticks(h)
    for i in range(len(height)-1):
        if height[i]>0 or i<=32:            
            ax.plot([timestamp2minutes(dane[i,0]-to),timestamp2minutes(dane[i+1,0]-to)],[flh[i,2],flh[i+1,2]], color='green',linestyle='-',linewidth=2)
        else:
            ax.plot([timestamp2minutes(dane[i,0]-to),timestamp2minutes(dane[i+1,0]-to)],[flh[i,2],flh[i+1,2]], color='red',linestyle='-',linewidth=2)
        
    plt.show()
#lin_wys_samolotu()

def lin_predkosc_samolotu():
    h = np.arange(0,timestamp2minutes(dane[-1,0]-to)+30,30)
    fig= plt.figure(figsize=(10, 5))
    ax=fig.add_subplot(1,1,1)
    ax.set_title("Zależność prędkości od czasu")
    ax.set_xlabel("Czas (min)")
    ax.set_ylabel("Prędkość (km/h)")
    ax.set_xticks(h)
    max_speed = 0
    max_speed_time = None
    for i in range(len(height)-1):
        speed = (dane[i,10])*1.85166
        time = timestamp2minutes(dane[i,0]-to)
        if speed > max_speed:
            max_speed = speed
            max_speed_time = time
        if height[i]>0 or i<=32:
            ax.plot([time,timestamp2minutes(dane[i+1,0]-to)],[speed,(dane[i+1,10])*1.85166], color='green',linestyle='-',linewidth=2)
        else:
            ax.plot([time,timestamp2minutes(dane[i+1,0]-to)],[speed,(dane[i+1,10])*1.85166], color='red',linestyle='-',linewidth=2)
    plt.show()
    print(f"Max speed: {max_speed} at time: {max_speed_time}")
    
#lin_predkosc_samolotu()

def lin_odl():
    
    h = np.arange(0,timestamp2minutes(dane[-1,0]-to)+30,30)
    fig= plt.figure(figsize=(10, 5))
    ax=fig.add_subplot(1,1,1)
    ax.set_title("Odległość samolotu od lotniska w funkcji czasu")
    ax.set_xlabel("Czas (min)")
    ax.set_ylabel("Odległość (km)")
    ax.set_xticks(h)
    ax.set_yticks(np.arange(0, 7500, 500))
    for i in range(len(height)-1):
       if height[i]>0 or i<=32:
            ax.plot([timestamp2minutes(dane[i,0]-to),timestamp2minutes(dane[i+1,0]-to)],[odl[i],odl[i+1]], color='green',linestyle='-',linewidth=2)
       else:
           ax.plot([timestamp2minutes(dane[i,0]-to),timestamp2minutes(dane[i+1,0]-to)],[odl[i],odl[i+1]], color='red',linestyle='-',linewidth=2)
    plt.show()
#lin_odl()

def az2odl():
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    radial_ticks = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500,5000, 5500, 6000, 6500, 7000, 7500]
    radial_labels = ['500', '1000', '1500', '2000', '2500', '3000', '3500', '4000', '4500','5000', '5500', '6000', '6500', '7000', '7500']
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels(radial_labels)
    ax.set_rlim(500, 7500)
    for i in range(len(azymuty)-1):
       if height[i]<0:
           ax.plot([azymuty[i],azymuty[i+1]],[odl[i],odl[i+1]], color='red',linestyle='-',linewidth=3)
       else:
           ax.plot([azymuty[i],azymuty[i+1]],[odl[i],odl[i+1]], color='green',linestyle='-',linewidth=3)
    ax.set_rticks(np.arange(0, 7500, 500))
    ax.set_rlabel_position(0)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 24, endpoint=False))
    ax.set_rmax(7500)
    plt.show()



















