import numpy as np
from pylab import *
from mpl_toolkits.mplot3d.axes3d import Axes3D
import scipy.spatial, scipy.linalg

def get_data(type=1,n=1000,sigma=0.2):
    """
        Generates data by extedning simple 2D to 3D and
        roting them around.

        type := {1,2,3}. which dataset should be used
        n := How many samples should be generated
        sigma = Variance of the noise added to data.

        Returns - 3xn matrix containing the data
    """
    if(type==1):
        phi = np.random.uniform(2*np.pi,4*np.pi,size=n)
        phi = np.sort(phi)
        x = (phi)*np.cos(phi)
        y = (phi)*np.sin(phi)
        z = np.random.uniform(-4,4,size=phi.shape)
        data = np.vstack([x,y,z])
        color=phi

    if(type==2):
        phi = np.random.uniform(2*np.pi,4*np.pi,size=n)
        phi = np.sort(phi)
        x = (phi)*np.cos(phi)
        y = (phi)*np.sin(phi)
        z = np.random.uniform(-10,10,size=phi.shape)
        data = np.vstack([x,y,z])
        color=phi

    if(type==3):
        data=[]
        phi = []
        x=np.random.uniform(0,5*np.pi,size=n)
        x=np.sort(x)
        y=np.sin(x)
        z=np.random.uniform(-4,4,size=x.shape)
        data=np.vstack([x,y,z])
        color=x

    if(type==4):
        x = np.linspace(-2,2,num=n)
        y = np.abs(x+2)+np.abs(x-2)+np.abs(x-1)+np.abs(x**2-2)+np.abs(x+1)
        z = np.random.uniform(-2,2,size=n)
        data = np.vstack([x,y,z])
        color=x

    data = data+np.random.multivariate_normal([0,0,0],[
                [sigma,0,0],[0,sigma,0],[0,0,sigma]],n).T
    return np.dot(rotate3D(60,30,45),data).T, color

def rotate3D(x_rotation, y_rotation,z_rotation):
    # first normalize from degrees to radians
    x_rotation = np.pi*x_rotation/180.0
    y_rotation = np.pi*y_rotation/180.0
    z_rotation = np.pi*z_rotation/180.0

    Rx=np.array([[1,0,0],[0,np.cos(x_rotation),-np.sin(x_rotation)],
                [0,np.sin(x_rotation),np.cos(x_rotation)]])
    Ry=np.array([[np.cos(y_rotation),0,np.sin(y_rotation)],[0,1,0],
                 [-np.sin(y_rotation),0,np.cos(y_rotation)]])
    Rz=np.array([[np.cos(z_rotation),-np.sin(z_rotation),0],
                 [np.sin(z_rotation),np.cos(z_rotation),0],[0,0,1]])
    return np.dot(Rx,np.dot(Ry,Rz))

def plot_demo(n=5000, angle=0):
	for i in range(1,5):
		data,color = get_data(type=i,n=4000,sigma=0)
		fig = plt.figure(figsize=(10,5))
		ax = fig.add_subplot(111, projection='3d')
		ax.view_init(elev=10., azim=angle)
		ax.scatter(data[:,0],data[:,1],data[:,2],c=color)
		plt.show()
