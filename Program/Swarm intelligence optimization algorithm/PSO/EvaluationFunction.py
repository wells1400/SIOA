import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class BaseEvaluationFunction(object):
    def __init__(self, x = np.array([0,0])):
        self.x = x
        
    def rastrigin(self, x=None):
        '''       
        :return: GMM=0 at (0,0...0) Recommend hypercube xi ∈ [-5.12, -5.12], for all i = 1, …, d. 
        '''
        if x is None:
            return 10*self.x.shape[0] + np.sum(self.x**2) - 10*np.sum(np.cos(2*np.pi*self.x))
        else:
            return 10 * x.shape[0] + np.sum(x ** 2) - 10 * np.sum(np.cos(2 * np.pi * x))

    def bukin(self, x=None):
        '''
        :return:GMM=0 at (-10,1) if x1 ∈ [-15, -5], x2 ∈ [-3, 3]
        '''
        if x is None:
            x1 = self.x[0]
            x2 = self.x[1]
            return 100*np.sqrt(np.abs(x2 - 0.01*x1**2)) + 0.01*np.abs(x1 + 10)
        else:
            return 100*np.sqrt(np.abs(x[1] - 0.01*x[0]**2)) + 0.01*np.abs(x[0] + 10)

    def ackley(self,x=None, a=20, b=0.2, c=2*np.pi):
        '''
        :param a: recommended a=20 
        :param b: recommended b=0.2 
        :param c: recommended c=2*pi 
        :return: GMM=0 at (0,0...0)
        '''
        if x is None:
            d = self.x.shape[0]
            t1 = -a*np.exp(-b*np.sqrt((np.sum(self.x**2))/d))
            t2 = -np.exp(np.sum(np.cos(c*self.x))/d)
            return t1 + t2 + + a + np.exp(1)
        d = x.shape[0]
        return -a*np.exp(-b*np.sqrt((np.sum(x**2))/d)) - np.exp(np.sum(np.cos(c*x))/d) + a + np.exp(1)

    def sphere(self, x=None):
        '''
        :return:GMM=0 at (0,0...0) Recommend hypercube xi ∈ [-5.12, 5.12], for all i = 1, …, d.
        '''
        if x is None:
            return np.sum(self.x**2)
        return np.sum(x**2)

    def beale(self, x=None):
        '''
        :return: GMM=0 at (3,0.5) Recommend square xi ∈ [-4.5, 4.5], for all i = 1, 2. 
        '''
        if x is None:
            x1 = self.x[0]
            x2 = self.x[1]
            return (1.5-x1+x1*x2)**2 + (2.25-x1+x1*(x2**2))**2 + (2.625-x1 + x1*(x2**3))**2
        return (1.5-x[0]+x[0]*x[1])**2 + (2.25-x[0]+x[0]*(x[1]**2))**2 + (2.625-x[0] + x[0]*(x[1]**3))**2
        
    def booth(self, x=None):
        '''
        :return: GMM=0 at (1,3) Recommend square xi ∈ [-10, 10], for all i = 1, 2.
        '''
        if x is None:
            x1 = self.x[0]
            x2 = self.x[1]
            return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2
        return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2
    
    def threehumpcamel(self,  x=None):
        '''
        :return: GMM=0 at (0,0) Recommend square xi ∈ [-5, 5], for all i = 1, 2. 
        '''
        if x is None:
            x1 = self.x[0]
            x2 = self.x[1]
            return 2*x1**2 - 1.05*x1**4 + x1*x2 + x2**2
        return 2*x[0]**2 - 1.05*x[0]**4 + x[0]*x[1] + x[1]**2
    
    def holdertable(self, x=None):
        '''
        :return: GMM=-19.2085 at (+- 8.05502, +-9.66459) Recommend square xi ∈ [-10, 10], for all i = 1, 2. 
        '''
        if x is None:
            x1 = self.x[0]
            x2 = self.x[1]
            return -np.abs(np.sin(x1)*np.cos(x2)*np.exp(np.abs(1-(np.sqrt(x1**2 + x2**2))/np.pi)))
        return -np.abs(np.sin(x[0])*np.cos(x[1])*np.exp(np.abs(1-(np.sqrt(x[0]**2 + x[1]**2))/np.pi)))


class PlotEvaluation3D:
    def __init__(self, xinterval_min=-5, xinterval_max=5, savepath =None, meshgrid=0.1):
        self.x1_min = xinterval_min
        self.x1_max = xinterval_max
        self.x2_min = xinterval_min
        self.x2_max = xinterval_max
    
        self.savepath = savepath
        self.meshgrid = meshgrid
        
        self.meshres = self.prodmeshgrid()
        self.X = self.meshres[0]
        self.Y = self.meshres[1]
        self.Z = np.zeros(self.X.shape)
        
    def prodmeshgrid(self):
        X = np.arange(self.x1_min, self.x1_max, self.meshgrid)
        Y = np.arange(self.x2_min, self.x2_max, self.meshgrid)
        X, Y = np.meshgrid(X, Y)
        return X, Y
        
    def drawfunc3D(self, title='Unname', z_min=0, z_max=100):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(self.X, self.Y, self.Z, cmap=plt.cm.hot)
        ax.set_zlim(z_min, z_max)
        ax.set_title(title)
        if self.savepath is not None:
            plt.savefig(self.savepath + title + '.png', bbox_inches='tight') # 保存图片
        plt.show()
    
    #  rastrigin测试函数
    def plot_rastrigin(self):
        '''
        :return: GMM=0 at (0,0...0) Recommend hypercube xi ∈ [-5.12, 5.12], for all i = 1, …, d.
        '''
        A = 10
        self.Z = 2 * A + self.X ** 2 - A * np.cos(2 * np.pi * self.X) + self.Y ** 2 - A * np.cos(2 * np.pi * self.Y)
        self.drawfunc3D(title='Rastrigin function', z_min=0, z_max=100)
        
    def plot_ackley(self):
        '''
        :return: GMM=0 at (0,0...0)
        '''
        self.Z = -20 * np.exp(-0.2 * np.sqrt(0.5 * (self.X**2 + self.Y**2))) - \
                 np.exp(0.5 * (np.cos(2 * np.pi * self.X) + np.cos(2 * np.pi * self.Y))) + np.e + 20
        #self.Z = - self.Z
        self.drawfunc3D(title='Ackley function', z_min=None, z_max=0)
        
    def plot_sphere(self):
        '''
        :return: GMM=0 at (0,0...0) Recommend hypercube xi ∈ [-5.12, 5.12], for all i = 1, …, d.
        '''
        self.Z = self.X**2 + self.Y**2
        self.drawfunc3D(title='Sphere function', z_min=None, z_max=20)
    
    def plot_beale(self):
        '''
        :return: GMM=0 at (3,0.5) Recommend square xi ∈ [-4.5, 4.5], for all i = 1, 2.
        '''
        self.Z = np.power(1.5 - self.X + self.X * self.Y, 2) + np.power(2.25 - self.X + self.X * (self.Y ** 2), 2)         + np.power(2.625 - self.X + self.X * (self.Y ** 3), 2)
        self.drawfunc3D(title='Beale function', z_min=None, z_max=150000)
    
    def plot_booth(self):
        '''
        :return: GMM=0 at (1,3) Recommend square xi ∈ [-10, 10], for all i = 1, 2.
        '''
        self.Z = np.power(self.X + 2*self.Y - 7, 2) + np.power(2 * self.X + self.Y - 5, 2)
        self.drawfunc3D(title='Booth function', z_min=None, z_max=2500)

    def plot_bukin(self):
        '''
        :return: GMM=0 at (-10,1) if x1 ∈ [-15, -5], x2 ∈ [-3, 3]
        '''
        self.Z = 100 * np.sqrt(np.abs(self.Y - 0.01 * self.X**2)) + 0.01 * np.abs(self.X + 10)
        self.drawfunc3D(title='Bukin function', z_min=None, z_max=200)
    
    def plot_three_humpCamel(self):
        '''
        :return: GMM=0 at (0,0) Recommend square xi ∈ [-5, 5], for all i = 1, 2.
        '''
        self.Z = 2 * self.X**2 - 1.05 * self.X**4 + (1/6) * self.X**6 + self.X*self.Y + self.Y*2
        self.drawfunc3D(title='three_humpCamel function', z_min=None, z_max=2000)
        pass
    
    def plot_holdertable(self):
        '''
        :return: GMM=-19.2085 at (+- 8.05502, +-9.66459) Recommend square xi ∈ [-10, 10], for all i = 1, 2.
        '''
        self.Z = -np.abs(np.sin(self.X) * np.cos(self.Y) * np.exp(np.abs(1 - np.sqrt(self.X**2 + self.Y**2)/np.pi)))
        self.drawfunc3D(title='Holder table function', z_min=-20, z_max=0)

