from __future__ import print_function, division
import IsingGrid
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch # pytorch package, allows using GPUs
import pickle as pickle
import os
# fix seed
seed=17
np.random.seed(seed)
torch.manual_seed(seed)

# Fundamental parameters

from torchvision import datasets # load data
from torch import from_numpy

def Monte_Calo_Ising(size, temperature, steps, interval, Jfactor,flag):
    g = IsingGrid.Grid(size, Jfactor,flag)
    g.randomize()

    # Animation parameters
    X_data=np.zeros((steps//interval,size**2))
    if(temperature<2.269):
        Y_data=np.zeros(steps//interval)
    else:
        Y_data=np.ones(steps//interval)
    # Simulation

    print("Simulation begins.Temperature is %f"%temperature)

    #退火
    for _ in range(5000):
        clusterSize = g.clusterFlip(temperature)

    for step in range(steps):
        clusterSize = g.clusterFlip(temperature)

        if (step + 1) % interval == 0:
            X_data[step//interval,:]=g.canvas.reshape(-1,size**2)

        if (step + 1) % (200 * interval) == 0:
            print("Step ", step + 1, "/", steps, ", Cluster size ", clusterSize, "/", size * size)

    print("Simulation completes.")

    return X_data,Y_data

class Create_Dataset(torch.utils.data.Dataset):#要用这个库划分数据集
    """Ising pytorch dataset."""

    def __init__(self,transform=False,size=100):
        """
        Args:
            data_type (string): `train`, `test` or `critical`: creates data_loader
            transform (callable, optional): Optional transform to be applied on a sample.

        """

        from sklearn.model_selection import train_test_split
        import collections
        import pickle as pickle
        T=np.linspace(1.2,3.3,36) # temperatures
        sample_test=1000
        interval = 1
        test_ratio=0.2
        Jfactor = 1

        for t in T:
            X_temp,Y_temp=Monte_Calo_Ising(size, t, round(sample_test*interval/test_ratio), interval, Jfactor,flag=0)
            X_temp=X_temp.astype(np.int32)
            Y_temp=Y_temp.astype(np.int32)
            with open('Ising2DFM_train_X_{:.2f}.pkl'.format(t),'wb') as f:
                pickle.dump(X_temp,f)
            with open('Ising2DFM_train_Y_{:.2f}.pkl'.format(t),'wb') as f:
                pickle.dump(Y_temp,f)


def Create_triangle(size):
    #create triangle Ising model
    import collections
    import pickle as pickle
    T=np.linspace(2.5,4.5,41) # temperatures
    sample_test=1000
    interval = 1
    test_ratio=1
    Jfactor = 1

    for t in T:
        X_temp,Y_temp=Monte_Calo_Ising(size, t, round(sample_test*interval/test_ratio), interval, Jfactor,flag=1)
        X_temp=X_temp.astype(np.int32)
        Y_temp=Y_temp.astype(np.int32)
        with open('Ising2DFM_train_X_triangle{:.2f}.pkl'.format(t),'wb') as f:
            pickle.dump(X_temp,f)
        with open('Ising2DFM_train_Y_triangle{:.2f}.pkl'.format(t),'wb') as f:
            pickle.dump(Y_temp,f)


if __name__ == "__main__":
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)

    os.chdir(current_dir)

    # 获取修改后的当前工作目录
    updated_dir = os.getcwd()
    print("Updated working directory:", updated_dir)



    Create_Dataset(transform=False,size=40)
    Create_triangle(size=40)
    # Animation parameters
    size = 100
    temperature = 5
    steps = 4000
    interval = 10
    Jfactor = 1
    fig, ax = plt.subplots()
    X_data,Y_data=Monte_Calo_Ising(size, temperature, steps, interval, Jfactor,flag=1)
    # Animation

    print("Animation begins.")

    for frame in range(0, len(X_data)):
        ax.cla()
        ax.imshow(X_data[frame].reshape(size,size), cmap=mpl.cm.winter)
        ax.set_title("Step {}".format(frame * interval))
        #plt.show()
        plt.pause(0.01)
    plt.show()
    print("Animation completes.")
