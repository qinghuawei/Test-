
#https://blog.csdn.net/qsmx666/article/details/105561765

#%matplotlib inline
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import torch
import random
import pandas as pd
import sys



from numpy.random import randn
from numpy.random import randint


import os
import random
import numpy as np

from numpy import zeros
from numpy import ones,full
seed = 19931005+19941216
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现。

torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子


import numpy as np
import matplotlib.pyplot as plt

# 定义计算单个高斯分布概率密度的函数
def gaussian_probability_density(x, mean, variance):
    """
    计算一个单独的高斯分布的概率密度。

    Args:
    x: 浮点数，用于计算高斯分布概率密度的输入值。
    mean: 浮点数，高斯分布的均值。
    variance: 浮点数，高斯分布的方差。

    Returns:
    在输入x处的高斯分布的概率密度。
    """
    gaussian_pd = 1 / np.sqrt(2 * np.pi * variance) * np.exp(-(x - mean) ** 2 / (2 * variance))
    return gaussian_pd


# 定义计算两个高斯分布混合概率密度的函数
def mixed_gaussian_probability_density(x, weights, means, covariances):
    """
    计算两个高斯分布混合概率密度。

    Args:
    x: 浮点数，用于计算混合高斯分布概率密度的输入值。
    weights: 包含两个浮点数的列表，分别表示两个高斯分布在混合中的权重。
    means: 包含两个浮点数的列表，分别表示两个高斯分布的均值。
    covariances: 包含两个浮点数的列表，分别表示两个高斯分布的方差。

    Returns:
    包含第一个高斯分布的概率密度、第二个高斯分布的概率密度以及组成混合的概率密度的列表。
    """
    weight1 = weights[0]
    weight2 = weights[1]
    gaussian1_mean = means[0]
    gaussian2_mean = means[1]
    gaussian1_covariance = covariances[0]
    gaussian2_covariance = covariances[1]

    # 计算高斯分布的概率密度
    gaussian1_pd = gaussian_probability_density(x, gaussian1_mean, gaussian1_covariance)
    gaussian2_pd = gaussian_probability_density(x, gaussian2_mean, gaussian2_covariance)

    # 计算混合高斯分布的概率密度
    mixed_gaussian_pd = weight1 * gaussian1_pd + weight2 * gaussian2_pd

    return [weight1 * gaussian1_pd, weight2 * gaussian2_pd, mixed_gaussian_pd]


# 定义生成高斯分布采样点的函数
def gaussian_sampling(x_mean=10, variance=8, x_lower=0, x_upper=20):
    """
    生成一个高斯分布采样点集。

    Args:
    x_mean: 浮点数，表示两个高斯分布的均值。
    variance: 浮点数，表示两个高斯分布的方差。
    x_lower: 浮点数，采样点的下界。
    x_upper: 浮点数，采样点的上界。

    Returns:
    包含采样点集，用于生成样本的离散化x值列表和相应的y值列表，用于计算概率密度的x值范围，以及相应的概率密度。
    """
    np.random.seed(19931005)  # 设置随机数生成器的随机种子

    # 计算高斯分布混合的参数
    total_number_of_samples = 4000
    alpha_k1 = 0.5
    alpha_k2 = 0.5
    k1_mean = x_mean
    k2_mean = x_mean
    k1_covariance = variance
    k2_covariance = variance
    weights = [alpha_k1, alpha_k2]
    means = [k1_mean, k2_mean]
    covariances = [k1_covariance, k2_covariance]

    # 计算概率密度
    real_x = np.arange(x_lower, x_upper, 0.001)
    mixed_gaussian_pd = mixed_gaussian_probability_density(real_x, weights, means, covariances)[2]

    # 通过离散化计算高斯分布混合的采样点
    x_step_size = 0.5
    sampled_x_list = np.arange(x_lower, x_upper, x_step_size)
    sampled_y_list = mixed_gaussian_probability_density(sampled_x_list, weights, means, covariances)[2]

    # 按概率密度进行采样，并生成带有频率的对应的采样点列表
    sampled_y_list_int = np.round(sampled_y_list * total_number_of_samples, 0)
    sampled_x_list_with_frequency = []
    sampled_y_list_int_new = []
    for index in range(0, len(sampled_y_list_int), 1):
        if index % 2 == 0:
            sampled_y_list_int_new.append(0)
        else:
            sampled_y_list_int_new.append(sampled_y_list_int[index])
    sampled_y_list_int = sampled_y_list_int_new

    for index in np.arange(0, len(sampled_y_list_int), 1):
        for number in np.arange(0, sampled_y_list_int[index], 1):
            sampled_x_list_with_frequency.append(sampled_x_list[index])
    X = sampled_x_list_with_frequency

    return [X, sampled_x_list, sampled_y_list_int, real_x, mixed_gaussian_pd]


# 生成三组不同参数的高斯分布采样点
C1_results = gaussian_sampling(x_mean=10, variance=30, x_lower=-10, x_upper=50)
C1_X = C1_results[0]
C1_Sampled_x_list = C1_results[1]
C1_Sampled_y_list_int = C1_results[2]
C1_Real_x = C1_results[3]
C1_Mixed_gaussian_pd = C1_results[4]

C2_results = gaussian_sampling(x_mean=43, variance=15, x_lower=0, x_upper=70)
C2_X = C2_results[0]
C2_Sampled_x_list = C2_results[1]
C2_Sampled_y_list_int = C2_results[2]
C2_Real_x = C2_results[3]
C2_Mixed_gaussian_pd = C2_results[4]

C3_results = gaussian_sampling(x_mean=70, variance=8, x_lower=0, x_upper=85)
C3_X = C3_results[0]
C3_Sampled_x_list = C3_results[1]
C3_Sampled_y_list_int = C3_results[2]
C3_Real_x = C3_results[3]
C3_Mixed_gaussian_pd = C3_results[4]




import matplotlib.pyplot as plt
import numpy as np

# Define the size of the figure
fig, ax1 = plt.subplots(figsize=(14, 10))
Fontsize = 20

# Set the font size of the x and y tick labels
plt.xticks(fontsize=Fontsize)
plt.yticks(fontsize=Fontsize)

# Plot the sampled data points of C1, C2, and C3 on the first y-axis as histograms
x_step_size = 0.5
ax1.bar(C1_Sampled_x_list, C1_Sampled_y_list_int, width=x_step_size, color='red')
ax1.bar(C2_Sampled_x_list, C2_Sampled_y_list_int, width=x_step_size, color='blue')
ax1.bar(C3_Sampled_x_list, C3_Sampled_y_list_int, width=x_step_size, color='black')

# Set the y-axis limit for the first y-axis
ax1.set_ylim(0,)  # Total_number_of_Samples*0.15

# Plot the probability density of the mixture of Gaussians distribution for C1, C2, and C3 on the second y-axis
ax2 = ax1.twinx()
ax2.scatter(C1_Real_x, C1_Mixed_gaussian_pd, color='black', s=0.3)
ax2.scatter(C2_Real_x, C2_Mixed_gaussian_pd, color='black', s=0.3)
ax2.scatter(C3_Real_x, C3_Mixed_gaussian_pd, color='black', s=0.3)

# Set the y-axis limit for the second y-axis
ax2.set_ylim(0,)  # 0.15

# Set the labels for both y-axes
ax1.set_ylabel('Frequency', color='black', fontsize=Fontsize+10)
ax2.set_ylabel('Probability Density', color='black', fontsize=Fontsize+10)

# Set the border linewidth to be 2 for all edges of the plot
border_width = 2
ax = plt.gca()
ax.spines['bottom'].set_linewidth(border_width)
ax.spines['left'].set_linewidth(border_width)
ax.spines['top'].set_linewidth(border_width)
ax.spines['right'].set_linewidth(border_width)

# Save the figure as an eps file
plt.savefig('Example 1-Figure 01, Sampling, Real_x and Gaussian_pd.eps', dpi=600, format='eps')

# Define a list of arrays that contain the X values for C1, C2, and C3
X_list = [
    np.expand_dims(np.array(C1_X), axis=1),
    np.expand_dims(np.array(C2_X), axis=1),
    np.expand_dims(np.array(C3_X), axis=1)]






class net_G_1(nn.Module):
    def __init__(self):
        super(net_G_1, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 5),
            nn.Linear(5, 5),
            nn.Linear(5, 1),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.model(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
                m.bias.data.zero_()

class net_D_1(nn.Module):
    def __init__(self):
        super(net_D_1, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 5),
            nn.Tanh(),
            nn.Linear(5, 5),
            nn.Tanh(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.model(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
                m.bias.data.zero_()




class net_G_2(nn.Module):
    def __init__(self):
        super(net_G_2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 5),
            nn.Linear(5, 5),
            nn.Linear(5, 1),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.model(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
                m.bias.data.zero_()

class net_D_2(nn.Module):
    def __init__(self):
        super(net_D_2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 5),
            nn.Tanh(),
            nn.Linear(5, 5),
            nn.Tanh(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.model(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
                m.bias.data.zero_()





class net_G_3(nn.Module):
    def __init__(self):
        super(net_G_3, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 5),
            nn.Linear(5, 5),
            nn.Linear(5, 1),


        )
        self._initialize_weights()

    def forward(self, x):
        x = self.model(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
                m.bias.data.zero_()

class net_D_3(nn.Module):
    def __init__(self):
        super(net_D_3, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 5),
            nn.Tanh(),
            nn.Linear(5, 5),
            nn.Tanh(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.model(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
                m.bias.data.zero_()





def update_D(X, Z, net_D, net_G, loss, trainer_D):
    batch_size = X.shape[0]
    ones = torch.ones(batch_size).view(batch_size, 1)
    zeros = torch.zeros(batch_size).view(batch_size, 1)
    real_Y = net_D(X)
    fake_X = net_G(Z)
    fake_Y = net_D(fake_X)
    loss_D = (loss(real_Y, ones)+loss(fake_Y, zeros))/2
    loss_D.backward()
    trainer_D.step()
    return loss_D.sum()

def update_G(Z, net_D, net_G, loss, trainer_G):
    batch_size = Z.shape[0]
    ones = torch.ones((batch_size,)).view(batch_size, 1)
    zeros = torch.zeros(batch_size).view(batch_size, 1)
    fake_X = net_G(Z)
    fake_Y = net_D(fake_X)
    loss_G = loss(fake_Y, ones)
    #loss_G = -loss(fake_Y, zeros)
    loss_G.backward()
    trainer_G.step()
    return loss_G.sum()



def train(D1,D2,D3,G1,G2,G3, X_list,lr_D1,lr_D2, lr_D3,  lr_G1, lr_G2, lr_G3, num_epochs,latent_dim):


    # D1 = D_list[0]
    # D2 = D_list[1]
    # D3 = D_list[2]
    #
    # G1 = G_list[0]
    # G2 = G_list[1]
    # G3 = G_list[2]



    #C111111111111111
    loss = nn.BCELoss()  # 二分类
    trainer_D1 = torch.optim.Adam(D1.parameters(), lr=lr_D1)
    trainer_G1 = torch.optim.Adam(G1.parameters(), lr=lr_G1)

    d_loss_point_C1 = []
    g_loss_point_C1 = []
    d_loss = 0
    g_loss = 0

    np.random.seed(1)
    X_1 = torch.tensor(X_list[0], dtype=torch.float32)
    Z_1 = torch.tensor(np.random.normal(
        0, 1, (X_1.shape[0], latent_dim)), dtype=torch.float32)

    for epoch in range(1, num_epochs+1):
        d_loss_sum = 0
        g_loss_sum = 0

        trainer_D1.zero_grad()

        # real_label=torch.tensor(real_label, dtype=torch.float32)
        # fake_label = torch.tensor(fake_label, dtype=torch.float32)

        d_loss = update_D(X_1, Z_1, D1, G1, loss, trainer_D1)
        d_loss_sum += d_loss

        trainer_G1.zero_grad()
        g_loss = update_G(Z_1,  D1, G1, loss, trainer_G1)
        g_loss_sum += g_loss


        if epoch%100==0:
            print(epoch, 'd_loss:', d_loss_sum, 'g_loss:', g_loss_sum)

        d_loss_point_C1.append(d_loss_sum)
        g_loss_point_C1.append(g_loss_sum)

    X_fake_C1 = G1(Z_1).detach().numpy()


    plt.figure(figsize=(7, 4))
    plt.ylabel('Loss', fontdict={'size': 15})
    plt.xlabel('Epoch', fontdict={'size': 15})
    #plt.xticks(range(0, num_epochs+1, 3))

    d_loss_point_C1=[d_loss_point_C1[i].cpu().detach().numpy().tolist() for i in np.arange(0,len(d_loss_point_C1))]
    g_loss_point_C1 = [g_loss_point_C1[i].cpu().detach().numpy().tolist() for i in np.arange(0, len(g_loss_point_C1))]


    # print('[-x for x in d_loss_point]:',[-x for x in d_loss_point])
    # print('[-x for x in g_loss_point]:', [-x for x in g_loss_point])
    plt.plot(range(1, num_epochs+1), [-x for x in g_loss_point_C1],
             color='orange', label='Discriminator_1')
    plt.plot(range(1, num_epochs+1), [-x for x in d_loss_point_C1],
             color='blue', label='Generator_1')
    plt.legend()
    plt.savefig('loss curve_C1.eps', dpi=600, format='eps')
    #plt.show()
    #print(d_loss, g_loss)











    #C22222222222222
    loss = nn.BCELoss()  # 二分类
    trainer_D2 = torch.optim.Adam(D2.parameters(), lr=lr_D2)
    trainer_G2 = torch.optim.Adam(G2.parameters(), lr=lr_G2)

    d_loss_point_C2 = []
    g_loss_point_C2 = []
    d_loss = 0
    g_loss = 0

    np.random.seed(1)
    X_2 = torch.tensor(X_list[1], dtype=torch.float32)
    Z_2 = torch.tensor(np.random.normal(
        0, 1, (X_2.shape[0], latent_dim)), dtype=torch.float32)

    for epoch in range(1, num_epochs+1):
        d_loss_sum = 0
        g_loss_sum = 0
        trainer_D2.zero_grad()

        # real_label=torch.tensor(real_label, dtype=torch.float32)
        # fake_label = torch.tensor(fake_label, dtype=torch.float32)

        d_loss = update_D(X_2, Z_2, D2, G2, loss, trainer_D2)
        d_loss_sum += d_loss

        trainer_G2.zero_grad()
        g_loss = update_G(Z_2,  D2, G2, loss, trainer_G2)
        g_loss_sum += g_loss


        if epoch%100==0:
            print(epoch, 'd_loss:', d_loss_sum, 'g_loss:', g_loss_sum)

        d_loss_point_C2.append(d_loss_sum)
        g_loss_point_C2.append(g_loss_sum)

    X_fake_C2 = G2(Z_2).detach().numpy()


    plt.figure(figsize=(7, 4))
    plt.ylabel('Loss', fontdict={'size': 15})
    plt.xlabel('Epoch', fontdict={'size': 15})
    #plt.xticks(range(0, num_epochs+1, 3))

    d_loss_point_C2=[d_loss_point_C2[i].cpu().detach().numpy().tolist() for i in np.arange(0,len(d_loss_point_C2))]
    g_loss_point_C2 = [g_loss_point_C2[i].cpu().detach().numpy().tolist() for i in np.arange(0, len(g_loss_point_C2))]


    # print('[-x for x in d_loss_point]:',[-x for x in d_loss_point])
    # print('[-x for x in g_loss_point]:', [-x for x in g_loss_point])
    plt.plot(range(1, num_epochs+1), [-x for x in g_loss_point_C2],
             color='orange', label='Discriminator_2')
    plt.plot(range(1, num_epochs+1), [-x for x in d_loss_point_C2],
             color='blue', label='Generator_2')
    plt.legend()
    plt.savefig('loss curve_C2.eps', dpi=600, format='eps')
    #plt.show()
    #print(d_loss, g_loss)




    #C33333333
    loss_3 = nn.BCELoss()  # 二分类
    trainer_D3 = torch.optim.Adam(D3.parameters(), lr=lr_D3)
    trainer_G3 = torch.optim.Adam(G3.parameters(), lr=lr_G3)

    d_loss_point_C3 = []
    g_loss_point_C3 = []
    d_loss = 0
    g_loss = 0

    #np.random.seed(1)
    X_3 = torch.tensor(X_list[2], dtype=torch.float32)
    Z_3 = torch.tensor(np.random.normal(
        0, 1, (X_3.shape[0], latent_dim)), dtype=torch.float32)


    for epoch in range(1, num_epochs+1):
        d_loss_sum = 0
        g_loss_sum = 0


        trainer_D3.zero_grad()

        # real_label=torch.tensor(real_label, dtype=torch.float32)
        # fake_label = torch.tensor(fake_label, dtype=torch.float32)

        d_loss = update_D(X_3, Z_3, D3, G3, loss_3, trainer_D3)
        d_loss_sum += d_loss

        trainer_G3.zero_grad()
        g_loss = update_G(Z_3,  D3, G3, loss_3, trainer_G3)
        g_loss_sum += g_loss


        if epoch%100==0:
            print(epoch, 'd_loss:', d_loss_sum, 'g_loss:', g_loss_sum)

        d_loss_point_C3.append(d_loss_sum)
        g_loss_point_C3.append(g_loss_sum)

    X_fake_C3 = G3(Z_3).detach().numpy()

    # np.random.seed(1)
    # X_fake_C3 = torch.tensor(np.random.normal(
    #     70, 8**0.5, (4000, 1)), dtype=torch.float32)

    plt.figure(figsize=(7, 4))
    plt.ylabel('Loss', fontdict={'size': 15})
    plt.xlabel('Epoch', fontdict={'size': 15})
    #plt.xticks(range(0, num_epochs+1, 3))

    d_loss_point_C3=[d_loss_point_C3[i].cpu().detach().numpy().tolist() for i in np.arange(0,len(d_loss_point_C3))]
    g_loss_point_C3 = [g_loss_point_C3[i].cpu().detach().numpy().tolist() for i in np.arange(0, len(g_loss_point_C3))]


    # print('[-x for x in d_loss_point]:',[-x for x in d_loss_point])
    # print('[-x for x in g_loss_point]:', [-x for x in g_loss_point])
    plt.plot(range(1, num_epochs+1), [-x for x in g_loss_point_C3],
             color='orange', label='Discriminator_3')
    plt.plot(range(1, num_epochs+1), [-x for x in d_loss_point_C3],
             color='blue', label='Generator_3')
    plt.legend()
    plt.savefig('loss curve_C3.eps', dpi=600, format='eps')
    #plt.show()
    #print(d_loss, g_loss)




    # Loss curve C1+C2+C3
    plt.figure(figsize=(6, 4))
    plt.ylabel('Loss', fontdict={'size': 15})
    plt.xlabel('Epoch', fontdict={'size': 15})
    # plt.xticks(range(0, num_epochs+1, 3))

    # d_loss_point_C3 = [d_loss_point_C3[i].cpu().detach().numpy().tolist() for i in np.arange(0, len(d_loss_point_C3))]
    # g_loss_point_C3 = [g_loss_point_C3[i].cpu().detach().numpy().tolist() for i in np.arange(0, len(g_loss_point_C3))]

    d_loss_point_average=[(a+b+c)/3 for a,b,c in zip(d_loss_point_C1,d_loss_point_C2,d_loss_point_C3)]
    g_loss_point_average = [(a + b + c) / 3 for a, b, c in zip(g_loss_point_C1, g_loss_point_C2, g_loss_point_C3)]

    # print('[-x for x in d_loss_point]:',[-x for x in d_loss_point])
    # print('[-x for x in g_loss_point]:', [-x for x in g_loss_point])
    plt.plot(range(1, num_epochs + 1), [-x for x in g_loss_point_average],
             color='orange', label='Average loss of discriminator ')
    plt.plot(range(1, num_epochs + 1), [-x for x in d_loss_point_average],
             color='blue', label='Average loss of generator')
    plt.legend()
    plt.savefig('loss curve average.eps', dpi=600, format='eps')
    # plt.show()
    # print(d_loss, g_loss)





    import seaborn as sns
    fig, ax1 = plt.subplots(figsize=(14, 10))

    Fontsize = 25
    plt.xticks(fontsize=Fontsize)  # 设置x轴刻度字体大小
    plt.yticks(fontsize=Fontsize)  # 设置x轴刻度字体大小
    ax2 = ax1.twinx()
    # ax1.bar(Sampled_x_list, Sampled_y_list_int, width=x_step_size, label='Real samples', color='blue',
    #         edgecolor='orchid')
    # ax1.set_ylim(0, Total_number_of_Samples * 0.3)

    # fig, ax1 = plt.subplots(figsize=(14, 10))

    # ax2 = ax1.twinx()

    x_step_size = 0.5
    ax1.bar(C1_Sampled_x_list, C1_Sampled_y_list_int, width=x_step_size,  color='red')
    ax1.bar(C2_Sampled_x_list, C2_Sampled_y_list_int, width=x_step_size, color='blue')
    ax1.bar(C3_Sampled_x_list, C3_Sampled_y_list_int, width=x_step_size, color='black')#, edgecolor='orchid'

    ax1.set_ylim(0, 12000*0.05)  # Total_number_of_Samples * 0.3

    bins = np.arange(-20 - 0.5 * x_step_size, 95 - 0.5 * x_step_size, x_step_size)
    #bins = np.arange(-0 - 0.5 * x_step_size, 20 - 0.5 * x_step_size, x_step_size)
    # sns.distplot(X_fake[:, 0], bins, ax=ax2, kde=False, label='Generated samples', color='red')
    sns.distplot(X_fake_C1, bins, ax=ax2, kde=False, label='Generated samples',color='darkorange' )
    sns.distplot(X_fake_C2, bins, ax=ax2, kde=False, label='Generated samples', color='darkgreen')
    sns.distplot(X_fake_C3, bins, ax=ax2, kde=False, label='Generated samples', color='grey')

    # X_fake_C3 = torch.tensor(np.random.normal(
    #     70, 8, (4000, 1)), dtype=torch.float32)

    ax2.set_ylim(0, 12000*0.05 * 0.5)  # Total_number_of_Samples * 0.3 * 0.55

    for tl in ax1.get_yticklabels():
        # tl.set_fontsize(15)
        tl.set_color('blue')

    for tl in ax2.get_yticklabels():
        # tl.set_fontsize(15)
        tl.set_color('red')

    # 设置坐标轴的标签
    ax1.set_ylabel('Frequency of real samples', color='blue', fontsize=Fontsize + 5)
    ax2.set_ylabel('Frequency of generated samples', color='red', fontsize=Fontsize + 5)

    # plt.xlim(-10, 20)
    # plt.legend()

    bwith = 2
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)

    # plt.xticks(fontsize=Fontsize)  # 设置x轴刻度字体大小
    plt.yticks(fontsize=Fontsize)  # 设置x轴刻度字体大小

    plt.savefig('generate distribution.eps', dpi=600, format='eps')

    plt.show()


    print("np.mean(X_fake_C1),np.var(X_fake_C1):",np.mean(X_fake_C1),np.var(X_fake_C1))
    print("np.mean(X_fake_C2),np.var(X_fake_C2):", np.mean(X_fake_C2), np.var(X_fake_C2))
    print("np.mean(X_fake_C3),np.var(X_fake_C3):", np.mean(X_fake_C3), np.var(X_fake_C3))

    sys.exit()





if __name__ == '__main__':



    lr_D1, lr_G1,  = 0.01, 0.001
    lr_D2, lr_G2, = 0.02, 0.00105
    lr_D3, lr_G3, = 0.011, 0.00502



    latent_dim, num_epochs= 1, 1500


    # generator_list,discriminator_list=[],[]
    # for i in range(1,number_of_class+1):
    #     locals()[f'generator_C{i}'] = net_G()
    #     locals()[f'discriminator_C{i}'] = net_D()
    #
    #     generator_list.append('generator_C' + str(i))
    #     discriminator_list.append('discriminator_C' + str(i))

    # print('generator,discriminator:',generator,discriminator)
    # sys.exit()
    #

    # generator_list=[net_G(),net_G(),net_G()]
    # discriminator_list = [net_D(), net_D(), net_D()]
    #

    D1 = net_D_1()
    D2 = net_D_2()
    D3 = net_D_3()

    G1 = net_G_1()
    G2 = net_G_2()
    G3 = net_G_3()



    X_list=X_list



    train(D1,D2,D3,G1,G2,G3,X_list,lr_D1,lr_D2, lr_D3,  lr_G1, lr_G2, lr_G3, num_epochs,latent_dim)

