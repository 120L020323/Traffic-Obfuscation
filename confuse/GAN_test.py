import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# GAN混淆
# 读取数据文件到 numpy 数组。
data_path = "E:/tyy_confuse/Pcaps/Browsing_test/Browsing_add/BROWSING_gate_SSL_Browsing_training_add.txt"
data = np.loadtxt(data_path)

# 数据预处理
X_train = data[:, :2]  # 选择前两列作为输入特征
y_train = np.ones((len(data), 1))  # 将标签设定为全为1的数组

# 将 numpy 数组转换为 PyTorch 张量以在神经网络使用
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)


# 定义生成器模型,接收低维噪声向量，并输出与训练数据维度相同的向量
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)


# 定义判别器模型,接收向量，并输出表示该向量是真实数据的概率的标量。
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# 初始化生成器、判别器和优化器
input_dim = 2  # 设置输入特征的维度为 2
output_dim = 2
generator = Generator(input_dim, output_dim)
discriminator = Discriminator(output_dim)
optimizer_G = optim.Adam(generator.parameters(), lr=0.0004, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0004, betas=(0.5, 0.999))


# 训练GAN模型
def train_gan(generator, discriminator, X_train_tensor, epochs=10000, batch_size=128):
    for epoch in range(epochs):
        idx = np.random.randint(0, X_train_tensor.shape[0], batch_size)
        real_X = X_train_tensor[idx]
        real_y = torch.ones(batch_size, 1)
        noise = torch.rand(batch_size, input_dim) * 150  # 生成随机噪声,通过生成器生成假数据
        fake_X = generator(noise)

        # 训练判别器以区分真实和假数据
        optimizer_D.zero_grad()
        loss_real = nn.BCELoss()(discriminator(real_X), real_y)
        loss_fake = nn.BCELoss()(discriminator(fake_X.detach()), torch.zeros(batch_size, 1))
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # 训练生成器以欺骗判别器
        optimizer_G.zero_grad()
        loss_G = nn.BCELoss()(discriminator(fake_X), torch.ones(batch_size, 1))
        loss_G.backward()
        optimizer_G.step()

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}], Discriminator Loss: {loss_D.item()}, Generator Loss: {loss_G.item()}")


# 训练GAN模型
train_gan(generator, discriminator, X_train_tensor)

# 生成混淆文件
num_samples = 10000
noise = torch.randint(0, 1501, (num_samples, input_dim), dtype=torch.float32)  # 修改这里，生成与输入特征维度相同的随机噪声
generated_data = generator(noise).detach().numpy()

# 保存混淆文件
save_path = "E:/tyy_confuse/Pcaps/Browsing_test/Browsing_GAN/generated_data.txt"
np.savetxt(save_path, generated_data, fmt='%.6f')

# 计算生成数据的概率,概率接近 1则意味着生成的数据与真实数据非常相似
probabilities = discriminator(torch.tensor(generated_data, dtype=torch.float32)).detach().numpy()
average_probability = np.mean(probabilities)
print("Average probability of generated data:", average_probability)
