import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from load_data import image_size, load_training_data


hidden_layer_size = 500
print_interval = 1
image_size = image_size[0] * image_size[1]
d_learning_rate = 1e-3
g_learning_rate = 1e-3
sgd_momentum = 0.9
num_epochs = 200
d_steps = 20
g_steps = 20


class Generator(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, f):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, x):
        x = self.map1(x)
        x = self.f(x)
        x = self.map2(x)
        x = self.f(x)
        x = self.map3(x)
        return x


class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, f):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, x):
        x = self.map1(x)
        x = self.f(x)
        x = self.map2(x)
        x = self.f(x)
        x = self.map3(x)
        return self.f(x)


def generator_sampler(n_sample):
    return torch.rand(n_sample).reshape(-1, 1)


def extract(v):
    return v.data.storage().tolist()


real_images = torch.from_numpy(load_training_data()).float()
minibatch_size = int(len(real_images) / d_steps)
G = Generator(1, hidden_layer_size, image_size, torch.sigmoid)
D = Discriminator(
    image_size * minibatch_size, hidden_layer_size, 1, torch.sigmoid)
criterion = nn.BCELoss()
d_optimizer = optim.SGD(
    D.parameters(), lr=d_learning_rate, momentum=sgd_momentum)
g_optimizer = optim.SGD(
    G.parameters(), lr=g_learning_rate, momentum=sgd_momentum)

for epoch in range(num_epochs):
    for d_real_data in DataLoader(
            real_images,
            shuffle=True,
            num_workers=4,
            batch_size=minibatch_size):
        D.zero_grad()
        #  1A: Train D on real
        d_real_decision = D(d_real_data.view(-1))
        # ones = real
        d_real_error = criterion(
            d_real_decision, Variable(torch.ones([1, 1])))
        d_real_error.backward()

        #  1B: Train D on fake
        d_gen_input = generator_sampler(minibatch_size)
        # detach to avoid training G on these labels
        d_fake_data = G(d_gen_input).detach().view(-1)
        d_fake_decision = D(d_fake_data)
        # zeros = fake
        d_fake_error = criterion(
            d_fake_decision, Variable(torch.zeros([1, 1])))
        d_fake_error.backward()
        # Only optimizes D's parameters; changes based on stored gradients
        # from backward()
        d_optimizer.step()

        dre, dfe = extract(d_real_error)[0], extract(d_fake_error)[0]

    for g_index in range(g_steps):
        # 2. Train G on D's response (but DO NOT train D on these labels)
        G.zero_grad()

        gen_input = Variable(generator_sampler(minibatch_size)).view(-1)
        g_fake_data = G(gen_input)
        dg_fake_decision = D(g_fake_data)
        # Train G to pretend it's genuine
        g_error = criterion(
            dg_fake_decision, Variable(torch.ones([1, 1])))

        g_error.backward()
        g_optimizer.step()  # Only optimizes G's parameters
        ge = extract(g_error)[0]

    if epoch % print_interval == 0:
            print('\n\n')
            print(f'Epoch {epoch}:')
            print(f'\tDiscriminator Error on Real Data: {dre}')
            print(f'\tDiscriminator Error on Fake Data: {dfe}')
            print(f'\tGenerator Error {ge}')

torch.save(D, 'discriminator.pkl')
torch.save(G, 'generator.pkl')
