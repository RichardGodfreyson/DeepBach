import pickle
from itertools import islice

import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import BACH_DATASET, indexed_chorale_to_score
from minibach import generator_pianoroll, reconstruct
from torch.autograd import Variable
from tqdm import tqdm


def sample_gumbel(input):
    # CDF: exp(-exp(-(x-mu) / beta))
    # quantile function: mu - beta ln(-ln(U))
    noise = torch.rand(input.size())
    eps = 1e-20
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    return Variable(noise.cuda())


def gumbel_softmax_sample(input):
    temperature = 1
    noise = sample_gumbel(input)
    x = (input + noise) / temperature
    x = F.softmax(x)
    return x.view_as(input)


class Minibach_stochastic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, d_size,
                 s1_size, s2_size, s3_size):
        super(Minibach_stochastic, self).__init__()
        self.filepath = 'models/minibach_stochastic_pytorch.h5'
        self.epsilon = 1e-4

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.d_size = d_size
        self.s1_size = s1_size
        self.s2_size = s2_size
        self.s3_size = s3_size

        # todo add hidden layer?
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2_d = nn.Linear(hidden_size, d_size)

        self.l2_s1 = nn.Linear(hidden_size, s1_size)
        self.l2_s2 = nn.Linear(hidden_size, s2_size)
        self.l2_s3 = nn.Linear(hidden_size, s3_size)

        self.l3 = nn.Linear(d_size + s1_size + s2_size + s3_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, output_size)

    def forward(self, input):

        hidden_layer = F.relu(
            self.l1(
                F.dropout(input, p=0.2))
        )

        hidden_layer = F.dropout(hidden_layer, p=0.5)

        d_layer = F.relu(
            self.l2_d(
                hidden_layer)
        )

        # compute log(alpha_k) for every stochastic node and draw samples
        s1_layer = gumbel_softmax_sample(self.l2_s1(hidden_layer))
        s2_layer = gumbel_softmax_sample(self.l2_s2(hidden_layer))
        s3_layer = gumbel_softmax_sample(self.l2_s3(hidden_layer))
        merge = torch.cat([d_layer, s1_layer, s2_layer, s3_layer], 1)

        predictions = F.sigmoid(self.l4(F.relu(self.l3(merge))))

        return predictions

    def loss_and_acc_on_epoch(self, batches_per_epoch, generator, train=True):
        mean_loss = 0
        mean_accuracy = 0
        sum_constraints, num_constraints = 0, 0
        for sample_id, next_element in tqdm(enumerate(islice(generator, batches_per_epoch))):
            input, target = next_element

            # todo requires_grad?
            input, target = Variable(torch.FloatTensor(input).cuda()), Variable(torch.FloatTensor(target).cuda())

            optimizer.zero_grad()
            output = self(input)

            loss = F.binary_cross_entropy(output, target=target)

            if train:
                loss.backward()
                optimizer.step()

            # compute mean loss and accuracy
            mean_loss += loss.data.mean()
            # todo accuracy
            # seq_accuracy, (sum_constraint, num_constraint) = accuracy(output_seq=output, targets_seq=input_seq_index)
            seq_accuracy, (sum_constraint, num_constraint) = 1, (1, 1)

            mean_accuracy += seq_accuracy
            sum_constraints += sum_constraint
            num_constraints += num_constraint

        return mean_loss / batches_per_epoch, mean_accuracy / batches_per_epoch, sum_constraints / num_constraints

    def train_model(self, batches_per_epoch, num_epochs, plot=False):

        if plot:
            import matplotlib.pyplot as plt
            # plt.ion()
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            fig, axarr = plt.subplots(3, sharex=True)
            x, y_loss, y_acc = [], [], []
            y_val_loss, y_val_acc = [], []

            # line1, = ax.plot(x, y, 'ko')
            fig.show()

        for epoch_index in range(num_epochs):
            self.train()
            mean_loss, mean_accuracy, constraint_accuracy = self.loss_and_acc_on_epoch(
                batches_per_epoch=batches_per_epoch,
                generator=generator_train, train=True)
            self.eval()
            mean_val_loss, mean_val_accuracy, constraint_val_accuracy = self.loss_and_acc_on_epoch(
                batches_per_epoch=int(batches_per_epoch / 5),
                generator=generator_val, train=False)
            print(
                f'Train Epoch: {epoch_index}/{num_epochs} \tLoss: {mean_loss}\tAccuracy: {mean_accuracy * 100} %'
            )
            print(
                f'\tValidation Loss: {mean_val_loss}\tValidation Accuracy: {mean_val_accuracy * 100} %'
            )

        if plot:
            x.append(epoch_index)

            y_loss.append(mean_loss)
            y_acc.append(mean_accuracy * 100)

            y_val_loss.append(mean_val_loss)
            y_val_acc.append(mean_val_accuracy * 100)

            axarr[0].plot(x, y_loss, 'r-', x, y_val_loss, 'r--')
            axarr[1].plot(x, y_acc, 'r-', x, y_val_acc, 'r--')

            fig.canvas.draw()
            plt.pause(0.001)

    def save(self):
        torch.save(self.state_dict(), self.filepath)

    def load(self):
        self.load_state_dict(torch.load(self.filepath))


class Minibach(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):

        super(Minibach, self).__init__()
        self.filepath = 'models/minibach_pytorch.h5'
        self.epsilon = 1e-4

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # todo add hidden layer?
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, input):

        hidden_layer = F.relu(
            self.l1(
                F.dropout(input, p=0.2))
        )

        hidden_layer = F.dropout(hidden_layer, p=0.5)

        predictions = F.sigmoid(
            self.l2(
                hidden_layer)
        )

        return predictions

    def loss_and_acc_on_epoch(self, batches_per_epoch, generator, train=True):
        mean_loss = 0
        mean_accuracy = 0
        epsilon = 1e-7
        sum_constraints, num_constraints = 0, 0
        for sample_id, next_element in tqdm(enumerate(islice(generator, batches_per_epoch))):
            input, target = next_element

            # todo requires_grad?
            input, target = Variable(torch.FloatTensor(input).cuda()), Variable(torch.FloatTensor(target).cuda())

            optimizer.zero_grad()
            output = self(input)
            output = torch.clamp(output, min=epsilon, max=1 - epsilon)

            loss = F.binary_cross_entropy(output, target=target)
            # loss = - (torch.log(output) * target + torch.log(1 - output) * (1 - target)).mean()

            if train:
                loss.backward()
                optimizer.step()

            # compute mean loss and accuracy
            mean_loss += loss.data.mean()
            # todo accuracy
            # seq_accuracy, (sum_constraint, num_constraint) = accuracy(output_seq=output, targets_seq=input_seq_index)
            seq_accuracy, (sum_constraint, num_constraint) = 1, (1, 1)

            mean_accuracy += seq_accuracy
            sum_constraints += sum_constraint
            num_constraints += num_constraint

        return mean_loss / batches_per_epoch, mean_accuracy / batches_per_epoch, sum_constraints / num_constraints

    def train_model(self, batches_per_epoch, num_epochs, plot=False):

        if plot:
            import matplotlib.pyplot as plt
            # plt.ion()
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            fig, axarr = plt.subplots(3, sharex=True)
            x, y_loss, y_acc = [], [], []
            y_val_loss, y_val_acc = [], []

            # line1, = ax.plot(x, y, 'ko')
            fig.show()

        for epoch_index in range(num_epochs):
            self.train()
            mean_loss, mean_accuracy, constraint_accuracy = self.loss_and_acc_on_epoch(
                batches_per_epoch=batches_per_epoch,
                generator=generator_train, train=True)
            self.eval()
            mean_val_loss, mean_val_accuracy, constraint_val_accuracy = self.loss_and_acc_on_epoch(
                batches_per_epoch=int(batches_per_epoch / 5),
                generator=generator_val, train=False)
            print(
                f'Train Epoch: {epoch_index}/{num_epochs} \tLoss: {mean_loss}\tAccuracy: {mean_accuracy * 100} %'
            )
            print(
                f'\tValidation Loss: {mean_val_loss}\tValidation Accuracy: {mean_val_accuracy * 100} %'
            )

        if plot:
            x.append(epoch_index)

            y_loss.append(mean_loss)
            y_acc.append(mean_accuracy * 100)

            y_val_loss.append(mean_val_loss)
            y_val_acc.append(mean_val_accuracy * 100)

            axarr[0].plot(x, y_loss, 'r-', x, y_val_loss, 'r--')
            axarr[1].plot(x, y_acc, 'r-', x, y_val_acc, 'r--')

            fig.canvas.draw()
            plt.pause(0.001)

    def save(self):
        torch.save(self.state_dict(), self.filepath)

    def load(self):
        self.load_state_dict(torch.load(self.filepath))


if __name__ == '__main__':
    batch_size = 128
    timesteps = 16
    generator_train = generator_pianoroll(batch_size, timesteps=timesteps, phase='train')
    generator_val = generator_pianoroll(batch_size, timesteps=timesteps, phase='test')

    features, target = next(generator_train)
    input_size = features.shape[-1]
    output_size = target.shape[-1]

    # Choose first line if model does not exist
    # model = Minibach(input_size, output_size,
    #                  hidden_size=1280)

    model = Minibach_stochastic(input_size, output_size,
                                hidden_size=512,
                                d_size=1280,
                                s1_size=2,
                                s2_size=3,
                                s3_size=12)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters())
    model.load()
    model.train_model(batches_per_epoch=100, num_epochs=1000,
                      plot=False)
    model.save()

    # Generate example
    # load dataset
    X, X_metadatas, voice_ids, index2notes, note2indexes, metadatas = pickle.load(open(BACH_DATASET, 'rb'))
    num_pitches = list(map(lambda x: len(x), index2notes))
    num_voices = len(voice_ids)
    del X, X_metadatas

    # pick up one example
    features, target = next(generator_val)
    features = features[0]
    target = target[0]

    # show original chorale
    reconstructed_chorale = reconstruct(features, target, num_pitches, timesteps, num_voices)
    score = indexed_chorale_to_score(reconstructed_chorale, BACH_DATASET)
    score.show()

    # show predicted chorale
    predictions = model(Variable(torch.FloatTensor(features[None, :]).cuda(), volatile=True))[0].data.cpu().numpy()
    reconstructed_predicted_chorale = reconstruct(features, predictions, num_pitches, timesteps, num_voices)
    score = indexed_chorale_to_score(reconstructed_predicted_chorale, BACH_DATASET)
    score.show()

    # show predicted chorale
    predictions = model(Variable(torch.FloatTensor(features[None, :]).cuda(), volatile=True))[0].data.cpu().numpy()
    reconstructed_predicted_chorale = reconstruct(features, predictions, num_pitches, timesteps, num_voices)
    score = indexed_chorale_to_score(reconstructed_predicted_chorale, BACH_DATASET)
    score.show()

