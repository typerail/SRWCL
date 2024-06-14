from model import Discriminator
from model import Encoder_Decoder
from torch.autograd import Variable
import torch.optim as optim
import torch
import numpy as np
import os
import time
import torch.nn as nn
import cv2
import torch.nn.functional as F


def hinge_loss_dis(output_fake, output_real):
    r"""
    Hinge loss for discriminator.

    Args:
        output_fake (Tensor): Discriminator output logits for fake images.
        output_real (Tensor): Discriminator output logits for real images.

    Returns:
        Tensor: A scalar tensor loss output.
    """
    loss = F.relu(1.0 - output_real).mean() + \
           F.relu(1.0 + output_fake).mean()

    return loss


def hinge_loss_gen(output_fake):
    r"""
    Hinge loss for generator.

    Args:
        output_fake (Tensor): Discriminator output logits for fake images.

    Returns:
        Tensor: A scalar tensor loss output.
    """
    loss = -output_fake.mean()

    return loss


def infonce_loss(l, m):
    r"""
    InfoNCE loss for local and global feature maps as used in DIM:
    https://github.com/rdevon/DIM/blob/master/cortex_DIM/functions/dim_losses.py

    Args:
        l (Tensor): Local feature map of shape (N, ndf, H*W).
        m (Tensor): Global feature vector of shape (N, ndf, 1).
    Returns:
        Tensor: Scalar loss Tensor.
    """

    l = torch.flatten(l, start_dim=2,
                               end_dim=3)  # (N, C, H, W) --> (N, C, H*W)
    m = torch.unsqueeze(m, 2)  # (N, C) --> (N, C, 1)
    # l = l.view(8, 64, 32 * 32)
    # m = m.view(8, 64, 1)
    N, units, n_locals = l.size()
    _, _, n_multis = m.size()

    # First we make the input tensors the right shape.
    l_p = l.permute(0, 2, 1)
    m_p = m.permute(0, 2, 1)

    l_n = l_p.reshape(-1, units)
    m_n = m_p.reshape(-1, units)

    # Inner product for positive samples. Outer product for negative. We need to do it this way
    # for the multiclass loss. For the outer product, we want a N x N x n_local x n_multi tensor.
    u_p = torch.matmul(l_p, m).unsqueeze(2)
    u_n = torch.mm(m_n, l_n.t())
    u_n = u_n.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1)

    # We need to mask the diagonal part of the negative tensor.
    mask = torch.eye(N)[:, :, None, None].to(l.device)
    n_mask = 1 - mask

    # Masking is done by shifting the diagonal before exp.
    u_n = (n_mask * u_n) - (10. * (1 - n_mask))  # mask out "self" examples
    u_n = u_n.reshape(N, N * n_locals, n_multis).unsqueeze(dim=1).expand(
        -1, n_locals, -1, -1)

    # Since this is multiclass, we concat the positive along the class dimension before performing log softmax.
    pred_lgt = torch.cat(
        [u_p, u_n], dim=2
    )  # So the first of each "row" is positive, and we have N+1 elements
    pred_log = F.log_softmax(pred_lgt, dim=2)

    # The positive score is the first element of the log softmax.
    loss = -pred_log[:, :, 0].mean()

    return loss

class Solver(object):
    """Solver for training and testing PIMoG."""

    def __init__(self, data_loader, data_loader_test, config):
        """Initialize configurations."""

        # Data loader.
        self.data_loader = data_loader
        self.data_loader_test = data_loader_test
        # Model configurations.
        self.image_size = config.image_size
        self.num_channels = config.num_channels
        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.lambda1 = config.lambda1
        self.lambda2 = config.lambda2
        self.lambda3 = config.lambda3
        self.num_epoch = config.num_epoch
        self.distortion = config.distortion

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.model_save_dir = config.model_save_dir
        self.model_name = config.model_name
        self.result_dir = config.result_dir
        self.embedding_epoch = config.embedding_epoch

        # Step size.
        self.log_step = config.log_step
        self.model_save_step = config.model_save_step

        # Build the model.
        self.build_model()

    def build_model(self):
        if self.dataset in ['train_mask']:
            self.net = Encoder_Decoder(self.distortion)
            self.net_Discriminator = Discriminator(self.num_channels)
            self.net_Discriminator.to(self.device)
            self.optimizer_Discriminator = optim.Adam(self.net_Discriminator.parameters())
            self.net_optimizer = torch.optim.Adam(self.net.parameters())
            self.print_network(self.net, self.dataset)
            self.net.to(self.device)
            self.net = torch.nn.DataParallel(self.net)
            if self.embedding_epoch != 0:
                self.net.load_state_dict(torch.load(
                    self.model_save_dir + '/' + self.distortion + '/' + self.model_name + '_mask_' + str(
                        self.embedding_epoch) + '.pth'))
        elif self.dataset in ['test_embedding']:
            self.net_ED = Encoder_Decoder(self.distortion)
            self.net_ED = self.net_ED.to(self.device)
            self.net_E = self.net_ED.Encoder
            self.net_ED = torch.nn.DataParallel(self.net_ED)
            self.net_ED.load_state_dict(torch.load(
                self.model_save_dir + '/' + 'SRWCL.pth'))
        elif self.dataset in ['test_accuracy']:
            self.net = Encoder_Decoder(self.distortion)
            self.print_network(self.net, self.dataset)
            self.net.to(self.device)
            self.net_D = self.net.Decoder
            self.net = torch.nn.DataParallel(self.net)
            self.net.load_state_dict(torch.load(
                self.model_save_dir + '/' + 'SRWCL.pth'))

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()

    def test_embedding(self):
        # Set data loader.
        data_loader = self.data_loader
        data_loader_test = self.data_loader_test
        criterion_MSE = nn.MSELoss()
        self.net_ED.eval()
        for i, (data, m, num, name) in enumerate(data_loader):
            inputs, m = Variable(data), Variable(m.float())
            inputs, m = inputs.to(self.device), m.to(self.device)
            inputs.requires_grad = True
            num = num.to('cpu').numpy()
            Encoded_image, Noised_image, Decoded_message = self.net_ED(inputs, m)
            loss_de = criterion_MSE(Decoded_message, m)
            loss_de.backward()
            inputgrad = inputs.grad.data
            mask = torch.zeros(inputgrad.shape).to(self.device)
            for ii in range(inputgrad.shape[0]):
                a = inputgrad[ii, :, :, :]
                a = (1 - (a - a.min()) / (a.max() - a.min())) + 1
                mask[ii, :, :, :] = a

            for j in range(Encoded_image.shape[0]):
                I1 = (inputs[j, :, :, :].detach().to('cpu').numpy() + 1) / 2 * 255
                I1 = np.transpose(I1, (1, 2, 0))
                I2 = (Encoded_image[j, :, :, :].detach().to('cpu').numpy() + 1) / 2 * 255
                I2 = np.transpose(I2, (1, 2, 0))
                I_no = (Noised_image[j, :, :, :].detach().to('cpu').numpy() + 1) / 2 * 255
                I_no = np.transpose(I_no, (1, 2, 0))
                I_mask = (mask[j, :, :, :].detach().to('cpu').numpy() - 1) * 255
                I_mask = np.transpose(I_mask, (1, 2, 0))
                I_res = (I2 - I1) * 5
                I5 = np.zeros((I1.shape[0], I1.shape[1] * 5, I1.shape[2]))
                I5[:, :I1.shape[1], :] = I1
                I5[:, I1.shape[1]:I1.shape[1] * 2, :] = I2
                I5[:, I1.shape[1] * 2:I1.shape[1] * 3, :] = I_no
                I5[:, I1.shape[1] * 3:I1.shape[1] * 4, :] = I_mask
                I5[:, I1.shape[1] * 4:I1.shape[1] * 5, :] = I_res
                index = num[j]
                cv2.imwrite('./results/embed/' + str(index) + '.png', I2)
        print('Embed finished!')



    def train_mask(self):
        # Set data loader.
        data_loader = self.data_loader
        data_loader_test = self.data_loader_test
        criterion = nn.BCEWithLogitsLoss()
        criterion_MSE = nn.MSELoss()
        start_epoch = self.embedding_epoch

        # Start training.
        print('Start training...')
        start_time = time.time()
        txtfile = open(self.log_dir + '/' + self.dataset + '_' + self.distortion + '.txt', 'w', encoding="utf-8")
        best_acc = 0.3
        for epoch in range(start_epoch, self.num_epoch):
            running_loss = 0.0

            for i, (data, m, v_mask) in enumerate(data_loader):
                inputs, m, v_mask = Variable(data), Variable(m.float()), Variable(v_mask)
                inputs, m, v_mask = inputs.to(self.device), m.to(self.device), v_mask.to(self.device)
                inputs.requires_grad = True
                Encoded_image, Noised_image, Decoded_message = self.net(inputs, m)
                loss_de = criterion_MSE(Decoded_message, m)

                inputgrad = torch.autograd.grad(loss_de, inputs, create_graph=True)[0]
                mask = torch.zeros(inputgrad.shape).to(self.device)
                for ii in range(inputgrad.shape[0]):
                    a = inputgrad[ii, :, :, :]
                    a = (1 - (a - a.min()) / (a.max() - a.min())) + 1
                    mask[ii, :, :, :] = a.detach()
                d_label_host = torch.full((inputs.shape[0], 1), 1, dtype=torch.float, device=self.device)
                d_label_decoded = torch.full((inputs.shape[0], 1), 0, dtype=torch.float, device=self.device)
                g_label_decoded = torch.full((inputs.shape[0], 1), 1, dtype=torch.float, device=self.device)

                # train the discriminator
                self.optimizer_Discriminator.zero_grad()
                i_generate, i_local, i_glob = self.net_Discriminator(inputs.detach())
                e_generate, e_local, e_glob = self.net_Discriminator(Encoded_image.detach())
                i_nce_loss = infonce_loss(i_local, i_glob)
                i_d_loss = hinge_loss_dis(e_generate, i_generate)
                d_loss = i_d_loss + 0.2 * i_nce_loss

                d_loss.backward()

                self.optimizer_Discriminator.step()

                # train the Encoder_Decoder
                g_decoded, local, glob = self.net_Discriminator(Encoded_image)
                i_nce_loss = infonce_loss(local, glob)
                i_gan_loss = hinge_loss_gen(g_decoded)
                g_loss = i_gan_loss + 0.2 * i_nce_loss

                loss_message = criterion_MSE(Decoded_message, m)

                loss_denoise = criterion_MSE(Encoded_image * mask.float(), inputs * mask.float()) * 2 + criterion_MSE(
                    Encoded_image * v_mask.float(), inputs * v_mask.float()) * 0.4
                print(v_mask.shape)
                print(v_mask)
                loss = loss_message * self.lambda1 + loss_denoise * self.lambda2 + g_loss * self.lambda3

                self.net_optimizer.zero_grad()
                loss.backward()

                self.net_optimizer.step()
                running_loss += loss.item()


                if i % 10 == 9:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 10))
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 10), file=txtfile)
                    running_loss = 0.0
                    print('message loss:%.3f, denoise loss:%.3f, gan_loss:%.3f, d_loss:%.3f' % (
                    loss_message.item(), loss_denoise.item(), g_loss.item(), d_loss.item()))
                    print('message loss:%.3f, denoise loss:%.3f, gan_loss:%.3f, d_loss:%.3f' % (
                    loss_message.item(), loss_denoise.item(), g_loss.item(), d_loss.item()), file=txtfile)

            self.net.eval()
            t = np.random.randint(inputs.shape[0])
            I1 = (inputs[t, :, :, :].detach().to('cpu').numpy() + 1) / 2 * 255
            I1 = np.transpose(I1, (1, 2, 0))
            I2 = (Encoded_image[t, :, :, :].detach().to('cpu').numpy() + 1) / 2 * 255
            I2 = np.transpose(I2, (1, 2, 0))
            I_no = (Noised_image[t, :, :, :].detach().to('cpu').numpy() + 1) / 2 * 255
            I_no = np.transpose(I_no, (1, 2, 0))
            I_mask = (mask[t, :, :, :].detach().to('cpu').numpy() - 1) * 255
            I_mask = np.transpose(I_mask, (1, 2, 0))
            I_vmask = (v_mask[t, :, :, :].detach().to('cpu').numpy() - 1) * 255
            I_vmask = np.transpose(I_vmask, (1, 2, 0))
            I_res = (I2 - I1) * 5
            I5 = np.zeros((I1.shape[0], I1.shape[1] * 6, I1.shape[2]))
            I5[:, :I1.shape[1], :] = I1
            I5[:, I1.shape[1]:I1.shape[1] * 2, :] = I2
            I5[:, I1.shape[1] * 2:I1.shape[1] * 3, :] = I_no
            I5[:, I1.shape[1] * 3:I1.shape[1] * 4, :] = I_mask
            I5[:, I1.shape[1] * 4:I1.shape[1] * 5, :] = I_vmask
            I5[:, I1.shape[1] * 5:I1.shape[1] * 6, :] = I_res
            if not os.path.exists(self.result_dir + '/Image/images/'):
                os.makedirs(self.result_dir + '/Image/images/')
            cv2.imwrite(self.result_dir + '/Image/images/' + str(epoch) + '.png', I5)

            print('validation!')
            print('validation!', file=txtfile)

            correct = 0
            total = 0
            for i, (data, m, v_mask) in enumerate(data_loader_test):
                inputs, m, v_mask = Variable(data), Variable(m.float()), Variable(v_mask)
                inputs, m, v_mask = inputs.to(self.device), m.to(self.device), v_mask.to(self.device)
                Encoded_image, Noised_image, Decoded_message = self.net(inputs, m)
                decoded_rounded = Decoded_message.detach().cpu().numpy().round().clip(0, 1)
                correct += np.sum(np.abs(decoded_rounded - m.detach().cpu().numpy()))
                total += inputs.shape[0] * m.shape[1]
            print("Correct Rate:%.3f" % ((1 - correct / total) * 100) + '%')
            print("Correct Rate:%.3f" % ((1 - correct / total) * 100) + '%', file=txtfile)

            if not os.path.exists(self.model_save_dir + '/' + self.distortion + '/'):
                os.makedirs(self.model_save_dir + '/' + self.distortion + '/')
            PATH_Encoder_Decoder = self.model_save_dir + '/' + self.distortion + '/' + self.model_name + '_mask_' + str(
                epoch) + '.pth'

            if epoch % (self.model_save_step) == (self.model_save_step - 1):
                torch.save(self.net.state_dict(), PATH_Encoder_Decoder)
            PATH_Encoder_Decoder_best = self.model_save_dir + '/' + self.model_name + '_mask_' + str(
                self.distortion) + '_best.pth'

            if 1 - correct / total >= best_acc:
                best_acc = 1 - correct / total
                torch.save(self.net.state_dict(), PATH_Encoder_Decoder_best)
            self.net.train()

    def test_accuracy(self):
        # Set data loader.
        data_loader_test = self.data_loader_test

        correct = 0
        total = 0
        for i, (data, m, num, name) in enumerate(data_loader_test):
            inputs, m = Variable(data), Variable(m.float())
            inputs, m = inputs.to(self.device), m.to(self.device)
            self.net_D.eval()
            Decoded_message = self.net_D(inputs)
            decoded_rounded = Decoded_message.detach().cpu().numpy().round().clip(0, 1)
            #print(decoded_rounded)
            p1 = np.sum(np.abs(decoded_rounded - m.detach().cpu().numpy()))
            p2 = inputs.shape[0] * m.shape[1]
            print('Accuracy of ' + name[0] + ' image: %.3f' % ((1 - p1 / p2) * 100) + '%')
            correct += np.sum(np.abs(decoded_rounded - m.detach().cpu().numpy()))
            total += inputs.shape[0] * m.shape[1]

        print('Accuracy of ' + self.distortion + ' image: %.3f' % ((1 - correct / total) * 100) + '%')
