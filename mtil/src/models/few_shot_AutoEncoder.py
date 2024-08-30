import torchvision.models as models
import torch.nn as nn
import clip.clip as clip
from torch.utils.data import DataLoader

from .. import datasets, templates, utils
from tqdm import tqdm
import torch.optim as optim
import torch
import torch.nn.functional as F
import os
from ..datasets.common import get_dataloader, maybe_dictionarize

class few_shot_autoencoder(nn.Module):
    """
    The class defines the autoencoder model which takes in the features from the last convolutional layer of the
    Alexnet model. The default value for the input_dims is 256*13*13.
    """

    def __init__(self, input_dims=256 * 13 * 13, code_dims=100):
        super(few_shot_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dims, code_dims),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(code_dims, input_dims),
            nn.Sigmoid())

    def forward(self, x):
        encoded_x = self.encoder(x)
        reconstructed_x = self.decoder(encoded_x)
        return reconstructed_x


class Alexnet_FE(nn.Module):
    """
    Create a feature extractor model from an Alexnet architecture, that is used to train the autoencoder model
    and get the most related model whilst training a new task in a sequence
    """

    def __init__(self, alexnet_model):
        super(Alexnet_FE, self).__init__()
        self.fe_model = nn.Sequential(*list(alexnet_model.children())[0][:-2])
        self.fe_model.train = False

    def forward(self, x):
        return self.fe_model(x)

# def exp_lr_scheduler(optimizer, epoch, init_lr=0.008, lr_decay_epoch=10):
def exp_lr_scheduler(optimizer, epoch, init_lr=0.0008, lr_decay_epoch=10):
    """
    Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.

    """
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def encoder_criterion(outputs, inputs):
    loss = nn.MSELoss()
    return loss(outputs, inputs)

def few_shot_AutoEncoder(args):
    if args.eval_only == True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        _, _, val_preprocess = clip.load(args.model, jit=False, args=args)
        pretrained_alexnet = models.alexnet(pretrained=True)
        # Derives a feature extractor model from the Alexnet model
        feature_extractor = Alexnet_FE(pretrained_alexnet).to(device)
        Autoencoder_list = nn.ModuleList()
        for i in range(12):
            model_autoencoder = few_shot_autoencoder(256 * 13 * 13).to(device)
            Autoencoder_list.append(model_autoencoder)
        if args.load:
            utils.torch_load(Autoencoder_list, args.load) 
        task_acc_list = []
        
        for j, dataset_name in enumerate(args.eval_datasets):
            print("Evaluating on", dataset_name)  # Caltech101
            dataset_class = getattr(datasets, args.eval_datasets[j])
            dataset = dataset_class(
                val_preprocess,
                location=args.data_location,
                batch_size=args.batch_size,
                batch_size_eval=args.batch_size_eval,
            )
            image_enc = None
            dataloader = get_dataloader(
                dataset, is_train=False, args=args, image_encoder=image_enc
            )
            best_router_list = []
            best_loss_list = []
            worst_loss = 0
            for _, data in enumerate(tqdm(dataloader)):
                data = maybe_dictionarize(data)
                images = data["images"].cuda()
                input_to_ae = feature_extractor(images)
                input_to_ae = input_to_ae.view(input_to_ae.size(0), -1)
                input_to_ae = input_to_ae.to(device)
                input_to_ae = F.sigmoid(input_to_ae) # GT

                model_autoencoder = Autoencoder_list[0].to(device)
                outputs = model_autoencoder(input_to_ae)
                best_l = encoder_criterion(outputs, input_to_ae)
                # print('0', best_l)
                best_router = 0
                for i in range(1, 11):
                    outputs = Autoencoder_list[i](input_to_ae)
                    new_l = encoder_criterion(outputs, input_to_ae)
                    # print('i', i, new_l)
                    if new_l < best_l:
                        best_l = new_l
                        best_router = i
                    if best_l>args.threshold:
                        best_router = -1
                    # print(best_l)
                best_router_list.append(best_router)
                best_loss_list.append(best_l.detach().cpu().numpy())
                if best_l>worst_loss:
                    worst_loss = best_l
            # print("max loss of every task", worst_loss)
            # print(best_router_list)
            # print(best_loss_list)
            count_of_zeros = best_router_list.count(j)

            total_elements = len(best_router_list)
            probability_of_zero = count_of_zeros / total_elements

            probability_percentage = probability_of_zero * 100

            
            task_acc_list.append(probability_percentage)
        # print(task_acc_list)

    else:
        # train
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        _, train_preprocess, val_preprocess = clip.load(args.model, jit=False, args=args)  # model='ViT-B/16'
        Autoencoder_list = nn.ModuleList()
        for i in range(12):
            model_autoencoder = few_shot_autoencoder(256 * 13 * 13).to(device)
            Autoencoder_list.append(model_autoencoder)
        # Initial model
        if args.load is not None:
            utils.torch_load(Autoencoder_list, args.load)

        model_autoencoder = Autoencoder_list[args.task_id]
        pretrained_alexnet = models.alexnet(pretrained=True)
        for k, v in pretrained_alexnet.named_parameters():  # frozen Alex
                v.requires_grad = False
        # Derives a feature extractor model from the Alexnet model
        feature_extractor = Alexnet_FE(pretrained_alexnet).to(device)

        dataset_class = getattr(datasets, args.train_dataset)
        dataset = dataset_class(  #
            train_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
        )

        # few-shot
        few_shot_data = {}

        for images, labels in dataset.train_loader:
            for image, label in zip(images, labels):
                label = label.item()
                if label not in few_shot_data:
                    few_shot_data[label] = []
                if len(few_shot_data[label]) < 5:
                    few_shot_data[label].append(image)

        few_shot_images = []
        few_shot_labels = []

        for label, images in few_shot_data.items():
            few_shot_images.extend(images)
            few_shot_labels.extend([label] * len(images))

        few_shot_images = torch.stack(few_shot_images)
        few_shot_labels = torch.tensor(few_shot_labels)

        few_shot_dataset = torch.utils.data.TensorDataset(few_shot_images, few_shot_labels)
        few_shot_data_loader = DataLoader(few_shot_dataset, batch_size=64, shuffle=True)
        print(len(few_shot_data_loader))

        optimizer_encoder = optim.Adam(model_autoencoder.parameters(), lr=0.003, weight_decay=0.0001)
        num_batches = len(dataset.train_loader)
        total_iterations = args.iterations

        running_loss = 0

        for iteration in tqdm(range(total_iterations + 1)):
            if iteration % num_batches == 0:
                data_iter = iter(few_shot_data_loader)

            try:
                images, labels = next(data_iter)
            except:
                data_iter = iter(few_shot_data_loader)
                images, labels = next(data_iter)
            images, labels = images.cuda(), labels.cuda()
            input_to_ae = feature_extractor(images)
            input_to_ae = input_to_ae.view(input_to_ae.size(0), -1)

            optimizer = exp_lr_scheduler(optimizer_encoder, int(iteration/num_batches), 0.003)
            optimizer.zero_grad()
            model_autoencoder.zero_grad()

            input_to_ae = input_to_ae.to(device)
            input_to_ae = F.sigmoid(input_to_ae)
            model_autoencoder.to(device)

            outputs = model_autoencoder(input_to_ae)
            loss = encoder_criterion(outputs, input_to_ae)
            loss.backward()
            optimizer.step()
            # if iteration % 100 == 0:
            #     dataset_class = getattr(datasets, args.train_dataset)
            #     dataset = dataset_class(
            #         val_preprocess,
            #         location=args.data_location,
            #         batch_size=args.batch_size,
            #         batch_size_eval=args.batch_size_eval,
            #     )
            #     image_enc = None
            #     dataloader = get_dataloader(
            #         dataset, is_train=False, args=args, image_encoder=image_enc
            #     )
            #     best_router_list = []
            #     best_loss_list = []
            #     worst_loss = 0
            #     for _, data in enumerate(tqdm(dataloader)):
            #         data = maybe_dictionarize(data)
            #         images = data["images"].cuda()
            #         input_to_ae = feature_extractor(images)
            #         input_to_ae = input_to_ae.view(input_to_ae.size(0), -1)
            #         input_to_ae = input_to_ae.to(device)
            #         input_to_ae = F.sigmoid(input_to_ae)  # GT
            #
            #         outputs = model_autoencoder(input_to_ae)
            #         best_l = encoder_criterion(outputs, input_to_ae)
            #
            #         best_loss_list.append(best_l.detach().cpu().numpy())
            #     # print("max loss of every task", max(best_loss_list))


        # Saving model
        if args.save is not None:
            to_save_model = Autoencoder_list
            path = os.path.join(args.save, f"{args.train_dataset}_autochooser.pth")
            utils.torch_save(to_save_model, path)
