import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils import merge_test_pred
import gc

from models import Generator, Discriminator


resolution_list = ["4x4", "8x8", "16x16", "32x32", "64x64", "128x128", "256x256"]
dataset_path = [f"./dataset/{i}" for i in resolution_list]
model_state_dict_path = [f"./model_state_dict/{i}" for i in resolution_list]



class Trainer():

    def __init__(self,
                steps: int,
                batch_size: int,
                device: torch.device,
                test_size: int
            ):

        self.steps = steps
        self.batch_size = batch_size
        self.device = device
        self.test_size = test_size

        directory_path = dataset_path[self.steps]

        self.trainloader = DataLoader(torch.cat((torch.load(f"{directory_path}/train_cat.pt"), torch.load(f"{directory_path}/train_dog.pt")), dim=0).type(torch.float32), batch_size=self.batch_size, shuffle=True)
        self.validloader = DataLoader(torch.cat((torch.load(f"{directory_path}/valid_cat.pt"), torch.load(f"{directory_path}/valid_dog.pt")), dim=0).type(torch.float32), batch_size=self.batch_size, shuffle=True)

        self.generator = Generator(steps=self.steps).to(self.device)
        self.discriminator = Discriminator(steps=self.steps).to(self.device)

        self.criterion = nn.BCELoss()
        self.generator_optim = Adam(self.generator.parameters(), lr=0.002, betas=(0.5, 0.999))
        self.discriminator_optim = Adam(self.discriminator.parameters(), lr=0.002, betas=(0.5, 0.999))

        # It will be used for testing.
        self.test_z = torch.randn((self.test_size, 128, 1, 1)).to(self.device)

        self.load_model()


    def save_model(self):
        # ------------------------------------------------------------------ generator model ---------------------------------------------------------------------------
        for i in range(self.steps+1):
            torch.save(self.generator.prog_blocks[i].state_dict(), f"{model_state_dict_path[self.steps]}/generator_model/prog_blocks_{i}.pt")
            torch.save(self.generator.torgb_layers[i].state_dict(), f"{model_state_dict_path[self.steps]}/generator_model/torgb_layers_{i}.pt")

        # ---------------------------------------------------------------- discriminator model -------------------------------------------------------------------------
        for i in range(self.steps+1):
            torch.save(self.discriminator.prog_blocks[i].state_dict(), f"{model_state_dict_path[self.steps]}/discriminator_model/prog_blocks_{i}.pt")
            torch.save(self.discriminator.fromrgb_layers[i].state_dict(), f"{model_state_dict_path[self.steps]}/discriminator_model/fromrgb_layers_{i}.pt")
    

    def load_model(self):
        if self.steps == 0:
            return

        # ------------------------------------------------------------------ generator model ---------------------------------------------------------------------------
        for i in range(self.steps):
            self.generator.prog_blocks[i].load_state_dict(torch.load(f"{model_state_dict_path[self.steps-1]}/generator_model/prog_blocks_{i}.pt"))
            self.generator.torgb_layers[i].load_state_dict(torch.load(f"{model_state_dict_path[self.steps-1]}/generator_model/torgb_layers_{i}.pt"))

        # ---------------------------------------------------------------- discriminator model -------------------------------------------------------------------------
        for i in range(1, self.steps+1):
            self.discriminator.prog_blocks[i].load_state_dict(torch.load(f"{model_state_dict_path[self.steps-1]}/discriminator_model/prog_blocks_{i-1}.pt"))
            self.discriminator.fromrgb_layers[i].load_state_dict(torch.load(f"{model_state_dict_path[self.steps-1]}/discriminator_model/fromrgb_layers_{i-1}.pt"))
    

    def clear_cuda_memory(self):
        gc.collect()
        torch.cuda.empty_cache()


    def test(self, epoch):
        self.generator.eval()
        self.discriminator.eval()

        pred = self.generator(self.test_z, alpha=self.alpha)
        pred = pred.detach().cpu()

        test_image = merge_test_pred(pred)
        test_image.save(fp=f"./train_log/{resolution_list[self.steps]}/epoch-{epoch}.jpg")


    def train(self):
        self.generator.train()
        self.discriminator.train()

        generator_avg_loss = 0
        discriminator_avg_loss = 0

        for _ in range(len(self.trainloader)):
            self.alpha += self.alpha_gap

            real_image = next(iter(self.trainloader)).to(self.device)

            real_label = torch.full((real_image.size(0), 1), 1).type(torch.float).to(self.device)
            fake_label = torch.full((real_image.size(0), 1), 0).type(torch.float).to(self.device)

            # ---------------------------------------------------------- discriminator train ------------------------------------------------------------
            z = torch.randn(real_image.size(0), 128, 1, 1).to(self.device)

            fake_image = self.generator(z, alpha=self.alpha)
            
            d_fake_pred = self.discriminator(fake_image, alpha=self.alpha)
            d_fake_loss = self.criterion(d_fake_pred, fake_label)

            d_real_pred = self.discriminator(real_image, alpha=self.alpha)
            d_real_loss = self.criterion(d_real_pred, real_label)

            d_loss = d_fake_loss + d_real_loss

            self.discriminator_optim.zero_grad()
            d_loss.backward()
            self.discriminator_optim.step()

            discriminator_avg_loss += (d_loss.item() / 2)

            # ---------------------------------------------------------- generator train -----------------------------------------------------------------
            z = torch.randn(real_image.size(0), 128, 1, 1).to(self.device)

            fake_image = self.generator(z, alpha=self.alpha)

            d_fake_pred = self.discriminator(fake_image, alpha=self.alpha)
            g_loss = self.criterion(d_fake_pred, real_label)

            self.generator_optim.zero_grad()
            g_loss.backward()
            self.generator_optim.step()

            generator_avg_loss += g_loss.item()


            self.clear_cuda_memory()

        generator_avg_loss /= len(self.trainloader)
        discriminator_avg_loss /= len(self.trainloader)

        return generator_avg_loss, discriminator_avg_loss

    
    def valid(self):
        self.generator.eval()
        self.discriminator.eval()

        generator_avg_loss = 0
        discriminator_avg_loss = 0

        for _ in range(len(self.validloader)):
            real_image = next(iter(self.validloader)).to(self.device)

            real_label = torch.full((real_image.size(0), 1), 1).type(torch.float).to(self.device)
            fake_label = torch.full((real_image.size(0), 1), 0).type(torch.float).to(self.device)

            # ----------------------------------------------------- discriminator valid ----------------------------------------------------------------

            z = torch.randn((real_image.size(0), 128, 1, 1)).to(self.device)
            fake_image = self.generator(z, alpha=self.alpha)

            d_fake_pred = self.discriminator(fake_image.detach(), alpha=self.alpha)
            d_fake_loss = self.criterion(d_fake_pred, fake_label)

            d_real_pred = self.discriminator(real_image, alpha=self.alpha)
            d_real_loss = self.criterion(d_real_pred, real_label)

            discriminator_avg_loss += ((d_fake_loss + d_real_loss).item() / 2)

            # ------------------------------------------------------ generator valid --------------------------------------------------------------------

            z = torch.randn((real_image.size(0), 128, 1, 1)).to(self.device)
            fake_image = self.generator(z, alpha=self.alpha)

            d_fake_pred = self.discriminator(fake_image.detach(), alpha=self.alpha)
            g_loss = self.criterion(d_fake_pred, real_label)

            generator_avg_loss += g_loss.item()

            self.clear_cuda_memory()

        generator_avg_loss /= len(self.validloader)
        discriminator_avg_loss /= len(self.validloader)

        return generator_avg_loss, discriminator_avg_loss


    def run(self, epochs):
        train_history = []
        valid_history = []

        self.alpha = 0
        self.alpha_gap = 1 / (len(self.trainloader) * (epochs[1] - epochs[0]))

        for epoch in range(*epochs):
            print("-"*100 + "\n" + f"Epoch: {epoch}")

            train_history.append(self.train())
            print(f"\tTrain\n\t\tG Loss: {train_history[-1][0]},\tD Loss: {train_history[-1][1]}")

            valid_history.append(self.valid())
            print(f"\tValid\n\t\tG Loss: {valid_history[-1][0]}, \t D Loss: {valid_history[-1][1]}")

            self.test(epoch)
    
        return train_history, valid_history

    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for steps in range(7):
        trainer = Trainer(steps=steps, batch_size=16, device=device, test_size=16)
        train_history, valid_history = trainer.run((0, 30))
        trainer.save_model()
