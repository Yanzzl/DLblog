import time
from options.train_options import TrainOptions
from data.dataprocess import DataProcess
from models.models import create_model
import torchvision
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import calculate_PSNR_SSIM
import path_colab
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


if __name__ == "__main__":
    cuda_gpu = torch.cuda.is_available()
    print(cuda_gpu)

    # for validation
    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    mask_val = glob('{:s}/*'.format(path_colab.val_path_MASK))
    de_val = glob('{:s}/*'.format(path_colab.val_path_DE))
    # st_path = glob('{:s}/*'.format(opt.st_root))
    image_len = len(de_val)

    PSNR_all = []
    SSIM_all = []
    epoch_all = []

    # train
    opt = TrainOptions().parse()
    # define the dataset
    # TODO 改mask 让他只有一种
    # dataset = DataProcess(opt.de_root,opt.st_root,opt.mask_root,opt,opt.isTrain)
    dataset = DataProcess(path_colab.train_path_DE, path_colab.train_path_ST, path_colab.train_path_MASK, opt, opt.isTrain)
    iterator_train = (data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.num_workers))
    # Create model
    model = create_model(opt)
    total_steps=0
    # Create the logs
    dir = os.path.join(opt.log_dir, opt.name).replace('\\', '/')
    if not os.path.exists(dir):
        os.mkdir(dir)
    writer = SummaryWriter(log_dir=dir, comment=opt.name)
    # Start Training
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        for detail, structure, mask in iterator_train:
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(detail, structure, mask)
            model.optimize_parameters()
            # display the training processing
            if total_steps % opt.display_freq == 0:
                input, output, GT = model.get_current_visuals()
                image_out = torch.cat([input, output, GT], 0)
                grid = torchvision.utils.make_grid(image_out)
                writer.add_image('Epoch_(%d)_(%d)' % (epoch, total_steps + 1), grid, total_steps + 1)
            # display the training loss
            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                writer.add_scalar('G_GAN', errors['G_GAN'], total_steps + 1)
                writer.add_scalar('G_L1', errors['G_L1'], total_steps + 1)
                writer.add_scalar('G_stde', errors['G_stde'], total_steps + 1)
                writer.add_scalar('D_loss', errors['D'], total_steps + 1)
                writer.add_scalar('F_loss', errors['F'], total_steps + 1)
                print('iteration time: %d' % t)
        # if epoch % opt.save_epoch_freq == 0:
        if epoch % 3 == 0: # check converge
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks(epoch)
            # TODO run validation set
            for i in tqdm(range(image_len)):
                # only use one mask for all image
                path_m = mask_val[0]
                path_d = de_val[i]
                name = path_d[-9:]
                path_s = path_colab.val_path_ST + name

                # print("path_m" + path_m)
                # print("path_d" + path_d)
                # print("path_s" + path_s)

                mask = Image.open(path_m).convert("RGB")
                detail = Image.open(path_d).convert("RGB")
                structure = Image.open(path_s).convert("RGB")

                mask = mask_transform(mask)
                detail = img_transform(detail)
                structure = img_transform(structure)
                mask = torch.unsqueeze(mask, 0)
                detail = torch.unsqueeze(detail, 0)
                structure = torch.unsqueeze(structure, 0)

                with torch.no_grad():
                    model.set_input(detail, structure, mask)
                    model.forward()
                    fake_out = model.fake_out
                    fake_out = fake_out.detach().cpu() * mask + detail * (1 - mask)
                    fake_image = (fake_out + 1) / 2.0
                output = fake_image.detach().numpy()[0].transpose((1, 2, 0)) * 255
                output = Image.fromarray(output.astype(np.uint8))
                output.save(rf"{path_colab.val_result_dir}/{name}")
            # TODO calculate and plot
            PSNR_avg, SSIM_avg = calculate_PSNR_SSIM.calculate(path_colab.val_path_DE, path_colab.val_result_dir)
            PSNR_all.append(PSNR_avg)
            SSIM_all.append(SSIM_avg)
            epoch_all.append(epoch)
            # plt.plot(epoch_all, PSNR_all)
            # plt.title('PSNR Vs Epoch')
            # plt.xlabel('Epoch')
            # plt.ylabel('PSNR')
            # plt.savefig(rf"{path_colab.plot_dir}/PSNR{epoch}.png")
            # plt.show()
            # plt.plot(epoch_all, SSIM_all)
            # plt.title('SSIM Vs Epoch')
            # plt.xlabel('Epoch')
            # plt.ylabel('SSIM')
            # plt.savefig(rf"{path_colab.plot_dir}/SSIM{epoch}.png")
            # plt.show()


            print(epoch_all)
            print(PSNR_all)
            print(SSIM_all)
            with open('data.txt', 'a+') as ww:
                ww.writelines(str(epoch_all))
                ww.writelines(str(PSNR_all))
                ww.writelines(str(SSIM_all))
                ww.close()


        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
    writer.close()

