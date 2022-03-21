import os
import numpy as np
import torch
import torch.nn.functional as F
import kornia
import cv2
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import  Model
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import  PSNR_RGB
# from torchsummary import summary
# from ptflops import get_model_complexity_info



class D4():
    def __init__(self, config):
        self.config = config

        self.model = Model(config).to(config.DEVICE)


        if config.PSNR == 'RGB':
            self.psnr = PSNR_RGB(255.0).to(config.DEVICE)


        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, crop_size=None, hazy_flist=config.TEST_HAZY_FLIST, clean_flist=config.TEST_CLEAN_FLIST, clean_path=config.TEST_CLEAN_PATH, augment=False,
                                        split=self.config.TEST_MODE)

        else:
            self.train_dataset = Dataset(config, crop_size=config.CROP_SIZE, clean_flist=config.TRAIN_CLEAN_FLIST, hazy_flist=config.TRAIN_HAZY_FLIST,  augment=True, split='unpair')
            self.val_dataset = Dataset(config, crop_size=None, hazy_flist=config.VAL_HAZY_FLIST, clean_path=config.VAL_CLEAN_PATH, augment=False, split='pair_test')

            self.test_dataset = Dataset(config, crop_size=None, hazy_flist=config.TEST_HAZY_FLIST, augment=False, split=self.config.TEST_MODE)
            self.sample_dataset = Dataset(config, crop_size=config.CROP_SIZE, clean_flist=config.TRAIN_CLEAN_FLIST, hazy_flist=config.TRAIN_HAZY_FLIST, augment=False, split='unpair')

            self.sample_iterator = self.sample_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + self.model.name + '.dat')

    def load(self):
        self.model.load()


    def save(self, save_best=False, psnr=None, iteration=None):
        self.model.save(save_best,psnr,iteration)



    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size= self.config.BATCH_SIZE,
            num_workers=0,
            drop_last=True,
            shuffle=False
        )


        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)
        epoch = self.model.epoch
        highest_psrn = 0
        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])
            print('epoch:', epoch)

            index = 0

            for items in train_loader:
                self.model.train()
                #self.model.eval_()
                clean_images, hazy_images= self.cuda(*items)


                if model == 1:

                    outputs, gen_loss, dis_loss, logs = self.model.process(clean_images, hazy_images)

                    psnr = self.psnr(self.postprocess(clean_images), self.postprocess(outputs))
                    mae = torch.mean((torch.abs(clean_images - outputs)))
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))


                    iteration = self.model.iteration



                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                index += 1
                progbar.add(len(clean_images), values=logs if self.config.VERBOSE else [x for x in logs ])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    self.sample()

                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    current_psnr = self.eval()
                    print('\naccuracy:', current_psnr)


                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

            # update epoch for scheduler
            self.model.epoch = epoch
            self.model.update_scheduler()

        print('\nEnd training....')

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            drop_last=False,
            shuffle=False
        )


        model = self.config.MODEL
        total = len(self.val_dataset)

        self.model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        psnrs=[]
        with torch.no_grad():
            for items in val_loader:
                iteration += 1
                clean_images, noisy_images = self.cuda(*items)

                if model == 1 and self.val_dataset.split == 'pair_test':
                    h, w = noisy_images.shape[2:4]


                    noisy_images_input = self.pad_input(noisy_images)
                    clean_images_h2c,_ = self.model.forward_h2c(noisy_images_input)
                    predicted_results = self.crop_result(clean_images_h2c, h, w)


                    psnr = self.psnr(self.postprocess(clean_images), self.postprocess(predicted_results))


                    psnrs.append(psnr.item())
                    logs = []
                    logs.append(('psnr_rgb', psnr.item()))



                logs = [("it", iteration), ] + logs
                progbar.add(len(noisy_images), values=logs)

        return np.mean(psnrs)

    def test(self):
        model = self.config.MODEL
        self.model.eval()
        create_dir(self.results_path)
        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0

        if self.config.DATASET == 'SOTS-indoor':
            use_guided_filter = False
        else:
            use_guided_filter = True

        psnrs = []

        # #################### summary ########################3
        # #summary(self.model.net_h2c, input_size=(3,256,256))
        # inp = torch.randn(1,3,256,256).cuda()
        # macs, params = get_model_complexity_info(self.model.net_h2c,(3,256,256), as_strings=True,print_per_layer_stat=True,verbose=True)
        # print('{:<30} {:<8}'.format('Flops:', macs))
        # print('params:'+str(params))
        # #print(############)
        times = []
        
        with torch.no_grad():
            for items in test_loader:
                if self.test_dataset.split =='pair_test':

                    name = self.test_dataset.load_name(index)[:-4]+'.png'

                    clean_images, hazy_images = self.cuda(*items)
                    index += 1

                    if model == 1:
                        ## check if the input size is multiple of 4
                        h, w = hazy_images.shape[2:4]



                        hazy_input_images = self.pad_input(hazy_images)

                        predicted_results= self.model.forward_h2c(hazy_input_images, use_guided_filter=use_guided_filter)

                        predicted_results = self.crop_result(predicted_results, h, w)

                        path = os.path.join(self.results_path, self.model.name)
                        create_dir(path)
                        save_name = os.path.join(path, name)
                        predicted_results = self.postprocess(predicted_results)[0]
                        predicted_results.save(save_name)


                        psnr = self.psnr(self.postprocess(predicted_results), self.postprocess(clean_images))
                        psnrs.append(psnr.item())
                        print('PSNR_RGB:', psnr)



                elif self.test_dataset.split == 'clean':

                    name = self.test_dataset.load_name(index)[:-4] + '.png'

                    clean_images = items.to(self.config.DEVICE)
                    index += 1



                    if model == 1:

                        ## check if the input size is multiple of 4
                        h, w = clean_images.shape[2:4]

                        if h * w > 2000 * 3000:
                            continue

                        clean_input_images = self.pad_input(clean_images)
                        predicted_depth = self.model.forward_depth(clean_input_images)
                        if use_guided_filter:
                            predicted_depth = self.model.net_c2h.transmission_estimator.get_refined_transmission(clean_input_images, predicted_depth)

                        for beta_in in [0.6,1.2,1.8]:

                            predicted_results = self.model.forward_c2h_given_parameters(clean_input_images, predicted_depth, beta_in)
                            predicted_results = self.crop_result(predicted_results, h, w)


                            predicted_results = self.postprocess(predicted_results)[0]


                            path = os.path.join(self.results_path, self.model.name + '_haze')
                            create_dir(path)
                            name_s = name[:-4]+'_'+str(beta_in)+'.png'
                            save_name = os.path.join(path, name_s)
                            imsave(predicted_results, save_name)
                            print(index, save_name)


                        predicted_depth = self.crop_result(predicted_depth, h, w)
                        predicted_depth= self.minmax_depth(predicted_depth)
                        predicted_depth = self.generate_color_map(predicted_depth, [h, w])

                        predicted_depth = predicted_depth[0]
                        path = os.path.join(self.results_path, self.model.name + '_depth')
                        create_dir(path)
                        name_s = name[:-4] + '_depth' + '.png'
                        save_name = os.path.join(path, name_s)
                        imsave(predicted_depth, save_name)
                        print(index, save_name)


            print('AVG times:'+str(np.mean(times)) )
            print('Total PSNR_', np.mean(psnrs))
            print('\nEnd test....')

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.sample_dataset) == 0:
            return
        self.model.eval()

        model = self.config.MODEL

        items = next(self.sample_iterator)
        clean_images, hazy_images= self.cuda(*items)



        # inpaint with edge model / joint model
        with torch.no_grad():
            iteration = self.model.iteration

            if model == 1:
                h, w = hazy_images.shape[2:4]
                ## check if the input size is multiple of 4
                clean_images_h2c, pred_ex_hazy, pred_beta_hazy = self.model.forward_h2c(hazy_images, require_paras=True)

                hazy_images_h2c2h = self.model.forward_c2h_given_parameters(clean_images_h2c, pred_ex_hazy, pred_beta_hazy)
                pred_ex_hazy_bydepth = self.model.forward_depth(clean_images_h2c)
                pred_ex_clean = self.model.forward_depth(clean_images)
                hazy_images_c2h = self.model.forward_c2h_random_parameters(
                    clean_images, pred_ex_clean)
                clean_images_c2h2c,t = self.model.forward_h2c(hazy_images_c2h)

                pred_ex_hazy = self.minmax_depth(pred_ex_hazy)
                pred_ex_clean = self.minmax_depth(pred_ex_clean)
                pred_ex_hazy_bydepth = self.minmax_depth(pred_ex_hazy_bydepth)

                pred_t = pred_ex_hazy



                images_sample = stitch_images(
                    self.postprocess(clean_images),
                    self.postprocess(clean_images_c2h2c),
                    self.postprocess(hazy_images_c2h),
                    self.generate_color_map(pred_ex_clean),
                    self.postprocess(hazy_images),
                    self.postprocess(hazy_images_h2c2h),
                    self.postprocess(clean_images_h2c),
                    self.generate_color_map(pred_ex_hazy_bydepth),
                    self.generate_color_map(pred_ex_hazy),
                    self.postprocess(pred_t),
                    img_per_row=1
                )


            path = os.path.join(self.samples_path, self.model.name)
            name = os.path.join(path, str(iteration).zfill(5) + ".png")
            create_dir(path)
            print('\nsaving sample ' + name)
            images_sample.save(name)


    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img, size=None):
        # [0, 1] => [0, 255]
        if size is not None:
            img = torch.nn.functional.interpolate(img,size,mode='bicubic')
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def generate_color_map(self, imgs, size=[256,256]):
        # N 1 H W -> N H W 3 color map
        imgs = (imgs*255.0).int().squeeze(1).cpu().numpy().astype(np.uint8)
        N, height,width = imgs.shape

        colormaps = np.full((N,size[0],size[1],3),1)

        for i in range(imgs.shape[0]):
            colormaps[i] = cv2.resize((cv2.applyColorMap(imgs[i], cv2.COLORMAP_HOT)),(size[1],size[0]))

        #transfer to tensor than to gpu
        #firstly the channel BGR->RGB
        colormaps = colormaps[...,[2,1,0]]

        #than to tensor, to gpu
        colormaps = torch.from_numpy(colormaps).cuda()

        return colormaps

    def crop_result(self, result, input_h, input_w, times=32):
        crop_h = crop_w = 0

        if input_h % times != 0:
            crop_h = times - (input_h % times)

        if input_w % times != 0:
            crop_w = times - (input_w % times)

        if crop_h != 0:
            result = result[...,:-crop_h, :]
        if crop_w != 0:
            result = result[...,:-crop_w]
        return result

    def pad_input(self, input, times=32):
        input_h, input_w = input.shape[2:]
        pad_h = pad_w = 0

        if input_h % times != 0:
            pad_h = times - (input_h % times)

        if input_w % times != 0:
            pad_w = times - (input_w % times)

        #print(pad_h, pad_w)

        input = torch.nn.functional.pad(input, (0,pad_w, 0, pad_h), mode='reflect')

        return input

    def minmax_depth(self, depth, blur=True):
        n, c, h, w = depth.shape
        # depth = F.avg_pool2d(depth,kernel_size=5)

        if blur:  # use median filter to filter out peak values for visualization.
            depth = F.pad(depth,[4,4,4,4],'reflect')
            depth = kornia.median_blur(depth,(9,9))
            depth = depth[:,:,3:h-3,3:w-3]

        D_max = torch.max(depth.reshape(n, c, -1), dim=2, keepdim=True)[0].unsqueeze(3)
        D_min = torch.min(depth.reshape(n, c, -1), dim=2, keepdim=True)[0].unsqueeze(3)
        depth = (depth - D_min) / (D_max - D_min)
        return depth

