import sys
import os
import configs
import ZSSR
import matplotlib.image as img
from skimage.measure import compare_psnr, compare_ssim 

def main(input_img, ground_truth, kernels, gpu, conf, results_path):
    # Choose the wanted GPU
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = '%s' % gpu

    # 0 input for ground-truth or kernels means None
    ground_truth = None if ground_truth == '0' else ground_truth
    print('*****', kernels)
    kernels = None if kernels == '0' else kernels.split(';')[:-1]

    # Setup configuration and results directory
    # conf = configs.Config()
    # if conf_str is not None:
    #     exec ('conf = configs.%s' % conf_str)
    conf.result_path = results_path

    # Run ZSSR on the image
    net = ZSSR.ZSSR(input_img, conf, ground_truth, kernels)
    output_img = net.run()
    # if ground_truth:
    #     with open('./results/result.txt', 'a+') as out:
    #         gnd_img = ground_truth if type(ground_truth) is not str else img.imread(ground_truth)
    #         psnr = compare_psnr(gnd_img, output_img)
    #         ssim = compare_ssim(gnd_img, output_img, data_range=output_img.max()-output_img.min(), multichannel=True)
    #         out.write('{}: PSNR:{}, SSIM:{}\n'.format(input_img, psnr, ssim))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
