% =========================================================================
% Test code for Super-Resolution Convolutional Neural Networks (SRCNN)
%
% Reference
%   Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. Learning a Deep Convolutional Network for Image Super-Resolution, 
%   in Proceedings of European Conference on Computer Vision (ECCV), 2014
%
%   Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. Image Super-Resolution Using Deep Convolutional Networks,
%   arXiv:1501.00092
%
% Chao Dong
% IE Department, The Chinese University of Hong Kong
% For any question, send email to ndc.forward@gmail.com
% =========================================================================

close all;
clear all;

%% set parameters
% up_scale = 3;
% model = 'model\9-5-5(ImageNet)\x3.mat';
% up_scale = 3;
% model = 'model\9-3-5(ImageNet)\x3.mat';
% up_scale = 3;
% model = 'model\9-1-5(91 images)\x3.mat';
up_scale = 2;
model = 'model\9-5-5(ImageNet)\x2.mat'; 
% up_scale = 4;
% model = 'model\9-5-5(ImageNet)\x4.mat';

img_path = '../data/example_with_gt';
save_path = '../results/zssr_one_jump';

img_dir = dir(fullfile(img_path, '*.png'));
for ii = 1:length(img_dir),
    if ~isempty(strfind(img_dir(ii).name, 'bridge.png')),
        im_l = imread(sprintf('%s/%s', img_path, img_dir(ii).name));
        if size(im_l,3)>1,
            im_l = rgb2ycbcr(im_l);
        end
        im_l = single(im_l)/255;
        im_b = imresize(im_l, up_scale, 'bicubic');
        im_h = imresize(im_l, up_scale, 'bicubic');
        im_h_y = SRCNN(model, im_b(:,:,1));
        im_h(:,:,1) = im_h_y;
        im_b = uint8(im_b * 255);
        im_h = uint8(im_h * 255);
        if size(im_l,3)>1,
            im_h = ycbcr2rgb(im_h);
            im_b = ycbcr2rgb(im_b);
        end
        imwrite(im_b, sprintf('%s/%s_bicubic_X2.00X2.00.png', save_path, img_dir(ii).name(1:end-4)))
        imwrite(im_h, sprintf('%s/%s_srcnn_X2.00X2.00.png', save_path, img_dir(ii).name(1:end-4)))
    end
end