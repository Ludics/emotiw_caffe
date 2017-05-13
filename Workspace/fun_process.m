function   ret  =   fun_process(filestr)

    [filePath, fileName, fileExt]   =   fileparts(filestr);
    disp(filestr);
    % load decoded audio and video frames from .mat file.
    ret_set =   {'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'};
    load(fullfile(filePath, [fileName, '.mat']));

    %%处理音频
    addpath('./audio');
    addpath('./audio/Emotion-from-speech-MFCC-master');
    load finalNetwork;
    feature=extract_audio_features(audio)';
    scores_audio=finalNetwork(feature);

    %%caffemodel处理图像
    %% caffe net set up
    %addpath('../caffe/matlab/');
    addpath('../Caffe/caffe-master/matlab/')
    %caffe.set_mode_cpu();
    caffe.set_mode_gpu();
    gpu_id = 0;  % we will use the first gpu
    caffe.set_device(gpu_id);
    % Initialize the network using BVLC CaffeNet for image classification
    % Weights (parameter) file needs to be downloaded from Model Zoo.
    model_dir = 'model/';
    net_model = [model_dir 'deploy.prototxt'];
    net_weights = [model_dir 'faces_train_iter_20000.caffemodel'];
    phase = 'test'; % run with phase test (so that dropout isn't applied)
    if ~exist(net_weights, 'file')
      error('May be there is something wrong with the Path');
    end

    % Initialize a network
    net = caffe.Net(net_model, net_weights, phase);

    % prepare_image
    mkdir(fullfile(filePath, fileName));
    for j=1:video.nrFramesTotal
        jpgname=[fullfile(filePath, fileName),'/',fileName,'_',num2str(j),'.jpg'];
        imwrite(video.frames(j).cdata,jpgname);
    end
    % 人脸检测
    system(['extract.sh ', fullfile(filePath, fileName)]);

    imageName = dir(fullfile(filePath, fileName));
    Number = length(imageName);
    scoreSum = zeros(7,1);
    times = 3;
    if (Number==2)
        % scores = rand(7,1);
        % scoreSum = scores / sum(scores) * times;
    else
        %image = zeros(128,128,Number);
        for m = 1:times
            imageStr = fullfile(filePath, fileName, imageName(randi([3,Number])).name);
            im_ = imread(imageStr);
            [a b c] = size(im_);
            im = zeros(a,b,c);
            im(:,:,1) = im_; im(:,:,2) = im_; im(:,:,3) = im_;
            tic;
            input_data = {prepare_image(im)};
            toc;
            % do forward pass to get scores
            % scores are now Channels x Num, where Channels == 1000
            tic;
            scores = net.forward(input_data);
            toc;
            scores = scores{1};
            scores = mean(scores,2);
            scoreSum = scores + scoreSum;
        end
        %[score, ret_vals] = max(scoreMat);
        % score = mean(score, 2);  % take average scores over 10 crops

    end
    [~, ret_val_image] = max(scoreSum);
    scores_image = scoreSum/times;
    [~, ret_val] = max(scores_audio + scores_image);
    ret = ret_set{ret_val};
end

% ------------------------------------------------------------------------
function crops_data = prepare_image(im)
% ------------------------------------------------------------------------
% caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat contains mean_data that
% is already in W x H x C with BGR channels
% d = load('../+caffe/imagenet/ilsvrc_2012_mean.mat');
    mean_data = caffe.io.read_mean('model/faces_mean.binaryproto');
    IMAGE_DIM = 256;
    CROPPED_DIM = 227;

    % Convert an image returned by Matlab's imread to im_data in caffe's data
    % format: W x H x C with BGR channels
    im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
    im_data = permute(im_data, [2, 1, 3]);  % flip width and height
    im_data = single(im_data);  % convert from uint8 to single
    im_data = imresize(im_data, [IMAGE_DIM IMAGE_DIM], 'bilinear');  % resize im_data
    im_data = im_data - mean_data;  % subtract mean_data (already in W x H x C, BGR)

    % oversample (4 corners, center, and their x-axis flips)
    crops_data = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single');
    indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
    n = 1;
    for i = indices
      for j = indices
        crops_data(:, :, :, n) = im_data(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :);
        crops_data(:, :, :, n+5) = crops_data(end:-1:1, :, :, n);
        n = n + 1;
      end
    end
    center = floor(indices(2) / 2) + 1;
    crops_data(:,:,:,5) = ...
      im_data(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:);
    crops_data(:,:,:,10) = crops_data(end:-1:1, :, :, 5);
end
