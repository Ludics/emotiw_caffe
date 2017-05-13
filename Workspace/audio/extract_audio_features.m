function features = extract_audio_features(audio)
    if audio.rate == 1
        audio.rate = 44100;  % Cannot be determined, set it to 44100.
    end

    data = audio.data(:, 1);
    len = length(data);
    % Energy.
    %energy = sum(data.^2) / len;

    % % Energy entropy.
    % non_zero_energies = (data).^2;
    % non_zero_energies = non_zero_energies(non_zero_energies > 1e-6);
    % entropy = -sum(non_zero_energies .* log2(non_zero_energies));

    % ZCR.
    %pos = data > 0;
    %neg = data < 0;
    %zcr = sum(pos(1:end-1) & neg(2:end) | ...
              %neg(1:end-1) & pos(2:end)) / (len - 1);

    % Spectrum.
    %window_size = round(audio.rate * 0.02);  % Set window size to 20ms.
    % Overlap by 50%, limit rows to 17.
    %spectrum = abs(spectrogram(data, window_size, round(window_size / 2), 8));
    %spectrum_var = var(spectrum, 0, 2);
    %spectrum_max = max(spectrum, [], 2);
    % MFCC.
    m=mymfcc(audio)';
    dtm = zeros(size(m));
    for i=3:size(m,1)-2
        dtm(i,:) = -2*m(i-2,:) - m(i-1,:) + m(i+1,:) + 2*m(i+2,:);
    end
    dtm = dtm / 3;
    %ddtm=zeros(size(m));
    %for i=3:size(m,1)-2
       % ddtm(i,:) = -2*dtm(i-2,:) - dtm(i-1,:) + dtm(i+1,:) + 2*dtm(i+2,:);
    %end
    %ddtm = ddtm / 3;
    %合并mfcc参数和一阶差分mfcc参数
    ccc = [m dtm];
    %去除首尾两帧，因为这两帧的一阶差分参数为0
    ccc = ccc(3:size(m,1)-2,:);
    %energy=zeros(size(ccc,1),1);
    %for i=1:size(ccc,1)
        %sum=0;
        %for k=1:13
         %   sum=sum+ccc(i,k)^2;
        %end
        %energy(i,1)=10*log10(sum);
    %end
    %ccc=[ccc energy]';
    %z=wcmvn(ccc,301,true)';
    %feature_avg=mean(z);
    %feature_var=var(z,0,1);
    %mfcc_var = var(mfccs, 0, 2);
    %mfcc_max = max(mfccs, [], 2);
    %mfcc_max_over_avg = mfcc_max ./ mean(mfccs, 2);
    %mfcc_median = median(mfccs, 2);

    %features = [energy * 100 zcr * 100 spectrum_var spectrum_max mfcc_var mfcc_max mfcc_max_over_avg mfcc_median];
    
    features=mean(ccc);

