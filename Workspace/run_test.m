close all;
clear all;
clc;

test_file   =   '/home/leon/repertory/caffe/EmotiW2016/faces/000029301';
test_avi    =   [test_file, '.avi'];
test_rst    =   'Neutral';

try
    test_ret    =   fun_process(test_avi);
catch   err
    test_ret    =   'None';
    warning('fun_process failed, no return value available.');
end

if (strcmp(test_rst, test_ret))
    disp(['Correct! The answer is ', test_rst, ' and the function output is ', test_ret, '.']);
else
    disp(['Wrong! The answer is ', test_rst, ' while the function output is ', test_ret, '.']);
end

% ------------------------------------------------------------------------
