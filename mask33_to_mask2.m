% matlab script to convert 33 masks to bin masks
clc;clear all;close all;

path = '/home/gaurav/Desktop/zVarun/training/imnames.txt';
fid = fopen(path);
tline = fgetl(fid);
while ischar(tline)
    img = imread(strcat('/home/gaurav/Desktop/zVarun/training/mask/',tline));
    s= size(img);
    imgr = zeros(s(1),s(2));
    for i = 1: s(1)
        for j= 1:s(2)

            if(img(i,j)==1)
                imgr(i,j) ==0;
            elseif (img(i,j)==0)
                imgr(i,j) ==0;
            else
                imgr(i,j)=1;
            end
        end
    end
    imwrite(imgr, strcat('/home/gaurav/Desktop/zVarun/training/binary_grnd/',tline))
    tline = fgetl(fid);
    disp(tline)
end

fclose(fid);

