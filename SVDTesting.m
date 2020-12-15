clear all;
close all;
clc;

%% Using SVD to investigate Data
images = loadMNISTImages('C:\Users\PhD\Documents\MATLAB\Exploring-AXB-using-the-MNIST-Database\MNIST-Dataset\train-images.idx3-ubyte');
[U,S,V] = svd(images,0);

figure()
semilogy(100*diag(S)/sum(diag(S)),'b*','linewidth',[2]);
title('Variance of the SVD (logarithmic Y)')
hold on; xlabel('singular values of the images matrix'); ylabel('relative importance [%]');
% Note there is a sharp drop off around X=650. We do not see a strong
% correlation with a singular value. The most dominant value contains only
% 6% of trhe relative importance. By the 13th singular value, the relative
% importance has dropped to be less than one per pixel. For 80% coverage,
% we need the first 232 Singular Values.
%%
pCapture = zeros(1,3); %percent covered by the Singular values
iter=ones(1,3);
Svec = diag(S);
testPs = [80,90,99];
for jter=1:length(testPs)
    while pCapture(jter) <testPs(jter);
        pCapture(jter) =100*sum(Svec(1:iter(jter)))/sum(Svec);
        iter(jter)=iter(jter)+1
    end
end