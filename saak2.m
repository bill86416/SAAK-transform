clear all;
%load
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
test_images = loadMNISTImages('t10k-images.idx3-ubyte');
test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
%zero padding
images32=zeros(32,32,30000);
images28=reshape(images(:,1:30000),28,28,30000);
images32(3:30,3:30,:)=images28(:,:,:);
clear images;
clear images28;
%test zero padding
test_images32=zeros(32,32,10000);
test_images28=reshape(test_images(:,1:10000),28,28,10000);
test_images32(3:30,3:30,:)=test_images28(:,:,:);
clear test_images;
clear test_images28;
%Stage 1
[trans_images16,aug_ker,bias]=saak_trans(images32,2,30000,1);
[test_trans_images16]=saak_testing(test_images32,2,10000,aug_ker,1);
b=min(min(min(min(trans_images16))),min(min(min(test_trans_images16))));
trans_images16=trans_images16+b+0.001;
test_trans_images16=test_trans_images16+b+0.001;
clear images32;
clear test_images32;
clear aug_ker;
%Stage 2
[trans_images8,aug_ker,bias]=saak_trans(trans_images16,2,30000,4);
[test_trans_images8]=saak_testing(test_trans_images16,2,10000,aug_ker,4);
b=min(min(min(min(trans_images8))),min(min(min(test_trans_images8))));
trans_images8=trans_images8+b+0.001;
test_trans_images8=test_trans_images8+b+0.001;
clear trans_images16;
clear test_trans_images16;
clear aug_ker;
%Stage 3
[trans_images4,aug_ker,bias]=saak_trans(trans_images8,2,30000,16);
[test_trans_images4]=saak_testing(test_trans_images8,2,10000,aug_ker,16);
b=min(min(min(min(trans_images4))),min(min(min(test_trans_images4))));
trans_images4=trans_images4+b+0.001;
test_trans_images4=test_trans_images4+b+0.001;
clear trans_images8;
clear test_trans_images8;
clear aug_ker;
%Stage 4
[trans_images2,aug_ker,bias]=saak_trans(trans_images4,2,30000,64);
[test_trans_images2]=saak_testing(test_trans_images4,2,10000,aug_ker,64);
b=min(min(min(min(trans_images2))),min(min(min(test_trans_images2))));
trans_images2=trans_images2+b+0.001;
test_trans_images2=test_trans_images2+b+0.001;
clear trans_images4;
clear test_trans_images4;
clear aug_ker;
%Stage 5
[trans_images1,aug_ker,bias]=saak_trans(trans_images2,2,30000,256);
[test_trans_images1]=saak_testing(test_trans_images2,2,10000,aug_ker,256);

clear trans_images2;
clear test_trans_images2;
clear aug_ker;
trans_images1=reshape(trans_images1,[],30000);

%%
test_trans_images1=reshape(test_trans_images1,[],10000);

%F score
[Index_imp]=f_score(labels(1:30000,1),trans_images1);
trans_images=trans_images1(Index_imp(1,1:500),:);

[coeff,~,latant]=pca(trans_images');
test_trans_images=test_trans_images1(Index_imp(1,1:500),:);

clear trans_images1;
clear test_trans_images1;

%SVM train
a=coeff'*trans_images;
b=coeff'*test_trans_images;
%%
N=30000;
svm_model=fitcecoc(a(1:300,:)',labels(1:N,1));
result1=predict(svm_model,a(1:300,:)');
error1=length(find(result1~=labels(1:N,1)));
result_svm=predict(svm_model,b(1:300,:)');
error=length(find(result_svm~=test_labels))