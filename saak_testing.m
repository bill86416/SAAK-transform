function [test_trans_images]=saak_testing(images,slicing_size,pic_num,mul,augmented_kernels)
    if nargin==5
        len=size(images,1)/slicing_size;
        test_trans_images=zeros(len,len,(pic_num*mul*(slicing_size.^2)*2));  
        
        for ii=1:len
            for jj=1:len
                im=images(((ii-1)*slicing_size+1):(ii*slicing_size),((jj-1)*slicing_size+1):(jj*slicing_size),:);
                im=reshape(im,(slicing_size.^2),pic_num*mul);
                tran_im=augmented_kernels(:,:,((ii-1)*len+jj))'*im;
                test_trans_images(ii,jj,:)=reshape(tran_im,1,1,[]);
            end
        end
        %RELU
        test_trans_images=(test_trans_images>0).*test_trans_images;
    end    
end