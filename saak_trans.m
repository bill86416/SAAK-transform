function [trans_images,augmented_kernels]=saak_trans(images,slicing_size,pic_num,mul)
    if nargin==4
        len=size(images,1)/slicing_size;
        total_patch=len.^2;
        trans_images=zeros(len,len,(pic_num*mul*(slicing_size.^2)*2));
        augmented_kernels=zeros((slicing_size.^2),((slicing_size.^2)*2),total_patch);
        
        for ii=1:len
            for jj=1:len
                im=images(((ii-1)*slicing_size+1):(ii*slicing_size),((jj-1)*slicing_size+1):(jj*slicing_size),:);
                im=reshape(im,(slicing_size.^2),pic_num*mul);
                %zero_mean
                aver=mean(im,1);
              
                aver_m=repmat(aver,(slicing_size.^2),1);
                z_im=im-aver_m;
                %KLT
                R=z_im*(z_im');
                [V,~]=eig(R);
                neg_V=V*(-1);
                kernel=zeros((slicing_size.^2),((slicing_size.^2)*2));
                kernel(:,1:2:(((slicing_size.^2)*2)-1))=V(:,1:1:(slicing_size.^2));
                kernel(:,2:2:((slicing_size.^2)*2))=neg_V(:,1:1:(slicing_size.^2));
                trans_im=kernel'*im;
                augmented_kernels(:,:,((ii-1)*len+jj))=kernel(:,:);
                trans_images(ii,jj,:)=reshape(trans_im,1,1,[]);
                %RELU
                trans_images=(trans_images>0).*trans_images;
            end
        end
    end
end
