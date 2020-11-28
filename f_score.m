function [index]=f_score(labels,coeff)
    num=size(coeff,1);
    F_score=zeros(1,num);
    BGV=0;
    WGV=0;
    S=mean(coeff,2);
    for ii=1:num
        for jj=1:10
            mark=find(labels==(jj-1));
            count=coeff(ii,mark);
            Sc=mean(count,2);
            BGV=BGV+((size(mark,1)*sum(((Sc-S(ii,1)).^2)))/9);
            wgv_count=sum((count-Sc).^2);
            WGV=WGV+(wgv_count/(size(coeff,1)-10));
        end
        F_score(1,ii)=BGV/WGV;
        BGV=0;
        WGV=0;
    end
    [~, ind] = sort(F_score,'descend');
    index=ind;
end