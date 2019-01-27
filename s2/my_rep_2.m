function [c] = my_rep_2(m)
    
imgSize = 32;
preproc = im_box([],0,1)*im_resize([],[imgSize-2 imgSize-2])*im_box([],1,0);
m_processed = m * preproc;
a = prdataset(m_processed);
b = im_features(a,a,'all');
c = [a b];
end

