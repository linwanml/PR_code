for index_rslt = 1 : length(rslt)
    fnum_opt = rslt(index_rslt).fnum_opt;
    w_feat = rslt(index_rslt).w_feat;
    w_final = w_feat(:, 1:fnum_opt); 
    rslt(index_rslt).w_final =  w_final;
    
    
  %test  
%     test_error(i) = prcrossval(a_optfeat, clsf, 10, 2);
 
    %[trn_pca tst_pca] = gendat(a_pca, 0.8);
    %trn_optfeat = trn_pca * w_feat;
    %tst_optfeat = tst_pca * w_feat;
    a_optfeat = a * w_pca * w_final;
    clsf = rslt(index_rslt).clsf{:};
    %test_error(index_rslt) = tst_optfeat * clsf(trn_optfeat) * testc;
%     test_error(i) = nist_eval('my_rep_smalldataset', w_pca * w_final * clsf(a_optfeat), 10);
    test_error_nist(index_rslt) = nist_eval('my_rep', w_pca * w_final * clsf(a_optfeat), 50);
    
    
end




