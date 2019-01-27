%load dataset
nist = prnist([0:9],[1:100:1000]);
a = my_rep(nist);
feat = im_features(a,a,'all');
%pca mapping
w_pca = pcam(a, 0.85);
a_pca = a * w_pca;
featnum = [1 : size(a_pca, 2)];
mf = max(featnum);
%%
%final mapping after feature selection 
%ldc pixel
[w_feat_p, r_feat_p] = featselm(a_pca,'maha-s', 'backward', 24);
w_final_p = w_feat_p(:, 1:24);
a_optfeat_p = a * w_pca * w_final_p; 
clsf = ldc;
w_ldc_1 = w_pca * w_final_p * clsf(a_optfeat_p)
test_error_nist_ldc1 = nist_eval('my_rep',w_ldc_1,50);

%%
%final mapping after feature selection
%ldc feature
[w_feat_k, r_feat_k] = featselm(feat,'eucl-s', 'ind', 18);
w_final_k = w_feat_k(:, 1:18);
a_optfeat_k = feat * w_final_k; 
clsf = ldc;
w_ldc_2 = w_final_k * clsf(a_optfeat_k)
test_error_nist_ldc2 = nist_eval('my_rep_1', w_ldc_2, 50);

%%
%final mapping after feature selection
%combine stack combine only the classifier
%w_combine = [w_ldc_1, w_ldc_2];
%w_c_max = w_combine * maxc;
%test_error_nist_combine_max = nist_eval('my_rep', w_c_max, 50);
%w_c_min = w_combine * minc;
%test_error_nist_combine_min = nist_eval('my_rep', w_c_min, 50);
%w_c_mean = w_combine * meanc;
%test_error_nist_combine_mean = nist_eval('my_rep', w_c_mean, 50);
%w_c_prod = w_combine * prodc;
%test_error_nist_combine_prod = nist_eval('my_rep', w_c_prod, 50);


%combine parallel combine both the classifiers and features
w_combine = [w_ldc_1; w_ldc_2];
w_c_max_p = w_combine * maxc;
test_error_nist_combine_max_p = nist_eval('my_rep_2', w_c_max_p,50);
w_c_min_p = w_combine * minc;
test_error_nist_combine_min_p = nist_eval('my_rep_2', w_c_min_p, 50);
w_c_mean_p = w_combine * meanc;
test_error_nist_combine_mean_p = nist_eval('my_rep_2',w_c_mean_p, 50);
w_c_prod_p = w_combine * prodc;
test_error_nist_combine_prod_p = nist_eval('my_rep_2', w_c_prod_p, 50);