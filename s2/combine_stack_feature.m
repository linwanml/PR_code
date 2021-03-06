%load dataset
nist = prnist([0:9],[1:100:1000]);
a = my_rep(nist);
feat = im_features(a,a,'all');
%pca mapping

%%
%final mapping after feature selection 
%ldc
[w_feat_p, r_feat_p] = featselm(feat,'eucl-s', 'backward', 21);
w_final_p = w_feat_p(:, 1:21);
a_optfeat_p = feat * w_final_p; 
clsf = ldc;
w_ldc_1 = w_final_p * clsf(a_optfeat_p)
test_error_nist_ldc1 = nist_eval('my_rep_1',w_ldc_1,50);

%%
%final mapping after feature selection
%ldc
[w_feat_k, r_feat_k] = featselm(feat,'eucl-s', 'ind', 18);
w_final_k = w_feat_k(:, 1:18);
a_optfeat_k = feat * w_final_k; 
clsf = ldc;
w_ldc_2 = w_final_k * clsf(a_optfeat_k)
test_error_nist_ldc2 = nist_eval('my_rep_1', w_ldc_2, 50);

%%
%final mapping after feature selection
%combine stack combine only the classifier
w_combine = [w_ldc_1, w_ldc_2];
w_c_max = w_combine * maxc;
test_error_nist_combine_max = nist_eval('my_rep_1', w_c_max, 50);
w_c_min = w_combine * minc;
test_error_nist_combine_min = nist_eval('my_rep_1', w_c_min, 50);
w_c_mean = w_combine * meanc;
test_error_nist_combine_mean = nist_eval('my_rep_1', w_c_mean, 50);
w_c_prod = w_combine * prodc;
test_error_nist_combine_prod = nist_eval('my_rep_1', w_c_prod, 50);


%%
%final mapping after feature selection
%combine parallel combine both the classifiers and features
%features:pixel and feature, so we dont use this



