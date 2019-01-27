%load dataset
nist = prnist([0:9],[1:2:1000]);
a = my_rep(nist);

%pca mapping
w_pca = pcam(a, 0.85);
a_pca = a * w_pca;
featnum = [1 : size(a_pca, 2)];
mf = max(featnum);
%%
%final mapping after feature selection 
%parzen
[w_feat_p, r_feat_p] = featselm(a_pca,'eucl-s', 'forward', mf);
fnum_opt_p = 28;
w_final_p = w_feat_p(:, 1:fnum_opt_p);
a_optfeat_p = a * w_pca * w_final_p; 
clsf = parzenc;
w_parzen = w_pca * w_final_p * clsf(a_optfeat_p)
test_error_nist_parzen = nist_eval('my_rep',w_parzen,50);

%%
%final mapping after feature selection
%knn
[w_feat_k, r_feat_k] = featselm(a_pca,'eucl-s', 'forward', mf);
fnum_opt_k = 21;
w_final_k = w_feat_k(:, 1:fnum_opt_k);
a_optfeat_k = a * w_pca * w_final_k; 
clsf = knnc;
w_knn = w_pca * w_final_k * clsf(a_optfeat_k)
test_error_nist_knn = nist_eval('my_rep', w_knn, 50);

%%
%final mapping after feature selection
%combine stack combine only the classifier
w_combine = [w_parzen, w_knn];
w_c_max = w_combine * maxc;
test_error_nist_combine_max = nist_eval('my_rep', w_c_max, 50);
w_c_min = w_combine * minc;
test_error_nist_combine_min = nist_eval('my_rep', w_c_min, 50);
w_c_mean = w_combine * meanc;
test_error_nist_combine_mean = nist_eval('my_rep', w_c_mean, 50);
w_c_prod = w_combine * prodc;
test_error_nist_combine_prod = nist_eval('my_rep', w_c_prod, 50);


%%
%final mapping after feature selection
%combine parallel combine both the classifiers and features

w_combine = [w_parzen; w_knn];
w_c_max_p = w_combine * maxc;
test_error_nist_combine_max_p = nist_eval('my_rep_1', w_c_max_p, 50);
w_c_min_p = w_combine * minc;
test_error_nist_combine_min_p = nist_eval('my_rep_1', w_c_min_p, 50);
w_c_mean_p = w_combine * meanc;
test_error_nist_combine_mean_p = nist_eval('my_rep_1', w_c_mean_p, 50);
w_c_prod_p = w_combine * prodc;
test_error_nist_combine_prod_p = nist_eval('my_rep_1', w_c_prod_p, 50);

