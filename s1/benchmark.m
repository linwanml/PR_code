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
[w_feat_p, r_feat_p] = featselm(a_pca,'eucl-s', 'forward', 28);
w_final_p = w_feat_p(:, 1:28);
a_optfeat_p = a * w_pca * w_final_p; 
clsf = parzenc;
w_parzen = w_pca * w_final_p * clsf(a_optfeat_p)
test_error_nist_parzen = nist_eval('my_rep',w_parzen,50);

%%
%final mapping after feature selection
%knn
[w_feat_k, r_feat_k] = featselm(a_pca,'eucl-s', 'forward', 21);
w_final_k = w_feat_k(:, 1:21);
a_optfeat_k = a * w_pca * w_final_k; 
clsf = knnc;
w_knn = w_pca * w_final_k * clsf(a_optfeat_k)
test_error_nist_knn = nist_eval('my_rep', w_knn, 50);


