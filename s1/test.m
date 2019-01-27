%% - Matlab Setup
clc;
clear;

%% - PRtools setting
%addpath('prtools');
%addpath('coursedata');
%addpath('prdatasets');
prwaitbar on;
prwarning off;


%% - Initialisations
load img_data.mat
a = img_data
[trn_ori, tst_ori] = gendat(a, 0.8);
% w_pca = pcam(trn_ori, 0.99);
w_pca = pcam(a, 0.85);
a_pca = a * w_pca;
trn = trn_ori * w_pca;
tst = tst_ori * w_pca;

%% - Feature selection
% featnum = [1 : size(trn, 2)];
featnum = [1 : size(a_pca, 2)];
mf = max(featnum);

%% 
% - selection routine setup
crits = ["maha-s"; "maha-m"; "eucl-s"; "eucl-m"];
methods = ["forward"; "backward"; "ind"];
% crits = ["eucl-m"];
% methods = ["forward"];
w_featsel = cell(length(crits), length(methods));
r_featsel = cell(length(crits), length(methods));

for index_crit = 1 : length(crits)
    for index_method = 1 : length(methods)
%         [w_tmp, r_tmp] = featselm(trn, char(crits(index_crit)), char(methods(index_method)), mf);
        [w_tmp, r_tmp] = featselm(a_pca, char(crits(index_crit)), char(methods(index_method)), mf);
        w_featsel(index_crit, index_method) = {w_tmp};
        r_featsel(index_crit, index_method) = {r_tmp};
    end
end


clsf = {nmc, ldc, qdc, fisherc, loglc, knnc, parzenc};
rslt = struct([]);
index_rslt = 1;

for index_clsf = 1 : length(clsf)
    for index_crit = 1 : length(crits)
        for index_method = 1 : length(methods)
            w_feat = w_featsel{index_crit, index_method};
            r_feat = r_featsel{index_crit, index_method};

            e = clevalf(trn*w_feat, clsf(index_clsf), featnum,[],1,tst*w_feat);
            %e = clevalf(a_pca*w_feat, clsf(index_clsf), featnum,[],1);

            % optimal feature number
            [error_min, fnum_opt] = min(e.error);

            % - result structure
            rslt(index_rslt).crit       =   crits(index_crit);
            rslt(index_rslt).method     =   methods(index_method);
            rslt(index_rslt).clsf       =   clsf(index_clsf);
            rslt(index_rslt).e          =   e;
            rslt(index_rslt).error_min  =   error_min;
            rslt(index_rslt).fnum_opt   =   fnum_opt; 
            rslt(index_rslt).w_feat     =   w_feat;
            rslt(index_rslt).r_feat     =   r_feat;
            index_rslt = index_rslt + 1;
        end
    end
end

%%
%{
for i = 1 : 7
    switch(i)       % select classifier
        case(1)
            clsf = nmc;
        case(2)
            clsf = ldc;
        case(3)
            clsf = qdc;
        case(4)
            clsf = fisherc;
        case(5)
            clsf = loglc;
        case(6)
            clsf = knnc;
        case(7)
            clsf = parzenc;     
    end
    
    % - test
    r = rslt(i).r_feat;    
    w = rslt(i).w_feat;
    featnum_opt = rslt(i).fnum_opt;
    e = rslt(i).e;
    
    w_final = w(:, 1:featnum_opt);
%     test_error(i) = prcrossval(trn_ori, w_pca * w_final* clsf, 10, 2);
%     test_error(i) = prcrossval(trn_ori, w_pca * clsf, 10, 2);
    opt_feat = r(1:featnum_opt, 3);
    P = zeros(length(r(1:featnum_opt, 3)), length(e(1).error));
    for k = 1 : length(r(1:featnum_opt, 3))
        P(k, opt_feat(k)) = 1;
    end
    trn_optfeat = trn * P';
%     load data_10000.mat;
    tst_ori = a_processed;
    tst = tst_ori * w_pca;
    tst_optfeat = tst * P';
    a_optfeat = a * w_pca * P';

%     test_error(i) = prcrossval(a_optfeat, clsf, 10, 2);
%     test_error(i) = tst_optfeat * clsf(a_optfeat) * testc;

    test_error(i) = nist_eval('my_rep_smalldataset', w_pca * P' * clsf(a_optfeat), 10);
end
%}
