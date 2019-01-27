%load img data

dir = 'D:\document\master\Q2\Pattern recognition\workspace\final_assignment\live_test_dataset';
index = 1;
img_matrix = [];
label = [];
big_image = cell(10,10);
imgSize = 32;
for row=1:10
    
   for line=1:10
      img_file = strcat(num2str(row-1),'_',num2str(line-1),'.png');
      file_name = fullfile(dir,img_file);
      current_img = imread(file_name);
      gray_img = rgb2gray(current_img);
      img_double = im2double(gray_img);
      big_image{row,line} = img_double;
      img_double_new = ~img_double;
      
      img_new = imdilate(img_double_new,ones(3));
      %imshow(img_new);
      img_new = im2double(img_new);
      img_new_1 = img_new * im_box(0,1);
      img_double_resize = im_resize(img_new_1, [imgSize-2, imgSize-2]);
      img_double_resize = img_double_resize * im_box(1,0);
      
      imshow(img_double_resize);
      img_matrix = [img_matrix img_double_resize];
      %img_cell = mat2cell(img_double)
      index = index + 1;
      
      %label
      %label_file = strcat('',num2str(row-1),'');
      %label = [label;label_file];
      
   end
end

%img data to pr data
labs = genlab([10 10 10 10 10 10 10 10 10 10],['digit_0'; 'digit_1'; 'digit_2'; 'digit_3'; 'digit_4'; 'digit_5'; 'digit_6'; 'digit_7'; 'digit_8'; 'digit_9']); 
dim = imgSize * ones(1,100);
img_dataset = mat2cell(img_matrix,[imgSize],dim);
a = prdataset(img_dataset',labs);

big_image_all = cell2mat(big_image);
imshow(big_image_all)

%% train
img_nist = prnist([0:9],[1:100:1000]);
pixel_data = my_rep(img_nist);

%pca
%pca_coeff = pixel_data * pcam(0.8)
%pixel_data = pixel_data*pca_coeff

%parzen
w_parzen = pcam(pixel_data,0.85) * parzenc([]);
w_p = pixel_data*w_parzen;
e_parzen = a*w_p*testc%test
%knn
w_knn = pcam(pixel_data,0.85) * knnc([]);
w_k = pixel_data*w_knn;
e_knn = a*w_k*testc%test
%ldc
w_ldc = pcam(pixel_data,0.85) * ldc([]);
w_l = pixel_data*w_ldc;
e_ldc = a*w_l*testc%test

%nmc
w_nmc = pcam(pixel_data,0.85) * nmc([]);
w_n = pixel_data*w_nmc;
e_nmc = a*w_n*testc%test

%fisherc
w_fisherc = pcam(pixel_data,0.85) * fisherc([]);
w_f = pixel_data*w_fisherc;
e_fisherc = a*w_f*testc%test

%qdc
w_qdc = pcam(pixel_data,0.85) * qdc([]);
w_q = pixel_data*w_qdc;
e_qdc = a*w_q*testc%test

%loglc
w_loglc = pcam(pixel_data,0.85) * loglc([]);
w_log = pixel_data*w_loglc;
e_loglc = a*w_log*testc%test

%svc
%w_svc = pcam(pixel_data,0.85) * svc([]);
%w_s = pixel_data*w_svc;
%e_svc = a*w_s*testc%test

