clc;
clear all;
close all;
%% file updates
    file1 = fopen('results.txt','w+');
    file2=fopen('results_nu_mu.txt','w+');
%% parameters
    epsilon = 0.01;
    no_part = 10.;
%     win_sz = 5+1;
 for load_file = 13:13    %% initializing variables case level and loading file
    switch load_file
        case 1
            file = 'Function1';%%sin(x)cos(x^2) %Function1 to Function6 used Gaussian noises with mean zero and standard deviation 0.2 
              nuvs = 10^1;
              muvs = 2^4;
              test_start = 201;
%         case 2
%             file = 'Function2'; 
%             nuvs= 10^1;
%             muvs= 2^-4;
%               test_start = 201;
%         case 3
%             file = 'Function3'; 
%             nuvs= 10^1;
%             muvs= 2^1;
%               test_start = 201;
%         case 4
%             file = 'Function4';  
%             nuvs= 10^1;
%             muvs= 2^-10;
%               test_start = 201;
%         case 5            
%             file = 'Function5'; 
%             nuvs= 10^1;
%             muvs= 2^0;
%               test_start = 201;
%         case 6
%             file = 'Function6'; 
%             nuvs= 10^0;
%             muvs= 2^4;
%               test_start = 201;
        case 7
            file = 'Function7';%Function7 to Function12 used Uniform noises over the interval [-0.2, 0.2]
            nuvs= 10^5;
            muvs= 2^2;
            test_start = 201;
%         case 8
%             file = 'Function8';
%             nuvs= 10^1;
%             muvs= 2^-4;
%             test_start = 201;
%         case 9
%             file = 'Function9';
%             nuvs= 10^1;
%             muvs= 2^1;
%             test_start = 201;
%         case 10
%             file = 'Function10';
%             nuvs= 10^5;
%             muvs= 2^-9;
%             test_start = 201;
%         case 11
%             file = 'Function11';
%             nuvs= 10^1;
%             muvs= 2^1;
%             test_start = 201;
%         case 12            
%             file = 'Function12';
%             nuvs= 10^5;
%             muvs= 2^5;
%              test_start = 201;
        case 13
            file = 'Function13'; %Function13 to Function18 used Gaussian noises with mean zero and standard deviation 0.05
            nuvs= 10^1;
            muvs= 2^5;
            test_start = 201;
%         case 14
%             file = 'Function14';
%             nuvs= 10^-5;
%             muvs= 2^-4;
%             test_start = 201;
%         case 15
%             file = 'Function15';
%             nuvs= 10^1;
%             muvs= 2^1;
%             test_start = 201;
%         case 16
%             file = 'Function16';
%             nuvs= 10^0;
%             muvs= 2^-7;
%             test_start = 201;
%         case 17
%             file = 'Function17';
%             nuvs= 10^0;
%             muvs= 2^2;
%              test_start = 201;
%         case 18
%             file = 'Function18';
%             nuvs= 10^1;
%             muvs= 2^4;
%              test_start = 201;
%        case 19
%             file = 'Function22a';
%             nuvs= 10^5;
%             muvs= 2^3;
%              test_start = 201;
%        case 20
%             file = 'Function22b';
%             nuvs= 10^5;
%             muvs= 2^3;
%              test_start = 201;
%        case 21
%             file = 'Function22c';
%             nuvs= 10^-2;
%             muvs= 2^4;
%              test_start = 201; 
        case 22
            file = 'demo';
            nuvs= 10^1;
            muvs= 2^3;
            test_start = 501;        
                  
        case 23
            file = 'google';
            nuvs= 10^5;
            muvs= 2^-7;
            test_start = 201;
        case 24
            file = 'ibm';
            nuvs= 10^5;
            muvs= 2^-5;
            test_start = 201;
      
        case 25
            file = 'ms_stock';
            nuvs= 10^1;
            muvs= 2^-3;
            test_start = 201;
        case 26
            file = 'machine';
            nuvs= 10^2;
            muvs= 2^-3;
            test_start = 151;
        case 27
            file = 'abalone';
            nuvs= 10^2;
            muvs= 2^-10;
            test_start = 1001;
        case 28
            file = 'concreteCS';
            nuvs= 10^2;
            muvs= 2^-1;
            test_start = 701;              
        case 29
            file = 'kin8192';
            nuvs= 10^1;
            muvs= 2^-1;
            test_start = 1001;
        case 30
            file = 'bank-32fh';
            nuvs= 10^5;
            muvs= 2^-10;
            test_start = 1001;
        case 31
            file = 'forestfires';
            nuvs= 10^0;
            muvs= 2^10;
            test_start = 201;              
        case 32
            file = 'flares'; 
            nuvs= 10^0;
            muvs= 2^0;
            test_start = 201;                
        case 33
            file = 'NO2';  
            nuvs= 10^5;
            muvs= 2^-9;
            test_start = 101;                
        case 34
            file = 'auto-original';
            nuvs= 10^1;
            muvs= 2^-4;
            test_start = 101; 
        case 35
            file = 'gas_furnace2';%size 293*6
            nuvs= 10^2;
            muvs= 2^-1;
            test_start = 148;
            
%         case 35
%             file = 'gasfurnacenew';%generated specially for {y(t-1),u(t-4)} to predict output variables of size 292*2
% %             nuvs= 10^2;
% %             muvs= 2^-5;
%             test_start = 101;
             
% %       case 50
% %             file = 'YearPredictionMSD1000Modified';%Without normalize
% %             nuvs= 10^-1;
% %             muvs = 2^-2;
% %             test_start = 801; 
% case 22
%             file = 'lorenz2';
% %             nuvs= 10^5;
% %             muvs= 2^0;
%              test_start = 501;
%               case 23
%             file = 'lorenz4';
% %             nuvs= 10^5;
% %             muvs= 2^0;
%               test_start = 501; 
%                 case 26
%             file = 'bodyfat';
% %             nuvs= 10^0;
% %             muvs= 2^-4;
%               test_start = 151;
%         case 27
%             file = 'RedHat';
% %             nuvs= 10^-3;
% %             muvs= 2^-3;
%               test_start = 201;
%         case 28
%             file = 'sunspots94';
% %             nuvs= 10^1;
% %             muvs= 2^1;
%               test_start = 101;
%         case 29
%             file = 'mg17';
% %             nuvs= 10^0;
% %             muvs= 2^3;
%               test_start = 501;
%                case 32
%                    file = 'mg30';
% %             nuvs= 10^1;
% %             muvs= 2^3;
%               test_start = 501;
%         case 33
%             file = 'santafeA';
% %             nuvs= 10^-1;
% %             muvs= 2^3;
%               test_start = 201;
%         case 34
%             file = 'intel';
% %             nuvs= 10^0;
% %             muvs= 2^-3;
%               test_start = 201;
%         case 35
%             file = 'SNP500';
% %             nuvs= 10^-5;
% %             muvs= 2^1;
%               test_start = 301; 
%               
%               case 40
%             file = 'boston';
% %             nuvs= 10^0;
% %             muvs= 2^-5;
%              test_start = 201;
%         case 41
%             file = 'winequality-white';
% %             nuvs= 10^0;
% %             muvs= 2^-1;
%             test_start = 1001;
%             case 46
%             file = 'servo';
% %             nuvs= 10^5;
% %             muvs= 2^1;
%              test_start = 101;   
                   
             
              otherwise
            continue;
    end
      
%               nuvs =[10^5,10^4,10^3,10^2,10^1,10^0,10^-1,10^-2,10^-3,10^-4,10^-5];
%               muvs =[2^-10,2^-9,2^-8,2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3,2^4,2^5,2^6,2^7,2^8,2^9,2^10]; 
%        nuvs = [10^5, 10^-1];
%        muvs = [2^-10, 2^-1];
       filename = strcat('../dataset/',file,'.txt');
        A = load(filename);
    
   %% windowing, for time series only
%     win_sz = 5+1;
%     [m,n] = size(A);
%     if n == 1
%         A = window(A,win_sz);
%         m = m - win_sz + 1;
%     end
    
    %% spliting into testing and training datapoints
    %if test_start = 1, then whole data points will be used for both
    %testing as well as training
    [m,n] = size(A);
    A_test = A(test_start:m,:);
    if test_start > 1
        A = A(1:test_start-1,:);
    end    
    
%      A = normalize(A);
%      A_test = normalize(A_test);

%         A = scale(A);
%         A_test = scale(A_test);
    % initializing crossvalidation variables
    [m,n] = size(A);
    min_err = 10^10.;
    min_nu = 11111111110;
    min_mew = 2^-12;

    % for each nu
    for nui = 1:length(nuvs)
        nu = nuvs(nui)
        % for each mu
        for mewi = 1:length(muvs)
            mew = muvs(mewi)
 % training statement
            block_size = m/(no_part*1.0);
            part = 0;
            avgerror = 0;
            t_1 = 0;
            t_2 = 0;
            while ceil((part+1) * block_size) <= m
                % seprating testing and training datapoints for crossvalidation
                t_1 = ceil(part*block_size);
                t_2 = ceil((part+1)*block_size);
                Data_test = A(t_1+1 :t_2,:);
                Data = [A(1:t_1,:); A(t_2+1:m,:)];

                % testing and training
                [RMSE,ytest,SD]= mohitSVR(Data,Data_test,mew,nu,epsilon);
                RMSE
                avgerror = avgerror + RMSE;
                part = part+1;
                 if min_err > avgerror
                    continue;
                else
                    break;
                end
            end
          
         % Testing statement updating optimum nu, mew
             fprintf(file1,'file = %s\t,mew = %8.6g\t,nu = %8.6g\t,avgerror = %8.6g\t\n',file,mew,nu,avgerror);
            if avgerror < min_err
                min_nu = nu;
                min_err = avgerror;
                min_mew = mew;
            end
        end
    end 

%                 min_mew = muvs;
%                 min_nu = nuvs;
                 
%   Replace comments by uncomments and vice-versa before this.
    %% final training and testing and Time taken for training and testing
      [RMSE,ytest,SD,time,w]=  mohitSVR(A,A_test,min_mew,min_nu,epsilon)
           
      %[RMSE,train]= lpp_TSVR(A,A,min_mew,min_nu,epsilon1,epsilon2); % used for generating graph for train data

    %% writing results
    train_data = A;
    test_data = A_test;
     ytest0=A_test(:,n);
 plotfile=strcat(file,'.mat');
 save(plotfile,'test_data','train_data','ytest', 'RMSE', 'min_mew','min_nu');
 fprintf(file2,'File Name = %s\t;RMSE = %8.6g\t, optimum_mew = %g\t optimum_nu = %g,SD=%g\n',file,RMSE,min_mew,min_nu,SD);
  plot(A_test(:,1),ytest0,'r.',A_test(:,1),ytest,'b.') 

end 
fclose(file1);
fclose(file2);
