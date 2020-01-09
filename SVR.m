function [RMSE,ytest,SD,time,w] = mohitSVR(train,test,mew,nu,epsilon)
    [no_input,no_col] = size(train);
    x1 = train(:,1:no_col-1);
    y1 = train(:,no_col);
    
    [no_test,no_col] = size(test);
    xtest0 = test(:,1:no_col-1);
    ytest0 = test(:,no_col);
    m = no_input;
    m3 = 3 * m;
    ep = 0.1;         %  penality parameter   %
    
    tol = 0.01;
    itmax = 100;
    beta = 10^-4;
    
    no_train = no_input;
    
    %% Kernel
    K=zeros(no_train,no_train);

    for i=1:no_input
        for j=1:no_input
             nom = norm( [x1(i,:) 1] - [x1(j,:) 1] );
             K(i,j) = exp( -mew * nom * nom );
        end
    end
    
    [m,n] = size(K);
    I = speye(m3);
    e = ones(m,1);
    y = y1;
    m1 = m+1;
    em1 = ones(m1,1);
    
    G = [K e];
    GT = G' ;
    
    iter = 0;
    u1 = ones(m,1);
    u2 = u1;
    u3 = u1;
    
    delphi= 1;
    
    delta = zeros(m3,1);
    alpha = 5.0;
    %% Iteration Starts Here
    
    while( iter < itmax & norm (delphi) > tol )
        iter = iter + 1;
        del11 = max(GT*u3-em1, 0);
        del12 = max(-GT*u3-em1, 0);
        del13 = max(-u1+u2+u3, 0);
        del14 = max(u1-u2-u3, 0);
        del15 = max(u1-nu*e, 0);
        del16 = max(-u1, 0);
        del17 = max(u2-nu*e, 0);
        del18 = max(-u2, 0);
        initial_train_time=tic;
        delphi_1 = ep*e*epsilon - del13 + del14 + del15 - del16;
        delphi_2 = ep*e*epsilon + del13 - del14 + del17 - del18;
        delphi_3 = -ep*y + G*(del11-del12) + del13 - del14;
        
        delphi = [delphi_1; delphi_2; delphi_3];
        
%         temp = diag(sign(del13)) + diag(sign(del14));
%         H11 = temp + diag(sign(del15)) + diag(sign(del16));
%         H12 = -temp;
%         H13 = -temp;
%         H22 = temp + diag(sign(del15)) + diag(sign(del18));
%         H23 = temp;
%         H33 = G*(diag(sign(del11))+diag(sign(del12)))*GT + temp;
        D1 = 1./(1+exp(-alpha*(-u1+u2+u3)));
        D2 = 1./(1+exp(-alpha*(u1-u2-u3)));
        D3 = 1./(1+exp(-alpha*(u1-nu*e)));
        D4 = 1./(1+exp(alpha*u1));
        D5 = 1./(1+exp(-alpha*(u2-nu*e)));
        D6 = 1./(1+exp(alpha*u2));
        D7 = 1./(1+exp(-alpha*(G'*u3-em1)));
        D8 = 1./(1+exp(-alpha*(-G'*u3-em1)));
        
        H12 = -diag(D1)-diag(D2);
        H13 = H12;
        H11 = -H12 + diag(D3)+diag(D4);
        H22 = -H12 + diag(D5) + diag(D6);
        H23 = -H12;
        H33 = G*(diag(D7)+diag(D8))*GT - H12;
        hessian = [H11 H12 H13; H12 H22 H23; H13 H23 H33];
        delta = (hessian + beta * I)\delphi;
        
        u1 = u1 - delta(1:m);
        u2 = u2 - delta((m+1):(2*m));
        u3 = u3 - delta((2*m + 1):m3);
    end
    iter
    %% Calculating Multiplication Mattrix
    del11 = max(GT*u3 - em1, 0);
    del12 = max(-GT*u3 - em1, 0);
    
    w = (del11-del12)/ep
      time = toc(initial_train_time);
    
    %% Testing Part Starts Here
    
    ytest = zeros(no_test,1);
    
    for i=1:no_test
        ytest(i) = w(m1);
        for k=1:no_input
            nom = norm( [xtest0(i,:) 1] -  [x1(k,:) 1] );
            ytest(i) = ytest(i) + exp( -mew * nom * nom ) * w(k);
        end    
    end
    ytest
    
    RMSE = sqrt( norm(ytest-ytest0)* norm(ytest-ytest0) /no_test );
    ytestMean=sum(ytest,1)/no_test;
    y1=(ytest-ytestMean).*(ytest-ytestMean);
    y2=sum(y1,1)/no_test;
    SD=sqrt(y2);
end
