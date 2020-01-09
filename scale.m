function [res] = scale(A)
    mx = max(A);
    mn = min(A);
    e = ones(size(A,1),1);
    res = (A-e*mn)./(e*(mx-mn));