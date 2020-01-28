function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

% You should return this values correctly
centroids = zeros(K, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should set centroids to randomly chosen examples from
%               the dataset X
%

m = size(X,1);

for k = 1:K
  centroids(k,:) = X(randi(m),:);
  for previousK = 1:(k-1) % check if centroid is not repeated
    if(centroids(previousK,:) == centroids(k,:))
      centroids(k,:) = X(randi(m),:);
      previousK = 1;
    endif
  endfor
endfor

% =============================================================

end

