function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

for i = 1:m
  a2 = zeros(size(Theta1, 1), 1);
  x = [1; X(i,:)'];
  for j = 1:size(Theta1, 1)
    a2(j) = sigmoid(Theta1(j,:)*x);
  endfor
  
  a2 = [1 ; a2];
  bestIndex = 0;
  bestPrediction = -1;
  for j = 1:num_labels
    currentPrediction = sigmoid(Theta2(j,:)*a2)(:);
    if currentPrediction > bestPrediction
      bestIndex = j;
      bestPrediction = currentPrediction;
    endif
  endfor
  
  p(i) = bestIndex;
endfor

% =========================================================================


end
