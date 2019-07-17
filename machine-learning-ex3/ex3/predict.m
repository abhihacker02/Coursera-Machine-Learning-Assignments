function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
X = [ones(m, 1) X];
a1=X;
z2=(Theta1*a1')';
a2=sigmoid(z2);
a2 = [ones(m, 1) a2];
z3=(Theta2*a2')';
a3=sigmoid(z3);

prob=a3;
%below code is taken from predictOneVsAll
for i=1:m
  mx=0;
  pos=0;
  for j=1:num_labels
    if (prob(i,j)>mx)
      mx=prob(i,j);
      pos=j;
    endif
  endfor
  p(i)=pos;
  if (pos==10)
    p(i)=0;
  endif
endfor

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









% =========================================================================


end
