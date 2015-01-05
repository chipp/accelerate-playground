L_A1 = 28 * 28;
L_A2 = 25;
L_A3 = 10;

z2 = Theta1 * X;
a2 = sigmoid(z2);
a2 = [ones(1,size(a2,2)); a2];

z3 = Theta2 * a2;
a3 = sigmoid(z3);

m = size(X, 2);

J = 0;
for i = 1:m,  
    J += sum(-1 * y(:,i) .* log(a3(:,i)) - (1 - y(:,i)) .* log(1 - a3(:,i)));
end;

J = J / m;

lambda = 1;
J += lambda / (2 * m) * (sum(sum(Theta1(:,2:end) .^ 2)) + sum(sum(Theta2(:,2:end) .^ 2)));

Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

delta3_ = a3 - y;
Theta2' * delta3_
% delta2_ = (Theta2' * delta3_)(2:end,:) .* sigmoidGradient(z2)

for i = 2:2,
    delta3 = delta3_(:,i);
	delta2 = (Theta2' * delta3)(2:end,:) .* sigmoidGradient(z2(:,i))

	% Delta1 += (delta2 * X(i,:))';
	% Delta2 += (delta3' * a2(i,:))';
end
