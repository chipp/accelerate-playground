L_A1 = 28 * 28;
L_A2 = 10;
L_A3 = 10;

z2 = X * Theta1;
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1) a2];

z3 = a2 * Theta2;
a3 = sigmoid(z3);

m = size(X, 1);

J = 0;
for i = 1:m,  
    J += sum(-1 * y(i,:) .* log(a3(i,:)) - (1 - y(i,:)) .* log(1 - a3(i,:)));
end;

J = J / m;

lambda = 1;
J += lambda / (2 * m) * (sum(sum(Theta1(2:end,:) .^ 2)) + sum(sum(Theta2(2:end,:) .^ 2)));

Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

_delta3 = a3 - y;

for i = 1:1,
    delta3 = _delta3(i,:);
	delta2 = (Theta2 * delta3')(2:end,:) .* sigmoidGradient(z2(:,i));

	% Delta1 += (delta2 * X(i,:))';
	% Delta2 += (delta3' * a2(i,:))';
end
