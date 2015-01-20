load('data.mat');
load('weights.mat');

L_A1 = 20 * 20;
L_A2 = 25;
L_A3 = 10;

m = size(X, 2);

a1 = [ones(1, m); X];
z2 = Theta1 * a1;
a2 = sigmoid(z2);
a2 = [ones(1,size(a2,2)); a2];

z3 = Theta2 * a2;
a3 = sigmoid(z3);

J = 0;
for i = 1:1,  
    J += sum(-1 * y(:,i) .* log(a3(:,i)) - (1 - y(:,i)) .* log(1 - a3(:,i)));
end;

% J = J / m;

lambda = 1;
J += lambda / (2 * m) * (sum(sum(Theta1(2:end,:) .^ 2)) + sum(sum(Theta2(2:end,:) .^ 2)))

% Delta1 = zeros(size(Theta1));
% Delta2 = zeros(size(Theta2));

% delta3_ = a3 - y;
% delta2_ = (Theta2' * delta3_)(2:end,:) .* sigmoidGradient(z2);

% for i = 1:m,
% 	delta3 = delta3_(:,i);
% 	delta2 = delta2_(:,i);

% 	Delta1 += delta2 * X(:,i)';
% 	Delta2 += delta3 * a2(:,i)';
% end

% th1_g = Delta1 / m;
% th2_g = Delta2 / m;

% th1_g(2:end,:) = th1_g(2:end,:) + lambda * Theta1(2:end,:) / m;
% th2_g(2:end,:) = th2_g(2:end,:) + lambda * Theta2(2:end,:) / m;
