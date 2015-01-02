L_A1 = 28 * 28;
L_A2 = 25;
L_A3 = 10;

z2 = X * Theta1;
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1) a2];

z3 = a2 * Theta2;
a3 = sigmoid(z3)
