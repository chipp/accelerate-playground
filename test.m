cost;
th1_cg = th1_g;
th1_c = Theta1;
eps = 1e-4;

Theta1 = th1_c + eps;
cost;
th1_gp = th1_g;

Theta1 = th1_c - eps;
cost;
th1_gm = th1_g;

grad1 = (th1_gp - th1_gm) / 2 / eps;

norm(grad1 - th1_cg) / norm(grad1 + th1_cg)