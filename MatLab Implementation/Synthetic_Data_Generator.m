%% Synthetic Data Generator
clear; clc;

N = 500;
sd = 0.15;
r0 = 2/3;
N_junk_features = 100;

rr_a = r0+ normrnd(0,sd,[N,1]);
rr_b = r0+ normrnd(0,sd,[N,1]);
theta_a = random('unif',-pi/6,pi*7/6,[N,1]);
theta_b = random('unif',-pi*7/6,pi/6,[N,1]);

xx_a = ones(N,1)*[-1/3,0] + (rr_a*ones(1,2)).*[cos(theta_a),sin(theta_a)];
xx_b = ones(N,1)*[1/3,0] + (rr_b*ones(1,2)).*[cos(theta_b),sin(theta_b)];

%% Adding Unrepresentative "Junk Features"

N_rand=rand(N,N_junk_features); 
xx_a=[xx_a N_rand];

N_rand=rand(N,N_junk_features); 
xx_b=[xx_b N_rand];

X=[xx_a;xx_b];

Y=[ones(N,1); -ones(N,1)];


%% Plot
figure
Font_size=15;
acolor=[0.5,0,0];
s=scatter(xx_a(:,1),xx_a(:,2));hold on

s.LineWidth = 0.4;
s.MarkerEdgeColor = acolor;
s.MarkerFaceColor = acolor;

acolor=[0,0,0.5];
s=scatter(xx_b(:,1),xx_b(:,2),'*')
s.MarkerEdgeColor = acolor;
s.MarkerFaceColor = acolor;

%% 
grid on
axis ([-1.4 1.4 -1.1 1.1])
set(gca,'FontSize',13)
xlabel('X_1','FontSize',Font_size)
ylabel('X_2','FontSize',Font_size)
aux = get(gca,'Children');

ax= gca;
ax.Position= [0.1 0.1100 0.87 0.85];

aux = get(gca,'Children');
lgd= legend([aux(2),aux(1)],'Class 1','Class 2','Location','NorthEast');
lgd.FontSize = Font_size;

%% figure
% Font_size=15;
% acolor=[0.5,0,0];
% s=scatter3(xx_a(:,1),xx_a(:,2),xx_a(:,3));hold on
% 
% s.LineWidth = 0.4;
% s.MarkerEdgeColor = acolor;
% s.MarkerFaceColor = acolor;
% 
% acolor=[0,0,0.5];
% s=scatter3(xx_b(:,1),xx_b(:,2),xx_b(:,3),'*')
% s.MarkerEdgeColor = acolor;
% s.MarkerFaceColor = acolor;
