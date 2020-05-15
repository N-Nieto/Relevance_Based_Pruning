%% Synthetic Data Generator
clear; clc;

N = 1000;
sd = 0.15;
r0 = 2/3;

rr_a = r0+ normrnd(0,sd,[N,1]);
rr_b = r0+ normrnd(0,sd,[N,1]);
theta_a = random('unif',-pi/6,pi*7/6,[N,1]);
theta_b = random('unif',-pi*7/6,pi/6,[N,1]);

xx_a = ones(N,1)*[-1/3,0] + (rr_a*ones(1,2)).*[cos(theta_a),sin(theta_a)];
xx_b = ones(N,1)*[1/3,0] + (rr_b*ones(1,2)).*[cos(theta_b),sin(theta_b)];

figure
Font_size=15;
acolor=[0.5,0,0];
s=scatter(xx_a(:,1),xx_a(:,2));hold on

s.LineWidth = 0.6;
s.MarkerEdgeColor = acolor;
s.MarkerFaceColor = acolor;

acolor=[0,0,0.5];
s=scatter(xx_b(:,1),xx_b(:,2))
s.MarkerEdgeColor = acolor;
s.MarkerFaceColor = acolor;


grid on
axis ([-1 1 -1 1])
set(gca,'FontSize',13)
xlabel('X_1','FontSize',Font_size)
ylabel('X_2','FontSize',Font_size)
aux = get(gca,'Children');
% lgd= legend([aux(1)],'Trn','Location','southeast');
% lgd.FontSize = Font_size;
ax= gca;
ax.Position= [0.1 0.1100 0.87 0.85];
%% Label

Y1= ones(N,1);
Y2= -ones(N,1);

X= [xx_a ; xx_b];
Y= [Y1 ; Y2];


%% Save plot
%- Figure size in centimeters
%- (it can be included in the paper without stretching):
width = 21;
height = 12;

%- Figure name:
filename = 'Sintetic_data.pdf';

%- Activate removal of margins (off by default)
remove_margin = false;

print_figure (filename, width, height, 'RemoveMargin', remove_margin,'FontSize',Font_size)
