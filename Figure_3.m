%% Plot DaSalla solo Forward
clear;
K=50;
CV=20;
M_AUC_F_tst=[];
M_AUC_F_trn=[];
root_dir= 'Resultados_Paper/DaSalla_full/';
load (strcat(root_dir,'DaSalla_s123_act123_N500_K50_cv20.mat'));
Neuronas_finales=1:1:500;
aux=Resul_general{1,1};
aux=aux.Acc.f;
Acc_trn_f=aux.trn;
Acc_tst_f=aux.tst;
for k=1:K 
    M_AUC_F_tst= [M_AUC_F_tst ; mean(Acc_trn_f(((k-1)*CV)+1:(k*CV),:))];
    M_AUC_F_trn= [M_AUC_F_trn ; mean(Acc_tst_f(((k-1)*CV)+1:(k*CV),:))];
end


%%
alpha=0.3;
F=Neuronas_finales;
Font_size=15;
close all
figure
acolor=[0.5,0,0];
stdshade3(M_AUC_F_trn,alpha,acolor,F);hold on
acolor=[0,0,0.5];
stdshade3(M_AUC_F_tst,alpha,acolor,F);hold on
axis ([min(Neuronas_finales) max(Neuronas_finales) 50 102])
grid on
set(gca,'FontSize',13)
xlabel('Number of Hidden Nodes M','FontSize',Font_size)
ylabel('Accuracy','FontSize',Font_size)
aux = get(gca,'Children');
lgd= legend([aux(1) aux(3)],'Train','Test','Location','southeast');
lgd.FontSize = Font_size;
ax= gca;
ax.Position= [0.1 0.1100 0.87 0.85];



set(ax,'xscale','log')

%% Save plot
%- Figure size in centimeters
%- (it can be included in the paper without stretching):
width = 21;
height = 12;

%- Figure name:
filename = 'Da_Salla_overfitting_all.pdf';

%- Activate removal of margins (off by default)
remove_margin = false;

print_figure (filename, width, height, 'RemoveMargin', remove_margin,'FontSize',Font_size)
