%% Figura 5 b
%% Plot P300
clear;
steps=1:ceil(5*log10(30000));
Neuronas_finales=ceil(10.^(steps/5));
K=20;
CV=5;

root_dir= 'Resultados_Paper/Exp_P300/';
N_subj=2;

M_AUC_F=[];
M_AUC_A=[];
M_AUC_R=[];
M_AUC_R_trn=[];
M_AUC_R_v=[];

for k=1:K
    % Load k data
    load (strcat(root_dir,'S',num2str(N_subj),'/Results_k',num2str(k)));
    % apilate the mean AUC of the 5 CVs
    M_AUC_F= [M_AUC_F ; mean(Resoults.AUC.f.tst)];
    M_AUC_A= [M_AUC_A ; mean(Resoults.AUC.a.tst)];
    
    
    load (strcat('Resultados_Paper/Subject_1_random/Results_',num2str(k)));
    
    M_AUC_R= [M_AUC_R ; mean(Results.AUC.a.tst)];
    M_AUC_R_trn= [M_AUC_R_trn ; mean(Results.AUC.a.trn)];
    
    M_AUC_R_v= [M_AUC_R_v ; mean(Results.AUC.a.val)];
    
end




%% Plotting Time
figure
alpha=0.3;
F=Neuronas_finales;
acolor=[0,0.5,0];
stdshade3(M_AUC_A,alpha,acolor,F);hold on
acolor=[0,0,0.5];
stdshade3(M_AUC_F,alpha,acolor,F);hold on
acolor=[0.5,0,0];
stdshade3(M_AUC_R,alpha,acolor,F);hold on
grid on
axis ([min(Neuronas_finales) max(Neuronas_finales) 0.5 max(mean(M_AUC_R))+0.02])

Font_size= 15;

set(gca,'FontSize',13)
xlabel('Number of Hidden Nodes M','FontSize',Font_size)
ylabel('AUC','FontSize',Font_size)


aux = get(gca,'Children');

lgd= legend([aux(1),aux(3), aux(5)],'RND','FWD','RBP','Location','southeast');
lgd.FontSize = Font_size;

ax= gca;
ax.Position= [0.1 0.1100 0.87 0.85];




%% Save plot
%- Figure size in centimeters
%- (it can be included in the paper without stretching):
width = 21;
height = 12;

%- Figure name:
filename = 'P300_RND_FWR_RBP.pdf';

%- Activate removal of margins (off by default)
remove_margin = false;

print_figure (filename, width, height, 'RemoveMargin', remove_margin,'FontSize',Font_size)
