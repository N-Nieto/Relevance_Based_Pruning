%% RBP ELMs on Synthetic Data
clear;clc; close all;
%% Set Up
M_star=500;
K=5;
CV=2;
Test_percent=0.90;
N_initial_trash_features=2;

N_steps=4;

General_results=cell(1,N_steps);

% % Datos Sintéticos Nuestros
load ('Synthetic_data.mat');
x=X;
y=Y;
N_ejemplos= size(x,1);


for n_e=1:N_steps
    %% Carga de datos
    n_e
    
    
    
    if n_e==1
        N_initial_trash_features0=0;
        N_rand=rand(N_ejemplos,N_initial_trash_features0);
    else
        N_initial_trash_features=N_initial_trash_features*4;
        N_rand=rand(N_ejemplos,N_initial_trash_features);
    end
    x=[x N_rand];
    
    %% Inicializacion
    Data_dimension=size(x,2);
    % para afterpruning
    reg_trn_a=[];
    reg_tst_a=[];
    Acc_trn_a=[];
    Acc_tst_a=[];
    Beta_min_a=[];
    Beta_norm_a =[];
    H_cond_a= [];
    AUC_tst_f=[];
    
    
    % para entrenamiento forward
    reg_trn_f=[];
    reg_tst_f=[];
    Acc_trn_f=[];
    Acc_tst_f=[];
    Beta_min_f= [];
    Beta_norm_f= [];
    H_cond_f= [];
    AUC_tst_a= [];
    %% Ejecucion
    % Cantidad de estructuras
    for k=1:K
        k
        W_full = 2*rand(Data_dimension,M_star) - 1;
        b_full = 2*rand(1,M_star) - 1;
        % Cantidad de validaciones por estructura
        for cv=1:CV
            cv
            indices=crossvalind('holdout',y,Test_percent) ;
            x_trn = x(indices,:);
            y_trn = y(indices);
            x_tst = x(~indices,:);
            y_tst = y(~indices);
            
            %% Set up
            N_ejemplos_trn = size(x_trn,1);
            N_ejemplos_tst = size(x_tst,1);
            
            %% Entrenamiento Forward
            
            for n=1:M_star
                n
                H_trn=x_trn*W_full(:,1:n) + b_full(:,1:n);
                % Genero H trn
                H_trn_full = (1 - exp(-H_trn))./(1 + exp(-H_trn)); % (funcion sigmoidea)
                % Entrenamiento
                Beta = pinv(H_trn_full) * y_trn;
                % Genero H _tst
                H_tst=x_tst*W_full(:,1:n) + b_full(:,1:n);
                H_tst_full = (1 - exp(-H_tst))./(1 + exp(-H_tst)); % (funcion sigmoidea)
                
                y_trn_pred = H_trn_full * Beta;
                y_trn_pred = sign(y_trn_pred);
                y_tst_pred = H_tst_full * Beta;
                y_tst_pred = sign(y_tst_pred);
                
                % Calculo el error
                tst_err_acc_f(n)=(sum(prod(y_tst_pred==y_tst, 2)) / size(y_tst, 1))*100;
                trn_err_acc_f(n)=(sum(prod(y_trn_pred==y_trn, 2)) / size(y_trn, 1))*100;
                
            end
            
            % Acumulación de resultados
            Acc_trn_f=[Acc_trn_f ; trn_err_acc_f ];
            Acc_tst_f=[Acc_tst_f ; tst_err_acc_f];
            
            
            %% Entrenamiento con pruning
            % Genero H trn
            H_trn=x_trn*W_full + b_full(ones(1,N_ejemplos_trn),:);
            H_trn_full = (1 - exp(-H_trn))./(1 + exp(-H_trn)); % (funcion sigmoidea)
            % Entrenamiento
            Beta = pinv(H_trn_full) * y_trn;
            
            % Genero H _tst
            H_tst=x_tst*W_full + b_full(ones(1,N_ejemplos_tst),:);
            H_tst_full = (1 - exp(-H_tst))./(1 + exp(-H_tst)); % (funcion sigmoidea)
            
            % Rankin de las neuronas
            v_rank_B= abs(Beta)'/norm(Beta);
            [Ranked_nodes, index_best_full ] = sort(v_rank_B, 'descend');
            
            
            
            %% Eliminacion de neuronas
            N_Neurons=size(Beta,1);
            for i=1:N_Neurons
                i
                l=N_Neurons+1-i;
                
                % Elimino una neurona
                index_best=index_best_full(1:l);
                
                % Elimino el B y el H correspondiente
                B_pruning =Beta(index_best);
                H_trn =H_trn_full(:,index_best);
                H_tst =H_tst_full(:,index_best);
                
                % Genero la prediccin
                y_trn_pred = H_trn * B_pruning;
                y_trn_pred = sign(y_trn_pred);
                y_tst_pred = H_tst * B_pruning;
                y_tst_pred = sign(y_tst_pred);
                
                % Calculo el error
                tst_err_acc_a(l)=(sum(prod(y_tst_pred==y_tst, 2)) / size(y_tst, 1))*100;
                trn_err_acc_a(l)=(sum(prod(y_trn_pred==y_trn, 2)) / size(y_trn, 1))*100;
                
                
            end
            % Apilo para generar estadisticos
            Acc_trn_a=[Acc_trn_a ; trn_err_acc_a];
            Acc_tst_a=[Acc_tst_a ; tst_err_acc_a];
            
            
        end
        Results.Acc.a.trn=Acc_trn_a;
        Results.Acc.a.tst= Acc_tst_a;
        Results.Acc.f.trn=Acc_trn_f;
        Results.Acc.f.tst= Acc_tst_f;
        
    end
    
    General_results{n_e}=Results ;
    clear Results
    
end



%% Graphic
N_steps=4;
Neuronas_finales=1:1:500;
for n_e=1:N_steps
    
    
    if n_e==2
    else
        Aux_strct=  General_results{n_e};
        
        
        Acc_trn_f= Aux_strct.Acc.f.trn;
        Acc_tst_f= Aux_strct.Acc.f.tst;
        
        
        grid on
         alpha=0.3;
        if n_e==1
            
            F=Neuronas_finales;
            acolor=[0.3,0,0.3]+[0.2,0.2,0.2];
            stdshade3(Acc_tst_f,alpha,acolor,F);hold on
        end
        if n_e==2
        
        F=Neuronas_finales;
        acolor=[0,0.245*n_e,n_e*0.25];
        stdshade3(Acc_tst_f,alpha,acolor,F);hold on
        end
        
        
        if n_e==3
            
            F=Neuronas_finales;
            acolor=[0,0.2,0.5]+[0.2,0.2,0.2];
            stdshade3(Acc_tst_f,alpha,acolor,F);hold on
        end
        
        if n_e==4
            
            F=Neuronas_finales;
            acolor=[0,0.5,0.2]+[0.2,0.2,0.2];
            stdshade3(Acc_tst_f,alpha,acolor,F);hold on
        end
        
        
        
        
    end
end


grid on
axis ([min(Neuronas_finales) max(Neuronas_finales) 50 95])

Font_size= 15;
set(gca,'FontSize',13)
xlabel('Number of Hidden Nodes M','FontSize',Font_size)
ylabel('Test AUC','FontSize',Font_size)
aux = get(gca,'Children');
lgd= legend([aux(1),aux(3),aux(5)],'TF=32','TF=8','TF=0','Location','southeast');
lgd.FontSize = Font_size;
ax= gca;
ax.Position= [0.1 0.1100 0.87 0.85];


%% Save plot
%- Figure size in centimeters
%- (it can be included in the paper without stretching):
width = 21;
height = 12;

%- Figure name:
filename = 'trash_features.pdf';

%- Activate removal of margins (off by default)
remove_margin = false;

print_figure (filename, width, height, 'RemoveMargin', remove_margin,'FontSize',Font_size)


