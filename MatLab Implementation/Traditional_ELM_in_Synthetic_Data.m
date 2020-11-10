% Traditional ELM with junk Features
% clear;clc; close all;
%% Set Up
M_star=500;
K=30;
CV=10;
Test_percent=0.9;
N_initial_trash_features=2;

% Times to add Junk Features
N_steps=4;
General_results=cell(1,N_steps);

% % Load Synthetic Data
load ('Synthetic_data.mat');
x=X;
y=Y;
N_ej= size(x,1);

for n_e=1:N_steps
    
    % Add Junk Features
    if n_e==1
        N_initial_trash_features0=0;
        N_rand=rand(N_ej,N_initial_trash_features0);
    else
        N_initial_trash_features=N_initial_trash_features*4;
        N_rand=rand(N_ej,N_initial_trash_features);
    end
    x=[x N_rand];
    
    %% Initializations
    Data_dimension=size(x,2);
    % For RBP
    reg_trn_a=[];
    reg_tst_a=[];
    Acc_trn_a=[];
    Acc_tst_a=[];
    Beta_min_a=[];
    Beta_norm_a =[];
    H_cond_a= [];
    AUC_tst_f=[];
    
    % For Traditional ELM
    reg_trn_f=[];
    reg_tst_f=[];
    Acc_trn_f=[];
    Acc_tst_f=[];
    Beta_min_f= [];
    Beta_norm_f= [];
    H_cond_f= [];
    AUC_tst_a= [];
    
    %% Main Loop
    % For diferents initializations
    for k=1:K
        
        % Create the netwotk
        W_full = 2*rand(Data_dimension,M_star) - 1;
        b_full = 2*rand(1,M_star) - 1;
        
        % Validations made for every initialization
        for cv=1:CV
            
            index=crossvalind('holdout',y,Test_percent) ;
            x_trn = x(index,:);
            y_trn = y(index);
            x_tst = x(~index,:);
            y_tst = y(~index);
            
            %% Set up
            N_ej_trn = size(x_trn,1);
            N_ej_tst = size(x_tst,1);
            
            %% Traditional ELM
            
            for n=1:M_star

                H_trn=x_trn*W_full(:,1:n) + b_full(:,1:n);
                % Genero H trn
                % Sigmoid
                H_trn_full = (1 - exp(-H_trn))./(1 + exp(-H_trn)); 
                % Train step
                Beta = pinv(H_trn_full) * y_trn;

                H_tst=x_tst*W_full(:,1:n) + b_full(:,1:n);
                % Sigmoid
                H_tst_full = (1 - exp(-H_tst))./(1 + exp(-H_tst)); 
                
                % Make a prediction
                y_trn_pred = H_trn_full * Beta;
                y_trn_pred = sign(y_trn_pred);
                y_tst_pred = H_tst_full * Beta;
                y_tst_pred = sign(y_tst_pred);
                
                % Calculate the acc
                tst_err_acc_f(n)=(sum(prod(y_tst_pred==y_tst, 2)) / size(y_tst, 1))*100;
                trn_err_acc_f(n)=(sum(prod(y_trn_pred==y_trn, 2)) / size(y_trn, 1))*100;
                
            end
            
            % Store the results
            Acc_trn_f=[Acc_trn_f ; trn_err_acc_f ];
            Acc_tst_f=[Acc_tst_f ; tst_err_acc_f];
            

            
        end
        % Store the results
        Results.Acc.f.trn=Acc_trn_f;
        Results.Acc.f.tst= Acc_tst_f;
        
    end
    % Store the results
    General_results{n_e}=Results ;
    clear Results
    
end