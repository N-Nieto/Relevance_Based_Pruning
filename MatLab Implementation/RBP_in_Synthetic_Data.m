% Traditional ELM with junk Features
% clear;clc; close all;
%% Set Up
M_star=500;
K=5;
CV=2;
Test_percent=0.9;
% Synthetic data contains 2 representative features an 100 unrepresentative
features = 100 ;

% % Load Synthetic Data 
% unrepresentative features
load ('Synthetic_data.mat');
x=X;
y=Y;
x = x(:,1:features);

%% Main Loop
% For diferents initializations
Acc_trn=[];
Acc_tst=[];
for k = 1:K

    % Create the netwotk
    W_full = 2*rand(features,M_star) - 1;
    b_full = 2*rand(1,M_star) - 1;

    % Validations made for every initialization
    for cv = 1:CV

        index = crossvalind('holdout',y,Test_percent) ;
        x_trn = x(index,:);
        y_trn = y(index);
        x_tst = x(~index,:);
        y_tst = y(~index);

        %% Set up
        N_ej_trn = size(x_trn,1);
        N_ej_tst = size(x_tst,1);

        %% RBP
        % Generate the Full Train H
        H_trn= x_trn * W_full + b_full(ones(1,N_ej_trn),:);
        % Sigmoid
        H_trn_full = (1 - exp(-H_trn))./(1 + exp(-H_trn)); 
        % Training step
        Beta = pinv(H_trn_full) * y_trn;

        % Generate the test H
        H_tst=x_tst*W_full + b_full(ones(1,N_ej_tst),:);
        % Sigmoid
        H_tst_full = (1 - exp(-H_tst))./(1 + exp(-H_tst)); 

        % Ranking neurons
        v_rank_B= abs(Beta)'/norm(Beta);
        [Ranked_nodes, index_best_full ] = sort(v_rank_B, 'descend');


        %% Pruning
        N_Neurons=size(Beta,1);
        for i=1:N_Neurons

            l=N_Neurons+1-i;

            % Select the best index
            index_best=index_best_full(1:l);

            % Prun B and H
            B_pruning =Beta(index_best);
            H_trn =H_trn_full(:,index_best);
            H_tst =H_tst_full(:,index_best);

            % Make a prediction
            y_trn_pred = H_trn * B_pruning;
            y_trn_pred = sign(y_trn_pred);
            y_tst_pred = H_tst * B_pruning;
            y_tst_pred = sign(y_tst_pred);

            % Calculate the accuracy
            tst_err_acc(l)=(sum(prod(y_tst_pred==y_tst, 2)) / size(y_tst, 1))*100;
            trn_err_acc(l)=(sum(prod(y_trn_pred==y_trn, 2)) / size(y_trn, 1))*100;

        end
        % Store the results
        Acc_trn=[Acc_trn ; tst_err_acc ];
        Acc_tst=[Acc_tst ; tst_err_acc];
    end
    % Store the results
    Results.Acc.trn=Acc_trn;
    Results.Acc.tst= Acc_tst;
end

plot(mean(Results.Acc.tst))
