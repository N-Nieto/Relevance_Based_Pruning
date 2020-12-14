% Traditional ELM with junk Features 
% clear;clc; close all;
%% Set Up
M_star=500;
K=5;
CV=5;
Test_percent=0.9;
% Synthetic data contains 2 representative features an 100 unrepresentative
features = 70 ;

% % Load Synthetic Data 
% unrepresentative features
load ('Synthetic_data.mat');
x=X;
y=Y;
x = x(:,1:features);    

%% Main Loop
% For diferents initializations
Acc_trn = [];
Acc_tst = [];

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

        %% Traditional ELM
        for n = 1:M_star
            
            % Generate H train
            H_trn = x_trn*W_full(:,1:n) + b_full(:,1:n);
            % Sigmoid
            H_trn_full = (1 - exp(-H_trn))./(1 + exp(-H_trn)); 
            % Train step - pseudoinverse
            Beta = pinv(H_trn_full) * y_trn;
            
            % Generate H test
            H_tst=x_tst*W_full(:,1:n) + b_full(:,1:n);
            % Sigmoid
            H_tst_full = (1 - exp(-H_tst))./(1 + exp(-H_tst)); 

            % Make a prediction
            y_trn_pred = H_trn_full * Beta;
            y_trn_pred = sign(y_trn_pred);
            y_tst_pred = H_tst_full * Beta;
            y_tst_pred = sign(y_tst_pred);

            % Calculate the acc
            tst_err_acc(n)=(sum(prod(y_tst_pred==y_tst, 2)) / size(y_tst, 1))*100;
            trn_err_acc(n)=(sum(prod(y_trn_pred==y_trn, 2)) / size(y_trn, 1))*100;

        end
        % Store the results for every CV
        Acc_trn=[Acc_trn ; trn_err_acc ];
        Acc_tst=[Acc_tst ; tst_err_acc];

    end
    % Store the results for every K
    Results.Acc.trn=Acc_trn;
    Results.Acc.tst= Acc_tst;
end

% Plot
plot(mean(Results.Acc.tst))


