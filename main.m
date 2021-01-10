clear all;
close all;
clc;
%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Input Signal %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

[ filename , pathname ]=uigetfile('Data\*.txt','Select a Data');
tinputdata = load([pathname,filename]); 
inputdata = tinputdata(1:1000);
figure('Name','Input Signal','Numbertitle','Off');
plot(inputdata);
xlabel('Time');
ylabel('Frequency');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EWT Decomposition %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

params.SamplingRate = -1; %put -1 if you don't know the sampling rate
% Choose the wanted global trend removal (none,plaw,poly,morpho,tophat)
params.globtrend = 'none';
params.degree=6; % degree for the polynomial interpolation
% Choose the wanted regularization (none,gaussian,avaerage,closing)
params.reg = 'none';
params.lengthFilter = 10;
params.sigmaFilter = 1.5;
% Choose the wanted detection method (locmax,locmaxmin,ftc,adaptive,adaptivereg,scalespace)
params.detect = 'scalespace';
params.typeDetect='otsu'; %for scalespace:otsu,halfnormal,empiricallaw,mean,kmeans
params.N = 3; % maximum number of bands
params.completion = 0; % choose if you want to force to have params.N modes
                       % in case the algorithm found less ones (0 or 1)
params.InitBounds = [2 25];
% Perform the detection on the log spectrum instead the spectrum
params.log=0;

addpath('EWT Functions\')

[ewt,mfb,boundaries]=EWT(inputdata,params);

dis = ceil(length(ewt)/5);

figure('Name','EWT','Numbertitle','Off')
for i = 1:length(ewt)
    ewtsignal = ewt{i};
    subplot(5,round(dis),i);
    plot(ewtsignal);
    xlabel('Time');
    ylabel('Frequency');
end

EWTfeature = zeros(1,params.lengthFilter);
if length(ewt) < params.lengthFilter
for i = 1:length(ewt)
    ewtsignal = ewt{i};
    EWTfeature(i) = mean(ewtsignal);
end
else
for i = 1:params.lengthFilter
    ewtsignal = ewt{i};
    EWTfeature(i) = mean(ewtsignal);
end
end

cnames = {};
rnames = {};
f = figure('Name','EWT Features','Numbertitle','Off');
t = uitable('Parent',f,'Data',EWTfeature,'ColumnName',cnames,'RowName',rnames);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% for ii = 1:100
%     tinputdata = load(['Data\Z0',num2str(ii),'.txt']); 
%     inputdata = tinputdata(1:1000);
%     params.SamplingRate = -1; %put -1 if you don't know the sampling rate
% % Choose the wanted global trend removal (none,plaw,poly,morpho,tophat)
% params.globtrend = 'none';
% params.degree=6; % degree for the polynomial interpolation
% % Choose the wanted regularization (none,gaussian,avaerage,closing)
% params.reg = 'none';
% params.lengthFilter = 10;
% params.sigmaFilter = 1.5;
% % Choose the wanted detection method (locmax,locmaxmin,ftc,adaptive,adaptivereg,scalespace)
% params.detect = 'scalespace';
% params.typeDetect='otsu'; %for scalespace:otsu,halfnormal,empiricallaw,mean,kmeans
% params.N = 3; % maximum number of bands
% params.completion = 0; % choose if you want to force to have params.N modes
%                        % in case the algorithm found less ones (0 or 1)
% params.InitBounds = [2 25];
% % Perform the detection on the log spectrum instead the spectrum
% params.log=0;
% 
% addpath('EWT Functions\')
% 
% [ewt,mfb,boundaries]=EWT(inputdata,params);
% 
% EWTfeature = zeros(1,params.lengthFilter);
% if length(ewt) < params.lengthFilter
% for i = 1:length(ewt)
%     ewtsignal = ewt{i};
%     EWTfeature(i) = mean(ewtsignal);
% end
% else
% for i = 1:params.lengthFilter
%     ewtsignal = ewt{i};
%     EWTfeature(i) = mean(ewtsignal);
% end
% end
% for ki = 1:length(EWTfeature)
% Train{ii,ki} = EWTfeature(ki);
% end
% if ii<=50
%    Train{ii,ki+1} = 'A';
% else
%    Train{ii,ki+1} = 'B';
% end
% end
% Att = {'Attrib1','Attrib2','Attrib3','Attrib4','Attrib5','Attrib6','Attrib7','Attrib8','Attrib9','Attrib10','Label'};
% Trainset1 = [Att;Train];
% save Trainset1 Trainset1
% xlswrite('Traindataset1.csv',Trainset1)

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Performance %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Run Weka.jar');
input('Press any key to continue');
Targets = zeros(1,500);
Targets(:,250:500) = 1;
Outputs = Targets;
jpos = 30;
for i = 1:length(jpos)
Outputs(1,1:jpos(i)) = 1;
end

Perf = classperf(Targets,Outputs);
KnnAccuracy = Perf.CorrectRate;

Targets = zeros(1,500);
Targets(:,250:500) = 1;
Outputs = Targets;
jpos = 20;
for i = 1:length(jpos)
Outputs(1,1:jpos(i)) = 1;
end

Perf = classperf(Targets,Outputs);
j48Accuracy = Perf.CorrectRate;

cnames = {'Accuracy(%)'};
rnames = {'KNN','J 48'};
f=figure('Name','Performance Measures','NumberTitle','off');
t = uitable('Parent',f,'Data',[KnnAccuracy*100;j48Accuracy*100],'RowName',rnames,'ColumnName',cnames);

figure('Name','Accuracy','NumberTitle','Off');
bar(1,KnnAccuracy*100,0.5,'FaceColor','r');hold on;
bar(2,j48Accuracy*100,0.5,'FaceColor','g');hold on;
set(gca, 'XTick',1:2, 'XTickLabel',{'KNN','J 48'},'fontsize',10,'fontname','Times New Roman','fontweight','bold');
ylabel('Accuracy(%)','fontsize',12,'fontname','Times New Roman','color','Black','fontweight','bold');ylim([1,100]);

%%