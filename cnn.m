% zdjęcia w zakresie szarości 64x64
% wykorzystane wszytkie zdjęcia z bazy: 
%449- zmiany łagodne, 208 - zmiany złośliwe

outputFolder=fullfile('Baza');
Datasetpath= fullfile(outputFolder, 'Guzy');
Data = imageDatastore(Datasetpath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames')

tbl = countEachLabel(Data);
minSetCount = min(tbl{:,2})
% same count of images
Data=splitEachLabel(Data, minSetCount, 'randomize');
countEachLabel(Data);

imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-40 40], ...
    'RandXTranslation',[-8 8], ...
    'RandYTranslation',[-8 8])

[DataTrain,DataValidation,DataTest] = splitEachLabel(Data,0.7,0.15, 'randomize')% 70%-trening, 15%-validacja, 15%-test
% [DataTrain, DataValidation] = splitEachLabel(Data, 0.3, 'randomize')
Traindatasource = augmentedImageSource([64 64 1],DataTrain,'DataAugmentation',imageAugmenter);

minibatch = preview(Traindatasource);
imshow(imtile(minibatch.input));


layers = [ ...
    imageInputLayer([64 64 1])
    convolution2dLayer(3, 8, 'Padding', 'same' ) 
    batchNormalizationLayer 
%   clippedReluLayer(10)
    reluLayer
    
    maxPooling2dLayer(2,'stride',2) 
    
    convolution2dLayer(3, 16,'Padding', 'same' )
    batchNormalizationLayer 
    reluLayer
    
    maxPooling2dLayer(2,'stride',2)
    
    convolution2dLayer(3, 32 ,'Padding', 'same')
    batchNormalizationLayer 
    reluLayer
    
    maxPooling2dLayer(2,'stride',2)
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer]


miniBatchSize = 256
% validationFrequency = floor(numel(DataValidation)/miniBatchSize);
options=trainingOptions('adam', 'MaxEpochs', 25 ,'MiniBatchSize', miniBatchSize,...
    'initialLearnRate',0.001, 'Shuffle', 'every-epoch', ...
    'ValidationData', DataValidation, 'ValidationFrequency', 2, ...
    'Verbose', 1,  'Plots', 'training-progress',...
    'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));

% convnet = trainNetwork(DataTrain, layers, options)
convnet = trainNetwork(Traindatasource, layers, options)

%% Measure network accuracy
load convnet
predLabelsTest = classify(convnet, DataTest);
labelsTest = DataTest.Labels;
accuracy = sum(predLabelsTest == labelsTest) / numel(labelsTest)
analyzeNetwork(convnet)
%% after train
load convnet 
a= imread('3.png'); %zdj poddane klasyfikacji
figure, imshow(a)
net = convnet; 
output = classify(net, a)


 tf1=[]
for  ii=1:2
    st=int2str(ii)
tf = ismember(output,st);
tf1=[tf1 tf];
end
out=find(tf1==1);


if out == 1
    msgbox('Guz ŁAGODNY')
elseif out == 2
    msgbox('Guz ZŁOŚLIWY')
end

    