a= imread('1.png'); %zdj poddane klasyfikacji
figure, imshow(a)
outputFolder=fullfile('Baza');
Datasetpath= fullfile(outputFolder, 'Guzy');
Data = imageDatastore(Datasetpath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames')


tbl = countEachLabel(Data); %%liczy ile jest zdjęć w danej kategorii, mogą być różne wartości czego nie chcemy
minSetCount = min(tbl{:,2}) %% druga kolumna
% same count of images
Data=splitEachLabel(Data, minSetCount, 'randomize');
countEachLabel(Data);

imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-10 10], ...
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3])


imageSize=[64 64 1];

%  numTrainFiles = 79; 
[DataTrain,DataValidation,DataTest] = splitEachLabel(Data,0.7,0.15, 'randomize')
% [DataTrain, DataValidation] = splitEachLabel(Data, 0.3, 'randomize')
Traindatasource = augmentedImageSource([64 64 1],DataTrain,'DataAugmentation',imageAugmenter);

minibatch = preview(Traindatasource);
imshow(imtile(minibatch.input));

% figure;
% perm = randperm(100, 20);
% for i = 1:20
%     subplot(5,4,i);
%     imshow(Data.Files{perm(i)});
% end
% img = readimage(Data,1);
% size(img)





layers = [ ...
    imageInputLayer([64 64 1])
    convolution2dLayer(6, 8, 'Padding', 'same' ) 
    batchNormalizationLayer 
    reluLayer %f.aktywacji
    
    maxPooling2dLayer(2,'stride',2) 
    
    convolution2dLayer(6, 6 ,'Padding', 'same')
    batchNormalizationLayer 
    reluLayer
    
    maxPooling2dLayer(2,'stride',2)
    
    convolution2dLayer(6, 4 ,'Padding', 'same')
    batchNormalizationLayer 
    reluLayer
    
    maxPooling2dLayer(2,'stride',2)
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer]


miniBatchSize = 64
% validationFrequency = floor(numel(DataValidation)/miniBatchSize);
options=trainingOptions('adam', 'MaxEpochs', 20 ,'MiniBatchSize', miniBatchSize,...
    'initialLearnRate',0.001, 'Shuffle', 'every-epoch', ...
    'ValidationData', DataValidation, 'ValidationFrequency', 2, ...
    'Verbose', 1,  'Plots', 'training-progress',...
    'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));

%  'ValidationPatience',5,
% 
% convnet = trainNetwork(DataTrain, layers, options)
convnet = trainNetwork(Traindatasource, layers, options)

% YPred = classify(convnet,DataValidation);
% YValidation = DataValidation.Labels;
% accuracy = sum(YPred == YValidation)/numel(YValidation)
% output = classify(convnet, a);

predLabelsTest = classify(convnet, DataTest);
labelsTest = DataTest.Labels;
accuracy = sum(predLabelsTest == labelsTest) / numel(labelsTest)
% output = classify(predLabelsTest, a); 

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

    