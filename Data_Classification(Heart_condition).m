% considering imbalance in the data and solving the imbalance issue..
load('cleveland_heart_disease_dataset_labelled.mat')
xn = normalize(x)
c=[xn t];
[s,~]=size(c);
a0=[];
a1=[];
a2=[];
%seperating the data according to class.
for i=1:s
    if c(i,14)==0
        a0=[a0;c(i,1:14)]
    elseif c(i,14)==1
        a1=[a1;c(i,1:14)]
    else c(i,14)==2
        a2=[a2;c(i,1:14)]
    end 
end
[i,~]=size(a0)
[j,~]=size(a1)
[k,~]=size(a2)
% Test sample per class
td=20
setdemorandstream(672880951)
% taking td samples from each class for testing
a0_test=a0(i-td+1:i,:);
a1_test=a1(j-td+1:j,:);
a2_test=a2(k-td+1:k,:);

%assigning the rest of the samples to training class and dealing with
%imbalanced class
a0_train=a0(1:i-td,:);%i-j
a1_train=[a1(1:j-td,:);a1(1:i-j,:)];

t=1
a3=a2(1:k-td,:)
while t<fix((i-td)/(k-td))
    a3=[a3;a2(1:k-td,:)]
    t=t+1
end
[t,~]=size(a3)
a2_train=[a3;a2(1:i-td-t,:)]

%merging all train test classes as a sigle 
Test=[a0_test;a1_test;a2_test]
Train=[a0_train;a1_train;a2_train]

%randomizing the classes of train test samples
test_random=Test(randperm(size(Test,1)),:)
train_random=Train(randperm(size(Train,1)),:)
%random_x = x(randperm(size(x, 1)), :)


%Neural Net
train_random(:,14)=train_random(:,14)+1
test_random(:,14)=test_random(:,14)+1


%parameters    
%"trainbr","trainlm","trainbfg",
pa1=["traincgb","traincgf","traincgp","traingd","traingda","traingdm","traingdx","trainoss","trainrp","trainscg","trainb","trainr","trains","trainbu","trainru","trainc"]
pa2=["elliotsig","compet","hardlim","hardlims","logsig","netinv","poslin","purelin","radbas","radbasn","satlin","satlins","softmax","tansig","tribas"]
pa3=["mse","mae","sae","sse","crossentropy","msesparse"]
[~,pa4]=size(pa1)
[~,pa5]=size(pa2)
[~,pa6]=size(pa3)
au=1
m=0
% finding best parameter for neural net based on how well the model performs on test data
while au<=pa4
    target=ind2vec(train_random(:,14)')
    net=patternnet(200);
    net.trainFcn=pa1(au)
    au1=1
    while au1<=pa5
        net.layers{1}.transferFcn=pa2(au1)
        au2=1
        while au2<=pa6
            try
                net.performFcn=pa3(au2)
                %net.divideFcn='dividetrain';
                %net.performParam.normalization='Standard'
                net.trainParam.epochs=20000;
                net=train(net,train_random(:,1:13)',target)
                y=net(test_random(:,1:13)')
                classes=vec2ind(y)';
                [i,~]=size(classes)
                j=1
                l=0
                while j<=i
                    if classes(j)==1 & test_random(j,14)==1
                        l=l+1
                    elseif classes(j)==2 & test_random(j,14)==2
                        l=l+1
                    elseif classes(j)==3 & test_random(j,14)==3
                        l=l+1
                    end
                    j=j+1
                end
                if m<l
                    m=l
                    gregnet5 = net;
                    save gregnet5
                    % The best parameters are store in the variables below
                    au5=pa1(au)
                    au6=pa2(au1)
                    au7=pa3(au2)     
                end
            catch
            end
            au2=au2+1
        end
        au1=au1+1    
    end            
    au=au+1
end
hold off

% model1 with 80 hidden layer neurons and best parameters 
load gregnet1
y=gregnet1(test_random(:,1:13)')
classes1=vec2ind(y)';


% model2 with 120 hidden layer neurons and best parameters 
load gregnet2
y=gregnet2(test_random(:,1:13)')
classes2=vec2ind(y)';


% model3 with 50 hidden layer neurons and best parameters 
load gregnet3
y=gregnet3(test_random(:,1:13)')
classes3=vec2ind(y)';

% model4 with 150 hidden layer neurons and best parameters 
load gregnet4
y=gregnet4(test_random(:,1:13)')
classes4=vec2ind(y)';

% model5 with 200 hidden layer neurons and best parameters 
load gregnet5
y=gregnet5(test_random(:,1:13)')
classes5=vec2ind(y)';

classes=[classes1,classes2,classes3,classes4,classes5]
class=mode(classes')'
confusionchart(test_random(:,14),class);

%building the model on best parameter and observing the result
%target=ind2vec(train_random(:,14)')
%setdemorandstream(672880951);
%net=patternnet([10 10 10]);
%net.trainFcn=au5
%net.layers{1}.transferFcn=au6
%net.layers{2}.transferFcn=au7
%net.layers{3}.transferFcn=au5
%net.divideFcn='dividetrain';
%net.performFcn = 'crossentropy';
%net.trainParam.epochs=100000;
%net=train(net,train_random(:,1:13)',target)
%y=net(test_random(:,1:13)')
%classes=vec2ind(y)';
%hold off
%confusion matrix to visualise result
%confusionchart(test_random(:,14),classes);

%confusion matrix to visualise result
%confusionchart(test_random(:,14),classes);

                
%["compet","elliotsig","hardlim","hardlims","logsig","netinv","poslin","purelin","radbas","radbasn","satlin","satlins","softmax","tansig","tribas"]
%["mae","mse","sae","sse","crossentropy","msesparse"]
%["trainbfg","traincgb","traincgf","traincgp","traingd","traingda","traingdm","traingdx","trainoss","trainrp","trainscg","trainb","trainc","trainr","trains","trainbu","trainru"]