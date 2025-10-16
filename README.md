clear; clc; close all;
fs = 250; t = 0:1/fs:5; nSignals = 4;
emotionLabels = {'Happy','Sad','Angry','Relaxed'};
EEG_data = zeros(nSignals, length(t));
for i = 1:nSignals
    alpha = sin(2*pi*10*t);
    beta = sin(2*pi*20*t);
    noise = 0.5*randn(size(t));
    EEG_data(i,:) = (i*0.2)*alpha + (i*0.1)*beta + noise;
end
features = [];
for i = 1:nSignals
    [c,l] = wavedec(EEG_data(i,:), 4, 'db4');
    featVec = [];
    for d = 1:4
        detail = detcoef(c,l,d);
        featVec = [featVec, mean(detail), std(detail)];
    end
    features = [features; featVec];
end
labels = [1; 2; 3; 4]; % 4 emotions
SVMModel = fitcecoc(features, labels);
predicted = predict(SVMModel, features);
figure('Name','Live Robot Emotion Simulator','Color','w');
for i = 1:nSignals
    clf; % clear figure to redraw robot
    drawLiveRobot(predicted(i)); 
    title(['Emotion Detected: ' emotionLabels{predicted(i)}], 'FontSize',14);
    pause(2); % simulate delay (2 seconds per state)
end
disp('Emotion Sequence Completed!');
for i = 1:nSignals
    fprintf('Time %d → Emotion: %s\n', i, emotionLabels{predicted(i)});
end
function drawLiveRobot(emotion)
    hold on; axis equal; axis([-1.5 1.5 -1.8 1.5]); axis off;

    % Head
    rectangle('Position',[-0.2,0.7,0.4,0.4],'Curvature',[1 1],'FaceColor',[0.8 0.9 1]);

    % Body
    plot([0 0],[-0.8 0.7],'k','LineWidth',4);

    % Legs
    plot([0 -0.3],[-0.8 -1.3],'k','LineWidth',4);
    plot([0 0.3],[-0.8 -1.3],'k','LineWidth',4);

    % Arms (change based on emotion)
    switch emotion
        case 1 % Happy – Arms raised
            plot([0 -0.5],[0.5 1.2],'r','LineWidth',4);
            plot([0 0.5],[0.5 1.2],'r','LineWidth',4);
            text(-1.2,1.2,'\leftarrow Happy','FontSize',12,'Color','r');

        case 2 % Sad – Arms down
            plot([0 -0.5],[0.3 -0.4],'b','LineWidth',4);
            plot([0 0.5],[0.3 -0.4],'b','LineWidth',4);
            text(-1.2,1.2,'\leftarrow Sad','FontSize',12,'Color','b');

        case 3 % Angry – Arms crossed
            plot([-0.5 0],[0.3 0],'m','LineWidth',4);
            plot([0 0.5],[0 0.3],'m','LineWidth',4);
            text(-1.2,1.2,'\leftarrow Angry','FontSize',12,'Color','m');

        case 4 % Relaxed – One arm bent
            plot([0 -0.3],[0.3 0.1],'g','LineWidth',4); % hanging
            plot([0 0.2],[0.3 0.0],'g','LineWidth',4);  % bent
            text(-1.2,1.2,'\leftarrow Relaxed','FontSize',12,'Color','g');
    end
end
