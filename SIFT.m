% EE368/CS232
% Homework 6
% Problem: SIFT Keypoints
% Script by David Chen, Huizhong Chen

clc; clear all;
addpath 'vlfeat-0.9.21/toolbox'
vl_setup;
imageFiles = {'pink.jpg'};

SIFTPeakThresh = [10 10];
SIFTEdgeThresh = [5 10];
for nImage = 1:length(imageFiles)
    % Load original image
    imgOrig = rgb2gray( imread(imageFiles{nImage}) );
    [height, width] = size(imgOrig);
    x_center = width/2;
    y_center = height/2;
    % Extract SIFT keypoints on original image
    peakThresh = 5;
    edgeThresh = 10;
    [fc, dc] = vl_sift(single(imgOrig), ...
        'PeakThresh', peakThresh, 'EdgeThresh', edgeThresh);
    xc = fc(1,:);
    yc = fc(2,:);
    numFeatures = size(fc,2);
    disp(['Features = ' num2str(numFeatures)]);
    % Show original image with Harris corners overlaid
    figure(1); clf;
    imshow(imgOrig); hold on;
    h1 = vl_plotframe(fc);
    h2 = vl_plotframe(fc);
    set(h1, 'Color', 'k', 'linewidth', 3);
    set(h2, 'Color', 'y', 'linewidth', 2);
    title(['Original Image: ' imageFiles{nImage}]);
    pause;
    % Rotate the image incrementally
    rotAngles = 0:20:360;
    numFeatureMatches = zeros(1, numel(rotAngles));
    for nAngle = 1:length(rotAngles)
        % Rotate
        rotAngle = rotAngles(nAngle);
        imgRot = imrotate(imgOrig, rotAngle, 'bicubic');
        [heightRot, widthRot] = size(imgRot);
        x_center_rot = widthRot/2;
        y_center_rot = heightRot/2;
        % Find SIFT keypoints on rotated image
        [fc_rot_actual, dc_rot_actual] = vl_sift(single(imgRot), ...
            'PeakThresh', peakThresh, 'EdgeThresh', edgeThresh);
        numFeaturesRot = size(fc_rot_actual,2);
        xc_rot_actual = fc_rot_actual(1,:);
        yc_rot_actual = fc_rot_actual(2,:);
        disp(['Angulo = ' num2str(rotAngle) '°, Caracteristicas = ' num2str(numFeaturesRot)]);

        % Find feature matches to original image
        yc_rot_predict = (yc - y_center)*cosd(rotAngle) ...
            - (xc - x_center)*sind(rotAngle) + y_center_rot;
        xc_rot_predict = (yc - y_center)*sind(rotAngle) ...
            + (xc - x_center)*cosd(rotAngle) + x_center_rot;
        for nOrig = 1:length(xc_rot_predict)
            matchFound = 0;
            for nTrans = 1:length(xc_rot_actual)
                threshold = 2;
                delta_x = abs(xc_rot_predict(nOrig)-xc_rot_actual(nTrans));
                delta_y = abs(yc_rot_predict(nOrig)-yc_rot_actual(nTrans));
                if ( delta_x <= threshold ) && ( delta_y <= threshold )
                    matchFound = 1;
                    break;
                end
            end % nTrans
            if matchFound == 1
                numFeatureMatches(nAngle) = numFeatureMatches(nAngle) + 1;
            end
        end % nOrig
        disp(['Coincidencias = ' num2str(numFeatureMatches(nAngle))]);
        disp(['Repetibilidad = ' num2str(numFeatureMatches(nAngle)*100/numFeatureMatches(1))]);
        disp(['__________________________________________________']);
        figure(2); clf;
        imshow(imgRot); hold on;
        h1 = vl_plotframe(fc_rot_actual);
        h2 = vl_plotframe(fc_rot_actual);
        set(h1, 'Color', 'k', 'linewidth', 3);
        set(h2, 'Color', 'g', 'linewidth', 2);
        fc(1,:) = xc_rot_predict;
        fc(2,:) = yc_rot_predict;
        h3 = vl_plotframe(fc);
        h4 = vl_plotframe(fc);
        set(h3, 'Color', 'k', 'linewidth', 3);
        set(h4, 'Color', 'y', 'linewidth', 2);
        print(['SIFT-rot' num2str(nAngle)],'-dpng');
    end % nAngle
    pause;
    % Plot repeatability against angle
    figure(3); clf; set(gcf, 'Position', [50 50 400 300]);
    h = plot(rotAngles, numFeatureMatches ./ numFeatures, 'b-o'); grid on;
    set(h, 'MarkerFaceColor', 'b');
    xlabel('Anguo de rotación'); ylabel('Repetibilidad');
    title(['Robustes a Rotación: ' imageFiles{nImage}]);
    axis([0 360 0 1]);
    
    % Rescale the image incrementally
    scaleFactors = 1.2 .^ (0:1:8);
    numFeatureMatches = zeros(1, numel(scaleFactors));
    for nScale = 1:length(scaleFactors)
        % Rescale
        scaleFactor = scaleFactors(nScale);
        imgScale = imresize(imgOrig, scaleFactor, 'bicubic');
        % Find SIFT keypoints
        [fc_scale_actual, dc_scale_actual] = vl_sift(single(imgScale), ...
            'PeakThresh', peakThresh, 'EdgeThresh', edgeThresh);
        numFeaturesScale = size(fc_scale_actual,2);
        xc_scale_actual = fc_scale_actual(1,:);
        yc_scale_actual = fc_scale_actual(2,:);
        disp(['Factor de escalado = ' num2str(scaleFactor) ...
            ', Caracteristicas = ' num2str(numFeaturesScale)]);
        % Find feature matches to original image
        yc_scale_predict = yc * scaleFactor;
        xc_scale_predict = xc * scaleFactor;
        for nOrig = 1:length(xc_scale_predict)
            matchFound = 0;
            for nTrans = 1:length(xc_scale_actual)
                threshold = 2;
                delta_x = abs(xc_scale_predict(nOrig)-xc_scale_actual(nTrans));
                delta_y = abs(yc_scale_predict(nOrig)-yc_scale_actual(nTrans));
                if ( delta_x <= threshold ) && ( delta_y <= threshold )
                    matchFound = 1;
                    break;
                end
            end % nTrans
            if matchFound == 1
                numFeatureMatches(nScale) = numFeatureMatches(nScale) + 1;
            end
        end % nOrig
        disp(['Coincidencias = ' num2str(numFeatureMatches(nScale))]);
        disp(['Repetibilidad = ' num2str(numFeatureMatches(nScale)*100/numFeatureMatches(1))]);
        disp(['__________________________________________________']);
         figure(2); clf;
        imshow(imgScale); hold on;
        h1 = vl_plotframe(fc_scale_actual);
        h2 = vl_plotframe(fc_scale_actual);
        set(h1, 'Color', 'k', 'linewidth', 3);
        set(h2, 'Color', 'g', 'linewidth', 2);
        fc(1,:) = xc_scale_predict;
        fc(2,:) = yc_scale_predict;
        h3 = vl_plotframe(fc);
        h4 = vl_plotframe(fc);
        set(h3, 'Color', 'k', 'linewidth', 3);
        set(h4, 'Color', 'y', 'linewidth', 2);
        print(['SIFT-esc' num2str(nScale)],'-dpng');
    end % nAngle
    % Plot repeatability against rescaling factorà
    figure(4); clf; set(gcf, 'Position', [100 100 400 300]);
    h = plot(log10(scaleFactors), numFeatureMatches ./ numFeatures, 'b-o'); grid on;
    set(h, 'MarkerFaceColor', 'b');
    xlabel('Factor de escalado'); ylabel('Repetibilidad');
    title(['Robustes a Escalado: ' imageFiles{nImage}]);
    set(gca, 'XTick', log10(scaleFactors));
    labels = cell(1, length(scaleFactors));
    for nLabel = 1:length(scaleFactors)
        labels{nLabel} = num2str(scaleFactors(nLabel), '%.2f');
    end % nLabel
    set(gca, 'XTickLabel', labels);
    axis([min(log10(scaleFactors)) max(log10(scaleFactors)) 0 1]);
    
    if nImage == 1
        pause;
    end
    
end % nImage