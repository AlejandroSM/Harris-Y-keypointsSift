% Harris detector

clear, clc, close all

imageFiles = {'pink.jpg'};
for nImage = 1:length(imageFiles)
    
    % Load image
    img = imread(imageFiles{nImage});
    img = im2double(img);
    imgGray = rgb2gray(img);
    figure(1); clf;
    imshow(img); title('Original Image');
    % Calculate Harris corner-ness
    C = cornermetric(imgGray, 'Harris', 'SensitivityFactor', 0.04);
    figure(2); clf;
    imshow(C,[-2 4]*1e-3); % colorbar;
    title('Harris Cornerness');
    print('HA2','-dpng');
    
    % Find extrema
    C(find(C < 0.01*max(C(:)))) = 0;
    imgExt = imregionalmax(C);
    figure(3); clf;
    imshow(C,[-2 4]*1e-3); title('Local Maxima');
    figure(4); clf;
    imshow(imdilate(imgExt, ones(3,3))); title('Local Maxima (Dilated)');
    
    % Show strongest keypoints
    numCorners = sum(imgExt(:))
    figure(5); clf;
    imshow(img); hold on; title('Harris Keypoints');
    [row,col] = find(imgExt == 1);
    responses = zeros(1,numel(row));
    for n = 1:numel(row)
        responses(n) = C(row(n), col(n));
    end % n
    [responses, sortIdx] = sort(responses, 'descend');
    for n = 1:70 % numel(row)
        h = plot(col(sortIdx(n)), row(sortIdx(n)), 'y+');
        set(h, 'MarkerSize', 3, 'MarkerFaceColor', 'y');
    end % n
    print('HA5','-dpng');
    pause;
    
end % nImage

pause;

imageFiles = {'pink.jpg'};
for nImage = 1:length(imageFiles)
    % Load original image
    
    imgOrig = rgb2gray( imread(imageFiles{nImage}) );
    [height, width] = size(imgOrig);
    x_center = width/2;
    y_center = height/2;
    % Extract Harris corners on original image
    [xc, yc] = harris_corners(imgOrig);
    numFeatures = length(xc);
    disp(['Features = ' num2str(numFeatures)]);
    % Show original image with Harris corners overlaid
    figure(1); clf;
    warning off; imshow(imgOrig); warning on; hold on;
    plot(xc, yc, 'yo'); title(['Original Image: ' imageFiles{nImage}]);
    % Rotate the image incrementally
    rotAngles = 0:20:360;
    numFeatureMatches = zeros(1, numel(rotAngles));
    for nAngle = 1:length(rotAngles)
        % Rotate
        rotAngle = rotAngles(nAngle);
        disp(['Ángulo = ' num2str(rotAngle) '°']);
        imgRot = imrotate(imgOrig, rotAngle, 'bicubic');
        % imwrite(imgRot, ['HRt_' num2str(rotAngle) '.PNG'] );
        [heightRot, widthRot] = size(imgRot);
        x_center_rot = widthRot/2;
        y_center_rot = heightRot/2;
        % Find Harris corners
        [xc_rot_actual, yc_rot_actual] = harris_corners(imgRot);
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
        disp(['Caracteristicas = ' num2str(length(xc_rot_actual))]);
        disp(['Conicidencias = ' num2str(numFeatureMatches(nAngle))]);
        disp(['Repetibilidad = ' num2str((numFeatureMatches(nAngle)/numFeatureMatches(1))*100) '%']);
        disp(['_______________________________________________________________'])
        figure(6); clf; warning off; imshow(imgRot); warning on; hold on;
        plot(xc_rot_actual, yc_rot_actual, 'g+');
        plot(xc_rot_predict, yc_rot_predict, 'yo');
        plot(xc_rot_predict(1), yc_rot_predict(1), 'ro');
        print(['rot' num2str(nAngle)],'-dpng');
    end % nAngle
    
    pause;
    
    % Plot repeatability against angle
    figure(3); clf; set(gcf, 'Position', [50 50 400 300]);
    h = plot(rotAngles, numFeatureMatches ./ numFeatures, 'b-o'); grid on;
    set(h, 'MarkerFaceColor', 'b');
    xlabel('Rotation Angle (degrees)'); ylabel('Repeatability');
    title(['Robustness to Rotation: ' imageFiles{nImage}]);
    axis([0 360 0 1]);
    % Rescale the image incrementally
    scaleFactors = 1.2 .^ (0:1:8);
    numFeatureMatches = zeros(1, numel(scaleFactors));
    for nScale = 1:length(scaleFactors)
        % Rescale
        scaleFactor = scaleFactors(nScale);
        disp(['Scaling Factor = ' num2str(scaleFactor)]);
        imgScale = imresize(imgOrig, scaleFactor, 'bicubic');
        % Find Harris corners
        [xc_scale_actual, yc_scale_actual] = harris_corners(imgScale);
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
        disp(['Caracteristicas = ' num2str(length(xc_scale_actual))]);
        disp(['Coincidencias = ' num2str(numFeatureMatches(nScale))]);
        disp(['Repetibilidad = ' num2str((numFeatureMatches(nScale)/numFeatureMatches(1))*100) '%']);
        disp(['_______________________________________________________________']);
        figure(6); clf; warning off; imshow(imgScale); warning on; hold on;
        plot(xc_scale_actual, yc_scale_actual, 'g+');
        plot(xc_scale_predict, yc_scale_predict, 'yo');
        print(['esc' num2str(nScale)],'-dpng');
    end % nAngle
    
    
    % Plot repeatability against rescaling factor
    figure(4); clf; set(gcf, 'Position', [100 100 400 300]);
    h = plot(log10(scaleFactors), numFeatureMatches ./ numFeatures, 'b-o'); grid on;
    set(h, 'MarkerFaceColor', 'b');
    xlabel('Rescaling Factor'); ylabel('Repeatability');
    title(['Robustness to Scaling: ' imageFiles{nImage}]);
    set(gca, 'XTick', log10(scaleFactors));
    labels = cell(1, length(scaleFactors));
    for nLabel = 1:length(scaleFactors)
        labels{nLabel} = num2str(scaleFactors(nLabel), '%.2f');
    end % nLabel
    set(gca, 'XTickLabel', labels);
    axis([min(log10(scaleFactors)) max(log10(scaleFactors)) 0 1]);
    
    pause;
    
end % nImage

pause;
function [corner_idx_x, corner_idx_y] = harris_corners(img)
[height, width] = size(img);
C = cornermetric(im2double(img), 'Harris', 'SensitivityFactor', 0.031);
threshold = 1e-4;
C(find(C < threshold)) = 0;
corner_peaks = imregionalmax(C);
corner_idx = find(corner_peaks == true);
[corner_idx_y, corner_idx_x] = ind2sub([height width], corner_idx);
end