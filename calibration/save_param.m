cameraMatrix1 = stereoParams.CameraParameters1.IntrinsicMatrix';
cameraMatrix2 = stereoParams.CameraParameters2.IntrinsicMatrix';
distCoeffs1 = [stereoParams.CameraParameters1.RadialDistortion, stereoParams.CameraParameters1.TangentialDistortion];
distCoeffs2 = [stereoParams.CameraParameters2.RadialDistortion, stereoParams.CameraParameters2.TangentialDistortion];
% k_3 = 0
distCoeffs1(5) = 0;
distCoeffs2(5) = 0;
R = stereoParams.RotationOfCamera2';
T = stereoParams.TranslationOfCamera2';

fileID = fopen('calib_param.yml', 'w');
fprintf(fileID, '%%YAML:1.0\n');
fprintf(fileID, 'cameraMatrixL: !!opencv-matrix\n');
fprintf(fileID, '   rows: 3\n   cols: 3\n   dt: d\n   data: [%f, %f, %f, %f, %f, %f, %f, %f, %f]\n', cameraMatrix1');
fprintf(fileID, 'distCoeffsL: !!opencv-matrix\n');
fprintf(fileID, '   rows: 1\n   cols: 5\n   dt: d\n   data: [%f, %f, %f, %f, %f]\n', distCoeffs1');
fprintf(fileID, 'cameraMatrixR: !!opencv-matrix\n');
fprintf(fileID, '   rows: 3\n   cols: 3\n   dt: d\n   data: [%f, %f, %f, %f, %f, %f, %f, %f, %f]\n', cameraMatrix2');
fprintf(fileID, 'distCoeffsR: !!opencv-matrix\n');
fprintf(fileID, '   rows: 1\n   cols: 5\n   dt: d\n   data: [%f, %f, %f, %f, %f]\n', distCoeffs2');
fprintf(fileID, 'R: !!opencv-matrix\n');
fprintf(fileID, '   rows: 3\n   cols: 3\n   dt: d\n   data: [%f, %f, %f, %f, %f, %f, %f, %f, %f]\n', R');
fprintf(fileID, 'T: !!opencv-matrix\n');
fprintf(fileID, '   rows: 3\n   cols: 1\n   dt: d\n   data: [%f, %f, %f]\n', T');
fclose(fileID);