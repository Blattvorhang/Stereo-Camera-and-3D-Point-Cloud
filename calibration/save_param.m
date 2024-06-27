K1 = stereoParams.CameraParameters1.IntrinsicMatrix';
K2 = stereoParams.CameraParameters2.IntrinsicMatrix';

RD1 = stereoParams.CameraParameters1.RadialDistortion;
TD1 = stereoParams.CameraParameters1.TangentialDistortion;
distCoeffs1 = [RD1(1), RD1(2), TD1(1), TD1(2), 0];

RD2 = stereoParams.CameraParameters2.RadialDistortion;
TD2 = stereoParams.CameraParameters2.TangentialDistortion;
distCoeffs2 = [RD2(1), RD2(2), TD2(1), TD2(2), 0];

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