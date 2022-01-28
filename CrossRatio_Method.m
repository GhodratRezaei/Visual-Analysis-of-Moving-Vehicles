clear;
close all;
img = imread('354.BMP');
figure; imshow(img);
title('Draw one pairs of parallel segments and press enter')

fprintf('Draw parallel segments\n');
FNT_SZ = 15;
% select a first pair of segments (images of 2 parallel lines)
segment1 = drawline('Color','red');
segment2 = drawline('Color','red');


% select a second pair of segments (images of 2 parallel lines)
% segment3 = drawline('Color','blue');
% segment4 = drawline('Color','blue');

fprintf('Press enter to continue\n');
pause

l1 = segToLine(segment1.Position);
l2 = segToLine(segment2.Position);

% m1 = segToLine(segment3.Position);
% m2 = segToLine(segment4.Position);

% compute the vanishing points
L = cross(l1,l2);
L = L./L(3); % vanishing point 
figure;
imshow(img);
hold all;

plot(L(1),L(2),'r.','MarkerSize',50);

[x y] = getpts;
plot (x,y,'or','MarkerSize',12);

a=[x(1) y(1) 1]';%% marker rear
text(a(1), a(2), 'a', 'FontSize', FNT_SZ, 'Color', 'k')

b=[x(2) y(2) 1]'; %% marker forward
text(b(1), b(2), 'b', 'FontSize', FNT_SZ, 'Color', 'k')

c=[x(3) y(3) 1]'; %% Center of the car 
text(c(1), c(2), 'c', 'FontSize', FNT_SZ, 'Color', 'k')

% M = cross(m1,m2);
% M = M./M(3);

%%
Va = L; %% vanishing point 
Cc = c;  %% Car center 
MR = a;  %% Marker rear
MF = b;  %% Marker forward
Mar_dis = 52 ;  %% distance between markers in real world (12 + 8) * 2 +12  = 52

Va_Cc = norm(Va - Cc);
MF_Cc = norm(MF - Cc);
Va_MR = norm(Va - MR);
MF_MR = norm(MF - MR);

% d is the real world distance of the car from MF
d = (Va_MR/MF_MR)*Mar_dis / (Va_Cc/MF_Cc)

%%

function [l] = segToLine(pts)
% convert the endpoints of a line segment to a line in homogeneous
% coordinates.
%
% pts are the endpoits of the segment: [x1 y1;
%                                       x2 y2]

% convert endpoints to cartesian coordinates
a = [pts(1,:)';1];
b = [pts(2,:)';1];
l = cross(a,b);
l = l./norm(l);
end
