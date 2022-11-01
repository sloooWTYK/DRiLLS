x = -1:.01:1; y = x; [xx, yy] = meshgrid(x, y);
f1 = exp(-20*((xx-0.5).^2+(yy-0.5).^2));
f2 = exp(-20*((xx-0.5).^2+(yy+0.5).^2));
f3 = exp(-20*((xx+0.5).^2+(yy-0.5).^2));
f4 = exp(-20*((xx+0.5).^2+(yy+0.5).^2));
% f1 = exp(-6*((xx-1).^2+(yy-1).^2));
% f2 = exp(-6*((xx-1).^2+(yy+1).^2));
% f3 = exp(-6*((xx+1).^2+(yy-1).^2));
% f4 = exp(-6*((xx+1).^2+(yy+1).^2));
%mesh(xx,yy,f1+f4)
%contour(xx,yy,xx.^2-yy.^2)
contour(xx,yy,f1+f2+f3+f4,100)
%mesh(xx,yy,f1)