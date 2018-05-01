clear;
clc;

dserial = dlmread('time_serial_pres.txt');
dcuda = dlmread('time_cuda.txt');
dserial = dserial(:,1);

t = 1:36;
figure(1);
plot(t,dserial,'-b');
hold on;
plot(t,dcuda,'-r');
title('Runtime comparison','FontSize',22);
xlabel('data indices','FontSize',22) 
ylabel('time(s)','FontSize',22) 
grid on;
axis([1 36 min(dserial) max(dserial)]);

figure(2);
plot(t,dcuda,'-o');
title('CUDA runtime','FontSize',22);
xlabel('data indices','FontSize',22) 
ylabel('time(s)','FontSize',22) 
grid on;
axis([1 36 min(dcuda) max(dcuda)]);

figure(3);
plot(t,dserial,'-o');
title('Serial runtime','FontSize',22);
xlabel('data indices','FontSize',22) 
ylabel('time(s)','FontSize',22) 
grid on;
axis([1 36 min(dserial) max(dserial)]);

figure(4);
plot(t,dserial./dcuda,'-o');
title('Speedup','FontSize',22);
xlabel('data indices','FontSize',22) 
ylabel('times','FontSize',22) 
grid on;
axis([1 36 min(dserial./dcuda) max(dserial./dcuda)]);