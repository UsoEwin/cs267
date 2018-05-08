clear;
clc;

dserial = dlmread('time_serial.txt');
dcuda = dlmread('time_cuda.txt');
dserial = dserial(1:100,1);
dcuda = dcuda(1:100);
t = 1:40;
dcuda = diff(dcuda);
dcuda = dcuda(1:40);
dserial = diff(dserial);
dserial = dserial(1:40);

figure(1);
subplot(2,1,1);
semilogy(t,dserial,'-o');
set(gca,'FontSize',20)
title('Serial runtime','FontSize',22);
xlabel('Steps','FontSize',22);
ylabel('Time(s)','FontSize',22);
grid on;
hold on;
subplot(2,1,2);
semilogy(t,dcuda,'-b');
title('CUDA runtime','FontSize',22);
xlabel('Steps','FontSize',22) 
ylabel('Time(s)','FontSize',22) 
grid on;
set(gca,'FontSize',20)
axis([1 40 min(dcuda) max(dcuda)]);

figure(2);

semilogy(t,dserial./dcuda,'-o');
title('Speedup','FontSize',22);
xlabel('Steps','FontSize',22) 
ylabel('Speedup','FontSize',22) 
grid on;
set(gca,'FontSize',20)
axis([1 40 min(dserial./dcuda) max(dserial./dcuda)]);