%参数
c = 3e8; % 雷达速 (m/s)
D_ASr = norm([50e3, 55e3]); % 初始距离 (m)
delta_tau = 1e-6; % 转发时延 (s)
Delta_Rg = c * delta_tau / 2; % 预设拖引量 (m)
S_r = [0, 0]; % 雷达真实位置 (原点)
delta_s = [500, -300]; % 位置误差 (m)
S_g = S_r + delta_s;%干扰机估计的雷达位置
% 真目标位置 (单帧假设静止)
A = [50e3, 55e3]; 
% 计算预设假目标位置 B_g
B_g = S_g + (A - S_g) / norm(A - S_g) * Delta_Rg;
% 计算实际假目标位置 B_act (考虑角度偏差)
a = atan2(delta_s(2), delta_s(1)) - atan2(A(2)-S_r(2), A(1)-S_r(1)); 
D_ASact = D_ASr * cos(a);
Delta_Ract = Delta_Rg - (1 - cos(a)) * D_ASr / 2;
B_act = A * (1 + Delta_Ract / D_ASr);

figure;
plot(S_r(1), S_r(2), 'ro', 'MarkerSize', 10, 'LineWidth', 2); hold on;
plot(A(1), A(2), 'go', 'MarkerSize', 10, 'LineWidth', 2);
plot(B_g(1), B_g(2), 'b*', 'MarkerSize', 10, 'LineWidth', 2);
plot(B_act(1), B_act(2), 'm*', 'MarkerSize', 10, 'LineWidth', 2);
legend('雷达', '真目标', '预设假目标', '实际假目标');
title('单帧假目标位置对比');
grid on; axis equal;

T_total = 100; % 总时长 (s)
dt = 1; % 时间步长 (s)
Vx = 50; % 真目标速度 x方向 (m/s)
Vy = 350; % 真目标速度 y方向 (m/s)
t = 0:dt:T_total;
N = length(t);
A_traj = zeros(N, 2); % 真目标轨迹
B_g_traj = zeros(N, 2); % 预设假目标轨迹
B_act_traj = zeros(N, 2); % 实际假目标轨迹
% 初始位置
A_traj(1, :) = [50e3, 55e3]; 

% 生成轨迹
for i = 1:N-1
    A_traj(i+1, :) = A_traj(i, :) + [Vx*dt, Vy*dt]; % 真目标运动
    D_ASr = norm(A_traj(i+1, :)); % 更新距离
    
    % 计算预设假目标位置
    B_g_traj(i+1, :) = S_g + (A_traj(i+1, :) - S_g) / norm(A_traj(i+1, :) - S_g) * Delta_Rg;
    
    % 计算实际假目标位置
    a = atan2(delta_s(2), delta_s(1)) - atan2(A_traj(i+1,2)-S_r(2), A_traj(i+1,1)-S_r(1));
    D_ASact = D_ASr * cos(a);
    Delta_Ract = Delta_Rg - (1 - cos(a)) * D_ASr / 2;
    B_act_traj(i+1, :) = A_traj(i+1, :) * (1 + Delta_Ract / D_ASr);
end

%% 可视化
figure;
plot(S_r(1), S_r(2), 'ro', 'MarkerSize', 10, 'LineWidth', 2); hold on;
plot(A_traj(:,1), A_traj(:,2), 'g-', 'LineWidth', 1.5);
plot(B_g_traj(:,1), B_g_traj(:,2), 'b--', 'LineWidth', 1.5);
plot(B_act_traj(:,1), B_act_traj(:,2), 'm-.', 'LineWidth', 1.5);
legend('雷达', '真目标轨迹', '预设假目标轨迹', '实际假目标轨迹');
title('多帧假目标轨迹对比');
grid on; axis equal;
%蒙特卡洛方法分析统计特征
num_samples = 1000; % 样本数
sigma_x = 500; % 横向误差标准差 (m)
sigma_y = 300; % 纵向误差标准差 (m)
delta_s_samples = mvnrnd([0, 0], diag([sigma_x^2, sigma_y^2]), num_samples);

B_act_samples = zeros(num_samples, 2);
Delta_B_samples = zeros(num_samples, 2);

A = [50e3, 55e3]; % 固定真目标位置
for i = 1:num_samples
    S_g = S_r + delta_s_samples(i, :); % 随机误差
    B_g = S_g + (A - S_g) / norm(A - S_g) * Delta_Rg;
    
    a = atan2(delta_s_samples(i,2), delta_s_samples(i,1)) - atan2(A(2)-S_r(2), A(1)-S_r(1));
    D_ASr = norm(A - S_r);
    D_ASact = D_ASr * cos(a);
    Delta_Ract = Delta_Rg - (1 - cos(a)) * D_ASr / 2;
    B_act = A * (1 + Delta_Ract / D_ASr);
    
    B_act_samples(i, :) = B_act;
    Delta_B_samples(i, :) = B_act - B_g;
end

mean_DeltaB = mean(Delta_B_samples);
cov_DeltaB = cov(Delta_B_samples);
figure;
scatter(Delta_B_samples(:,1), Delta_B_samples(:,2), 10, 'filled');
title('假目标位置偏移分布');
xlabel('横向偏移 (m)'); ylabel('纵向偏移 (m)');
grid on;
