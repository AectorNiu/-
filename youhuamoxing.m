%参数
c = 3e8; % 雷达速(m/s)
Sr = [0, 0]; % 真实雷达位置
A0 = [50e3, 55e3]; % 真目标初始位置(m)
V = [50, 350]; % 真目标速度(m/s)
delta_s = [500, -300]; % 雷达站址误差[Δx, Δy](m)
Sg = Sr + delta_s; % 干扰机估计的雷达位置
T = 10; % 总时间(s)
dt = 1; % 时间步长(s)
t = (0:dt:T)';
N = length(t);

% 生成真目标轨迹
A_trajectory = A0 + V .* t;

% 预设假目标轨迹
Delta_Rg = 1000; % 预设拖引量
B_desired = A_trajectory + [Delta_Rg, Delta_Rg]; 

%%未优化的实际假目标轨迹 
B_act = zeros(N, 2);
for i = 1:N
    A = A_trajectory(i, :);
    D_ASr = norm(A - Sr);
    D_ASg = norm(A - Sg);
    
    % 计算偏移角α（由雷达定位误差引起）
    alpha = atan2(delta_s(2), delta_s(1)) - atan2(A(2)-Sr(2), A(1)-Sr(1));
    
    % 实际拖引量
    Delta_Ract = Delta_Rg - (1 - cos(alpha)) * D_ASr / 2;
    B_act(i, :) = A + (A - Sr) / D_ASr * Delta_Ract;
end
%%优化多帧模型
%最小二乘法优化
fun_lsq = @(x) [B_act(:,1) - (A_trajectory(:,1) + x(1)*t + x(2)); 
                B_act(:,2) - (A_trajectory(:,2) + x(1)*t + x(2))];
x0_lsq = [0, 0];
options_lsq = optimoptions('lsqnonlin', 'Display', 'off', 'Algorithm', 'levenberg-marquardt');
x_opt_lsq = lsqnonlin(fun_lsq, x0_lsq, [], [], options_lsq);
B_opt_lsq = A_trajectory + x_opt_lsq(1)*t + x_opt_lsq(2);

%模型预测控制（MPC）
options_mpc = optimoptions('fmincon', ...
    'Display', 'off', ...
    'Algorithm', 'sqp', ...
    'SpecifyObjectiveGradient', true);

B_opt_mpc = zeros(N, 2);
H = 3; % 预测步长

for k = 1:N-H
    fun_mpc = @(x) mpc_objective_with_gradient(x, B_act, A_trajectory, t, k, H);
    x_opt_mpc = fmincon(fun_mpc, [0, 0], [], [], [], [], [-100, -100], [100, 100], [], options_mpc);
    B_opt_mpc(k:k+H, :) = A_trajectory(k:k+H, :) + x_opt_mpc(1)*t(k:k+H) + x_opt_mpc(2);
end

function [f, grad] = mpc_objective_with_gradient(x, B_act, A_trajectory, t, k, H)
    B_pred = A_trajectory(k:k+H, :) + x(1)*t(k:k+H) + x(2);
    error = B_act(k:k+H, :) - B_pred;
    f = sum(vecnorm(error, 2, 2).^2);
    
    t_extended = repelem(t(k:k+H), 1, 2);
    t_extended = t_extended(:);
    
    grad = zeros(2, 1);
    grad(1) = -2 * sum(error(:) .* t_extended);
    grad(2) = -2 * sum(error(:));
end

%遗传算法优化（全局搜索）
ga_options = optimoptions('ga', 'MaxGenerations', 50, 'PopulationSize', 100, 'Display', 'off');
fun_ga = @(x) sum(vecnorm(B_act - (A_trajectory + x(1)*t + x(2)), 2).^2);
x_opt_ga = ga(fun_ga, 2, [], [], [], [], [-1000, -1000], [1000, 1000], [], ga_options);
B_opt_ga = A_trajectory + x_opt_ga(1)*t + x_opt_ga(2);

RMSE_act = sqrt(mean(vecnorm(B_act - B_desired, 2, 2).^2));
RMSE_lsq = sqrt(mean(vecnorm(B_opt_lsq - B_desired, 2, 2).^2));
RMSE_mpc = sqrt(mean(vecnorm(B_opt_mpc - B_desired, 2, 2).^2));
RMSE_ga = sqrt(mean(vecnorm(B_opt_ga - B_desired, 2, 2).^2));


figure;
set(gcf, 'Position', [100 100 800 600], 'Color', 'w');
h_sr = scatter(Sr(1), Sr(2), 200, 'k', 'filled', 'Marker', 'pentagram', 'LineWidth', 2); hold on;
h_sg = scatter(Sg(1), Sg(2), 200, [0 0.5 1], 'filled', 'Marker', 'pentagram', 'LineWidth', 2);
h_a = scatter(A0(1), A0(2), 200, 'r', 'filled', 'Marker', 'o', 'LineWidth', 2);
h_bg = scatter(B_desired(1,1), B_desired(1,2), 200, 'b', 'filled', 'Marker', 'square', 'LineWidth', 2);
h_bact = scatter(B_act(1,1), B_act(1,2), 200, 'm', 'filled', 'Marker', 'd', 'LineWidth', 2);

plot([Sr(1), A0(1)], [Sr(2), A0(2)], 'r--', 'LineWidth', 1.5, 'Color', [1 0.5 0]); 
plot([Sg(1), A0(1)], [Sg(2), A0(2)], 'b--', 'LineWidth', 1.5); 
plot([Sg(1), B_desired(1,1)], [Sg(2), B_desired(1,2)], 'b:', 'LineWidth', 2); 
plot([Sr(1), B_act(1,1)], [Sr(2), B_act(1,2)], 'm-.', 'LineWidth', 2); 

text(Sr(1)+500, Sr(2)+500, '真实雷达', 'FontSize', 12, 'Color', 'k', 'FontWeight', 'bold');
text(Sg(1)+500, Sg(2)-500, '估计雷达', 'FontSize', 12, 'Color', [0 0.5 1], 'FontWeight', 'bold');
text(A0(1)-3000, A0(2)+3000, '真目标', 'FontSize', 12, 'Color', 'r', 'FontWeight', 'bold');
annotation('arrow', [0.4 0.5], [0.6 0.55], 'Color', 'b', 'LineWidth', 1.5, 'HeadWidth', 15);
text(mean([Sg(1), A0(1)]), mean([Sg(2), A0(2)])+1000, '\alpha', 'FontSize', 14, 'Color', 'b');

legend([h_sr, h_sg, h_a, h_bg, h_bact], {'真实雷达', '估计雷达', '真目标', '预设假目标', '实际假目标'}, ...
       'Location', 'northeast', 'FontSize', 10, 'Box', 'off');
title('单帧模型几何关系与假目标偏移', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('X (m)', 'FontSize', 12); ylabel('Y (m)', 'FontSize', 12);
grid on; axis equal; set(gca, 'FontSize', 11, 'LineWidth', 1.2);

figure;
set(gcf, 'Position', [100 100 1000 800], 'Color', 'w');

subplot(2,2,1);
plot(A_trajectory(:,1), A_trajectory(:,2), 'r-', 'LineWidth', 2); hold on;
plot(B_desired(:,1), B_desired(:,2), 'k--', 'LineWidth', 1.5);
plot(B_act(:,1), B_act(:,2), 'm-.', 'LineWidth', 1.5, 'Color', [0.8 0 0.8]);
title('未优化轨迹对比', 'FontSize', 12, 'FontWeight', 'bold');
legend('真目标', '预设轨迹', '实际轨迹', 'FontSize', 9, 'Location', 'southeast');
grid on; axis equal; set(gca, 'FontSize', 10);

subplot(2,2,2);
plot(A_trajectory(:,1), A_trajectory(:,2), 'r-', 'LineWidth', 2); hold on;
plot(B_opt_lsq(:,1), B_opt_lsq(:,2), 'Color', [0 0.6 0], 'LineStyle', ':', 'LineWidth', 2);
title('最小二乘法优化', 'FontSize', 12, 'FontWeight', 'bold');
legend('真目标', '优化轨迹', 'FontSize', 9, 'Location', 'southeast');
grid on; axis equal; set(gca, 'FontSize', 10);

subplot(2,2,3);
plot(A_trajectory(:,1), A_trajectory(:,2), 'r-', 'LineWidth', 2); hold on;
plot(B_opt_mpc(:,1), B_opt_mpc(:,2), 'Color', [0 0 0.8], 'LineStyle', '-.', 'LineWidth', 1.5);
title('模型预测控制优化', 'FontSize', 12, 'FontWeight', 'bold');
legend('真目标', '优化轨迹', 'FontSize', 9, 'Location', 'southeast');
grid on; axis equal; set(gca, 'FontSize', 10);

subplot(2,2,4);
plot(A_trajectory(:,1), A_trajectory(:,2), 'r-', 'LineWidth', 2); hold on;
plot(B_opt_ga(:,1), B_opt_ga(:,2), 'Color', [0.5 0 0.5], 'LineStyle', '--', 'LineWidth', 1.5);
title('遗传算法优化', 'FontSize', 12, 'FontWeight', 'bold');
legend('真目标', '优化轨迹', 'FontSize', 9, 'Location', 'southeast');
grid on; axis equal; set(gca, 'FontSize', 10);

sgtitle('多帧轨迹优化效果对比', 'FontSize', 14, 'FontWeight', 'bold');

error_act = vecnorm(B_act - B_desired, 2, 2);
error_lsq = vecnorm(B_opt_lsq - B_desired, 2, 2);
error_mpc = vecnorm(B_opt_mpc - B_desired, 2, 2);
error_ga = vecnorm(B_opt_ga - B_desired, 2, 2);

figure;
set(gcf, 'Position', [100 100 900 500], 'Color', 'w');

h1 = plot(t, error_act, 'm-', 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'm'); hold on;
h2 = plot(t, error_lsq, 'Color', [0 0.6 0], 'LineStyle', ':', 'LineWidth', 2, 'Marker', 's', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
h3 = plot(t, error_mpc, 'Color', [0 0 0.8], 'LineStyle', '-.', 'LineWidth', 2, 'Marker', 'd', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
h4 = plot(t, error_ga, 'Color', [0.5 0 0.5], 'LineStyle', '--', 'LineWidth', 2, 'Marker', '^', 'MarkerSize', 8, 'MarkerFaceColor', 'c');

text(t(end), error_act(end), sprintf('%.1f m', error_act(end)), 'FontSize', 10, 'Color', 'm', 'VerticalAlignment', 'bottom');
text(t(end), error_ga(end), sprintf('%.1f m', error_ga(end)), 'FontSize', 10, 'Color', [0.5 0 0.5], 'VerticalAlignment', 'top');

legend([h1, h2, h3, h4], {'未优化', '最小二乘法', 'MPC', '遗传算法'}, 'Location', 'northeast', 'FontSize', 10);
title('假目标轨迹误差随时间变化', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('时间 (s)', 'FontSize', 12); ylabel('误差 (m)', 'FontSize', 12);
grid on; set(gca, 'FontSize', 11, 'LineWidth', 1.2);
ylim([0 max(error_act)*1.1]);

num_sim = 100;
RMSE_results = zeros(num_sim, 4);

for sim = 1:num_sim
    delta_s = [500*randn, -300*randn];
    Sg = Sr + delta_s;
    B_act = zeros(N, 2);
    for i = 1:N
        A = A_trajectory(i, :);
        D_ASr = norm(A - Sr);
        alpha = atan2(delta_s(2), delta_s(1)) - atan2(A(2)-Sr(2), A(1)-Sr(1));
        Delta_Ract = Delta_Rg - (1 - cos(alpha)) * D_ASr / 2;
        B_act(i, :) = A + (A - Sr) / D_ASr * Delta_Ract;
    end
    x_opt_lsq = lsqnonlin(fun_lsq, x0_lsq, [], [], options_lsq);
    B_opt_lsq = A_trajectory + x_opt_lsq(1)*t + x_opt_lsq(2);
    RMSE_results(sim, :) = [sqrt(mean(vecnorm(B_act - B_desired, 2, 2).^2)), ...
                           sqrt(mean(vecnorm(B_opt_lsq - B_desired, 2, 2).^2)), ...
                           sqrt(mean(vecnorm(B_opt_mpc - B_desired, 2, 2).^2)), ...
                           sqrt(mean(vecnorm(B_opt_ga - B_desired, 2, 2).^2))];
end

figure;
set(gcf, 'Position', [100 100 700 500], 'Color', 'w');
boxplot(RMSE_results, 'Labels', {'未优化', 'LSQ', 'MPC', 'GA'}, 'Colors', 'kbrg', 'Widths', 0.6);
h = findobj(gca, 'Tag', 'Box');
colors = [0.9 0.6 0.6; 0.6 0.9 0.6; 0.6 0.6 0.9; 0.8 0.6 0.9];
for i = 1:length(h)
    patch(get(h(i), 'XData'), get(h(i), 'YData'), colors(i,:), 'FaceAlpha', 0.4);
end

medians = median(RMSE_results);
for i = 1:4
    text(i, medians(i), sprintf('中位数: %.1f', medians(i)), ...
         'VerticalAlignment', 'top', 'HorizontalAlignment', 'center', 'FontSize', 10);
end

title('不同优化方法的RMSE统计分布（100次模拟）', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('RMSE (m)', 'FontSize', 12); xlabel('优化方法', 'FontSize', 12);
grid on; set(gca, 'FontSize', 11, 'LineWidth', 1.2);

%% 输出RMSE对比结果
fprintf('未优化RMSE: %.2f m\n', RMSE_act);
fprintf('最小二乘法RMSE: %.2f m\n', RMSE_lsq);
fprintf('模型预测控制RMSE: %.2f m\n', RMSE_mpc);
fprintf('遗传算法RMSE: %.2f m\n', RMSE_ga);