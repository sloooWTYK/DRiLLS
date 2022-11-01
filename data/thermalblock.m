function [sample_set,f_data, f_deri]= thermalblock(testcase,n_samp)
disp('thermal block for different configurations and corresponding plots');
%testcase = 1;
%n_samp = 500;

plot_params = [];
plot_params.axis_equal = 1;
plot_params.axis_tight = 1;

switch testcase
    case 1  % 9-dim case
        fprintf(1, '9-dim parametric case\n');
        params = [];
        params.B1 = 3;
        params.B2 = 3;
        params.mu_range = [0.1;10];
        params.numintervals_per_block = 20;
        model = standalone_thermalblock_model(params); % model parameter setting
        model_data = gen_model_data(model); % load precomputed matrices and grids
    case 2
        fprintf(1, '16-dim parametric case\n');
        params = [];
        params.B1 = 4;
        params.B2 = 4;
        params.mu_range = [0.1;10];
        params.numintervals_per_block = 15;
        model = standalone_thermalblock_model(params);
        model_data = gen_model_data(model);
    case 3
        fprintf(1, '25-dim parametric case\n');
        params = [];
        params.B1 = 5;
        params.B2 = 5;
        params.mu_range = [0.1;10];
        params.numintervals_per_block = 12;
        model = standalone_thermalblock_model(params);
        model_data = gen_model_data(model);
        
    case 4
        fprintf(1, '36-dim parametric case\n');
        params = [];
        params.B1 = 6;
        params.B2 = 6;
        params.mu_range = [0.1;10];
        params.numintervals_per_block = 10;
        model = standalone_thermalblock_model(params);
        model_data = gen_model_data(model);
end

sample_set = params.mu_range(1)*ones(n_samp,1)+...
    (params.mu_range(2)-params.mu_range(1))*rand(n_samp, params.B1*params.B2);
f_data = zeros(n_samp,1);
f_deri = zeros(n_samp,params.B1*params.B2);
for i = 1:n_samp
    mu = sample_set(i,:);
    model = set_mu(model,mu);
    [sim_data, sim_deri] = detailed_simulation(model,model_data);
    f_data(i,1) = sim_data.s;
    f_deri(i,:) = sim_deri';
    if ismember(i, [1,fix(n_samp/2),n_samp])
%         figure(i)
%         plot_sim_data(model,model_data,sim_data,plot_params);
    end
end

return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% model generation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function model = standalone_thermalblock_model(params)
% Thermal Block example
%
% i.e.
% - div ( a(x) grad u(x)) = q(x)    on Omega
%                   u(x)) = g_D(x)  on Gamma_D
%     a(x) (grad u(x)) n) = g_N(x)  on Gamma_N
%                       s = l(u) linear output functional
%
% where Omega = [0,1]^2 is divided into B1 * B2 blocks
% QA := B1*B2. The heat conductivities are given by mu_i:
%
%   -------------------------
%   | ...   | ..... | mu_QA |
%   -------------------------
%   | ....  | ...   | ..... |
%   -------------------------
%   | mu_1  | ...   | mu_B1 |
%   -------------------------
%
% `Gamma_D =` upper edge
% `Gamma_N = boundary(Omega)\Gamma_D`
%
% `a(x) = mu_i(x) if x\in B_i`
% `q(x) = 0`
% `g_D  = 0` on top
% `g_N  = 1` on lower edge, 0 otherwise
% `l(u)` = average over lower edge
%
% The discretization information is fully cut out here and loaded
% from disc in gen_model_data

model = [];
if nargin == 0
    params.B1=3;
    params.B2=3;
end
model.B1 = params.B1;
model.B2 = params.B2;
model.number_of_blocks = params.B1*params.B2;

if ~isfield(params,'numintervals_per_block')
    params.numintervals_per_block = 5;
end
model.numintervals_per_block = params.numintervals_per_block;

mu_names = {};
mu_ranges = {};
if ~isfield(params,'mu_range')
    params.mu_range = [0.1,10];
end
mu_range = params.mu_range;
for p = 1:model.number_of_blocks
    mu_names = [mu_names,{['mu',num2str(p)]}];
    mu_ranges = [mu_ranges,{mu_range}];
end
model.mu_names = mu_names;
model.mu_ranges = mu_ranges;

%default values 1 everywhere
model.mus = ones(model.number_of_blocks,1);

% basis generation settings
%model.RB_train_rand_seed = 100;
model.RB_train_size = 1000;
model.RB_stop_epsilon = 1e-3;
model.RB_stop_Nmax = 100;
model.RB_generation_mode = 'greedy_rand_uniform_with_orthogonalization';
return


function model = set_mu(model,mu)
if length(mu)~=model.number_of_blocks
    error('length of mu does not fit to number of blocks!');
end
model.mus = [mu(:)];
return


function model_data = gen_model_data(model)
% simply load precomputed matrices and grid information
model_data_varname = ['model_data_',num2str(model.B1),'_',num2str(model.B2),'_',...
    num2str(model.numintervals_per_block)];
load('data_rb_tutorial_standalone',model_data_varname);
eval(['model_data =', model_data_varname,';']);
return

function [sim_data, sim_deri] = detailed_simulation(model,model_data)
% high-fidelity model
sim_data =[];
A_coeff = model.mus;
Q_A = length(A_coeff);
A =  A_coeff(1) * model_data.A_comp{1};
for q=2:Q_A
    if A_coeff(q)~=0
        A = A + A_coeff(q)*model_data.A_comp{q};
    end
end
uh = A\model_data.f;
sim_data.uh = uh;
sim_data.s = (model_data.l(:)') * sim_data.uh;

sim_deri = zeros(Q_A, 1);
if nargout>1
    y = (A')\model_data.l(:); % adjoint problem solution
    for q=1:Q_A
        sim_deri(q) = -y' * model_data.A_comp{q} * uh;
    end
end

return


function p = plot_sim_data(model,detailed_data,sim_data, ...
    plot_params)
if nargin<4
    plot_params = [];
end

% expand vector to full size including dirichlet values
uh_ext = zeros(length(detailed_data.grid.X),1);
uh_ext(detailed_data.grid.non_dirichlet_indices) = sim_data.uh(:);

p = plot_vertex_data(detailed_data.grid,uh_ext,plot_params);
hold on;

% plot coarse mesh
if ~isfield(plot_params,'plot_blocks')
    plot_params.plot_blocks = 1;
end
if plot_params.plot_blocks
    X = [0:1/model.B1:1];
    Y = [0:1/model.B2:1];
    l1 = line([X;X],...
        [zeros(1,model.B1+1);...
        ones(1,model.B1+1)]);
    set(l1,'color',[0,0,0],'linestyle','-.');
    %keyboard;
    l2 = line([zeros(1,model.B2+1);...
        ones(1,model.B2+1)],...
        [Y;Y]);
    set(l2,'color',[0,0,0],'linestyle','-.');
    p = [p(:);l1(:);l2(:)];
end
return


function p = plot_vertex_data(grid,data,plot_params)
if nargin<3
    plot_params = [];
end
if ~isfield(plot_params,'title')
    plot_params.title = '';
end
if ~isfield(plot_params,'axis_equal')
    plot_params.axis_equal = 0;
end
if ~isfield(plot_params,'no_lines')
    plot_params.no_lines = 1;
end
if ~isfield(plot_params,'show_colorbar')
    plot_params.show_colorbar = 1;
end
if ~isfield(plot_params,'colorbar_location')
    plot_params.colorbar_location = 'EastOutside';
end
% expand vector to full size including dirichlet values
XX = grid.X(grid.VI(:));
XX = reshape(XX,size(grid.VI)); % nelements*nneigh matrix
YY = grid.Y(grid.VI(:));
YY = reshape(YY,size(grid.VI)); % nelements*nneigh matrix
CC = data(grid.VI);
p = patch(XX',YY',0*CC', CC');
c = jet(256);
colormap(c);
if plot_params.axis_equal
    axis equal;
    axis tight;
end
if plot_params.no_lines
    set(p,'linestyle','none');
end
if plot_params.show_colorbar
    if isfield(plot_params,'clim')
        set(gca,'Clim',plot_params.clim)
    end
    colorbar(plot_params.colorbar_location);
end
title(plot_params.title);
return


