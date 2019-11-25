%This is to create the necessary plots for the paper
%{


%% EACH DEVICe WITH ITS OUTPUTS
load struct1;
load mymap2;
close all;
Nstart = [401 303 1];%these should be odd
new_mask1 = zeros(N);
new_mask2 = zeros(N);
new_mask1(floor(Nx/2)+(-90:-75),floor(Ny/2)+100,1) = 1;
new_mask2(floor(Nx/2)+(75:90),floor(Ny/2)+100,1) = 1;


for i = 1:3
    load(sprintf('struct%d',i));
    ft = FT.e;
    ge = GT.prim;
    eq = EquationType(ft,ge);
    
    
    
    m=1;
    equ = MatrixEquation(eq,pml, osc_cell{m}.in_omega0(), eps_cell, mu_cell, s_cell{m}, J_cell, M_cell, grid3d_cell{m});
    [A,b] = equ.matrix_op();
    order = equ.r;
    e_omega = A\b;
    [~, ~, GfromF] = equ.matrixfree_op();
        
    h_omega = GfromF(e_omega);
    [E, H] = EH_from_eh(e_omega, h_omega, eq, grid3d);
        
    real_flux1 = real(E{3}.*conj(H{1}));
    
    rec_vals = zeros(1,2);
    for ii = 1:2
        out_mat = 0.5*real_flux1.*new_mask1;
        rec_vals(ii) = sum(out_mat(:))*gdl;
    end
    out_omega1 = rec_vals/denum_cell{m};
    figure();
    eps = eps_cell{3}(floor(Nx/2)+(-150:150),floor(Ny/2)+(-100:100));
    field = abs(E{3}(floor(Nx/2)+(-150:150),floor(Ny/2)+(-100:100)));
    ax1 = axes;
    imagesc(ax1,field);colormap(ax1,cmap);axis off;axis equal;
    ax2 = axes;
    contour(ax2,eps);colormap(ax2,gray);axis off;axis equal;
    linkaxes([ax1,ax2]);ax2.Visible = 'off';
    
    m=2;
    equ = MatrixEquation(eq,pml, osc_cell{m}.in_omega0(), eps_cell, mu_cell, s_cell{m}, J_cell, M_cell, grid3d_cell{m});
    [A,b] = equ.matrix_op();
    order = equ.r;
    e_omega = A\b;
    [~, ~, GfromF] = equ.matrixfree_op();
        
    h_omega = GfromF(e_omega);
    [E, H] = EH_from_eh(e_omega, h_omega, eq, grid3d);
        
    real_flux2 = real(E{3}.*conj(H{1}));
    
    rec_vals = zeros(1,2);
    for ii = 1:2
        out_mat = 0.5*real_flux2.*new_mask2;
        rec_vals(ii) = sum(out_mat(:))*gdl;
    end
    out_omega2 = rec_vals/denum_cell{m};
    figure();
    eps = eps_cell{3}(floor(Nx/2)+(-150:150),floor(Ny/2)+(-100:100));
    field = abs(E{3}(floor(Nx/2)+(-150:150),floor(Ny/2)+(-100:100)));
    ax3 = axes;
    imagesc(ax3,field);colormap(ax3,cmap);axis off;axis equal;
    ax4 = axes;
    contour(ax4,eps);colormap(ax4,gray);axis off;axis equal;
    linkaxes([ax3,ax4]);ax4.Visible = 'off';
    
end

%{
ax1 = axes;
ax2 = axes;
imagesc(ax1,abs(E{3}(floor(Nx/2)+(-150:150),floor(Ny/2)+(-100:100))));
contour(ax2,wow);
linkaxes([ax1,ax2])
ax2.Visible = 'off';
colormap(ax2,gray);
colormap(ax2,'gray');
%}

%set(gcf,'renderer','Painters')%%%%%%VERY VERY IMPORTANT!!!!
%}

%% TRANSMISSION OF THE DEVICES IN DIFFERENT FREQUENCIES

clear
close all


load struct2;
load mymap2;
close all;
N = [400 302 1];%these should be odd
new_mask1 = zeros(N);
new_mask2 = zeros(N);
new_mask1(floor(Nx/2)+(-90:-75),floor(Ny/2)+100,1) = 1;
new_mask2(floor(Nx/2)+(75:90),floor(Ny/2)+100,1) = 1;
new_mask_cell = {new_mask1,new_mask2};

denum_mask = zeros(N);
denum_mask(floor(Nx/2)+(-7:7),floor(Ny/2)+(100)) = 1; 

%i=1;
%load(sprintf('struct%d',i));
load struct2;

ft = FT.e;
ge = GT.prim;
eq = EquationType(ft,ge);

lambda = 1450:1600;
%s_array = zeros(3,length(lambda));
wow1 = zeros(1,length(lambda));
wow2 = zeros(1,length(lambda));

parfor wvn_idx = 1:length(lambda)
    osc = Oscillation(lambda(wvn_idx), unit);
    grid3d = Grid3d(osc.unit, lprim_cell, Npml, bc);
    s_factor = generate_s_factor(osc.in_omega0(), grid3d, deg_pml, R_pml);
    
    equ = MatrixEquation(eq,pml, osc.in_omega0(), eps_cell_h, mu_cell, s_factor, J_cell, M_cell, grid3d);
    [A,b] = equ.matrix_op();
    ej_new = A\b;
    [~, ~, GfromF] = equ.matrixfree_op();
    hj = GfromF(ej_new);
    [E, H] = EH_from_eh(ej_new, hj, eq, grid3d);
    real_flux = real(E{3}.*conj(H{1}));
    out_mat = 0.5*real_flux.*denum_mask;
    %denum_cell{m} = sum(out_mat(:))*(gdl*gdl);
    denum = sum(out_mat(:))*gdl;
    s_array(1,wvn_idx) = denum;
    
    equ = MatrixEquation(eq,pml, osc.in_omega0(), eps_cell, mu_cell, s_factor, J_cell, M_cell, grid3d);
    [A,b] = equ.matrix_op();
    order = equ.r;
    e_omega = A\b;
    [~, ~, GfromF] = equ.matrixfree_op();
        
    h_omega = GfromF(e_omega);
    [E, H] = EH_from_eh(e_omega, h_omega, eq, grid3d);
        
    real_flux1 = real(E{3}.*conj(H{1}));
    
    rec_vals = zeros(1,2);
    for ii = 1:2
        out_mat = 0.5*real_flux1.*new_mask_cell{ii};
        rec_vals(ii) = sum(out_mat(:))*gdl;
    end
    out_omega = rec_vals/denum;
    %s_array(2,wvn_idx) = out_omega(1);
    wow1(wvn_idx) = out_omega(1);
    %s_array(3,wvn_idx) = out_omega(2);
    wow2(wvn_idx) = out_omega(2);
    fprintf('wavelength %d of %d\n',wvn_idx,length(lambda));

end
%data_str = sprintf('str%d_test_data',i);
save('transmission_data_m2');
%}