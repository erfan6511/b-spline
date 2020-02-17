clear;
clc;
addpath(genpath('.\extra_files'))
%% INITIAL SETUPS:
in_lam = [1500,1550];%nm
gt_out = [1 0 ;0 1];

Nstart = [401 303 1];%these should be odd
Res = 0.25*[10^-7,10^-7,10^-7];
NPML = {[20 20],[20 20],[0 0]};
bc = [BC.p BC.p BC.p];

ft = FT.e;
ge = GT.prim;
eq = EquationType(ft,ge);

%building the system    
    
L0 = 10^-9; % this is the length everything is measured against
lam0 = 1500; %wavelength. here is 100 micro meters
unit = PhysUnit(L0);
osc = Oscillation(lam0, unit);


%generateing the grid

%***generating lprim
lprim_cell = cell(1, Axis.count);
Npml = NaN(Axis.count, Sign.count);

for w = Axis.elems
	dl_intended = Res(w)/L0;
    N = Nstart(w);
    if N~=1
        Nm = floor(N/2)+1;% index of the middle point
        Nw = round(((N-Nm)*dl_intended+(N-Nm)*dl_intended)/dl_intended);
        lprim =linspace(-(N-Nm)*dl_intended,(N-Nm)*dl_intended,Nw+1);
    else
        Nw = 1;
        lprim = linspace(0,1,Nw+1);
    end
	Npml(w,Sign.n) = NPML{w}(Sign.n);
	Npml(w,Sign.p) = NPML{w}(Sign.p);
	lprim_cell{w} = lprim;
end

%***generating grid3d
grid3d = Grid3d(osc.unit, lprim_cell, Npml, bc);

%***creating the structures
eps_cell = {ones(grid3d.N), ones(grid3d.N), ones(grid3d.N)};
mu_cell = {ones(grid3d.N), ones(grid3d.N), ones(grid3d.N)};


%for PML
pml = PML.sc;
R_pml = exp(-16);  % target reflection coefficient
deg_pml = 4;  % polynomial degree
s_factor_cell = generate_s_factor(osc.in_omega0(), grid3d, deg_pml, R_pml);

eps_node_cell = cell(1,Axis.count);
mu_node_cell = cell(1,Axis.count);

%applying boundary condition
if bc(1) == 0
    ERxx([1,end],:,:) = -inf;ERyy([1,end],:,:) = -inf;ERzz([1,end],:,:) = -inf;
elseif bc(1) == -1
    URxx([1,end],:,:) = -inf;URyy([1,end],:,:) = -inf;URzz([1,end],:,:) = -inf;
end
if bc(2) == 0
    ERxx(:,[1,end],:) = -inf;ERyy(:,[1,end],:) = -inf;ERzz(:,[1,end],:) = -inf;
elseif bc(2) == -1
    URxx(:,[1,end],:) = -inf;URyy(:,[1,end],:) = -inf;URzz(:,[1,end],:) = -inf;
end
if bc(3) == 0
    ERxx(:,:,[1,end]) = -inf;ERyy(:,:,[1,end]) = -inf;ERzz(:,:,[1,end]) = -inf;
elseif bc(3) == -1
    URxx(:,:,[1,end]) = -inf;URyy(:,:,[1,end]) = -inf;URzz(:,:,[1,end]) = -inf;
end



J_cell = cell(1, Axis.count);
M_cell = cell(1, Axis.count);
for w = Axis.elems
	J_cell{w} = zeros(grid3d.N);
    M_cell{w} = zeros(grid3d.N);
end

N = grid3d.N;

Nx = N(Axis.x);
Ny = N(Axis.y);
Nz = N(Axis.z);


%% B-SPLINE INITIALIZATION

%setting up the splines
Tx = 1:9:201;
u_vec = 1:201;

Ty = 1:9:101;
v_vec = 1:101;

k = 3;%degree
mx = length(Tx)-1;%number of knots minus 1
nx = mx-k-1;%number of control points minus 1

my = length(Ty)-1;%number of knots minus 1
ny = my-k-1;%number of control points minus 1

lenU = length(u_vec);
lenV = length(v_vec);

Nikx = zeros(nx+1,lenU);
Njky = zeros(ny+1,lenV);



for nn = 0:nx
    for ii = 1:lenU
        Nikx(nn+1,ii) = gen_N(nn+1,k,u_vec(ii),Tx);
    end
end

for nn = 0:ny
    for ii = 1:lenV
        Njky(nn+1,ii) = gen_N(nn+1,k,v_vec(ii),Ty);
    end
end

%generate control points
Pij = randi([-1,1],nx+1,ny+1);
%initialize the surface
my_surf = (Nikx.')*Pij*Njky;
%place holder for spline gradient
grad_spline = zeros(nx+1,ny+1);


%% DEVICE AND PORTS

% defining the medium
eps_sio2 = 2.1;
eps_si = 12.12;

eps_cell{Axis.z}([floor(Nx/2)-65,floor(Nx/2)-55],floor(Ny/2)+(50:151),1) = -inf;
eps_cell{Axis.z}([floor(Nx/2)+55,floor(Nx/2)+65],floor(Ny/2)+(50:151),1) = -inf;
eps_cell{Axis.z}([floor(Nx/2)-5,floor(Nx/2)+5],floor(Ny/2)+(-150:50),1) = -inf;

eps_cell{Axis.z}(floor(Nx/2)+(-100:100),floor(Ny/2)+(-50:50),1) = eps_si;
eps_cell{Axis.z}(floor(Nx/2)+(-64:-56),floor(Ny/2)+(50:151),1) = eps_si;
eps_cell{Axis.z}(floor(Nx/2)+(56:64),floor(Ny/2)+(50:151),1) = eps_si;
eps_cell{Axis.z}(floor(Nx/2)+(-4:4),floor(Ny/2)+(-150:-50),1) = eps_si;


%defining the ports
rec_cell = cell(1,2);
for i = 1:2
    rec_cell{i} = zeros(N);
end
rec_cell{1}(floor(Nx/2)+(-64:-56),floor(Ny/2)+(100),1) = 1;
rec_cell{2}(floor(Nx/2)+(56:64),floor(Ny/2)+(100),1) = 1;

%initializing the device
temp = (eps_si-eps_sio2)*double(my_surf>=0);

mask = zeros(N);
mask(floor(Nx/2)+(-100:100),floor(Ny/2)+(-50:50),1)=1;


%% PREPARING THE SIMULATION
%some initial setups
gdl = (grid3d.dl{Axis.x,GT.prim}(floor(Nx/2)));
omega0 = osc.in_omega0();

%set up the input
J_cell{Axis.z}(floor(Nx/2),floor(Ny/2)-130,1) = 1;


osc_cell= {Oscillation(in_lam(1), unit),Oscillation(in_lam(2), unit)};
grid3d_cell = {Grid3d(osc_cell{1}.unit, lprim_cell, Npml, bc),...
    Grid3d(osc_cell{2}.unit, lprim_cell, Npml, bc)};
s_cell = {generate_s_factor(osc_cell{1}.in_omega0(), grid3d_cell{1}, deg_pml, R_pml),...
    generate_s_factor(osc_cell{2}.in_omega0(), grid3d_cell{2}, deg_pml, R_pml)};

%calculating the denumenator for the flux normalization
eps_cell_h = {ones(N),ones(N),ones(N)};
eps_cell_h{3}(floor(Nx/2)+(-4:4),floor(Ny/2)+(-150:151))=eps_si;
eps_cell_h{3}([floor(Nx/2)-5,floor(Nx/2)+5],floor(Ny/2)+(-150:151))= -inf;
denum_cell = cell(1,2);
denum_mask = zeros(N);
denum_mask(floor(Nx/2)+(-4:4),floor(Ny/2)+(100)) = 1; 


for m = 1:2    
    %solve for the electric field 
    equ = MatrixEquation(eq,pml, osc_cell{m}.in_omega0(), eps_cell_h, mu_cell, s_cell{m}, J_cell, M_cell, grid3d_cell{m});
    [A,b] = equ.matrix_op();
    ej_new = A\b;
    [~, ~, GfromF] = equ.matrixfree_op();
    hj = GfromF(ej_new);
    [E, H] = EH_from_eh(ej_new, hj, eq, grid3d);
    Ez_init = E{3};
    Hx_init = H{1};
    Hy_init = H{2};
    real_flux = real(E{3}.*conj(H{1}));
    out_mat = 0.5*real_flux.*denum_mask;
    denum_cell{m} = sum(out_mat(:))*gdl;

end
%}
fileID = fopen('spline_cost.txt','w');
iter_lim = 1;
lr = {4e-3,8e-3};
%% TRAINING STAGE
for iter = 1:iter_lim  %training iteration
    for m = 1:2   %selecting the wavelength
        %solving for the electric and magnetic fields
        equ = MatrixEquation(eq,pml, osc_cell{m}.in_omega0(), eps_cell, mu_cell, s_cell{m}, J_cell, M_cell, grid3d_cell{m});
        [A,b] = equ.matrix_op();
        order = equ.r;
        e_omega = A\b;
        [~, ~, GfromF] = equ.matrixfree_op();
        
        h_omega = GfromF(e_omega);
        [E, H] = EH_from_eh(e_omega, h_omega, eq, grid3d);
        
        Ez = E{3};
        Hx = H{1};
        
        %calculate the normalized output
		rec_vals = zeros(1,2);
        for ii = 1:2
            out_mat = 0.5*real(Ez.*conj(Hx)).*rec_cell{ii};
			rec_vals(ii) = sum(out_mat(:))*gdl;
        end
        out_omega = rec_vals/denum_cell{m};
        
        cost = 0.5*sum((out_omega - gt_out(m,:)).^2);
        
        %calculating the derivative of the cost w.r.t the field
        SEz = [zeros(Nx,1),Ez(:,1:Ny-1)];
        rec_out = zeros(N);
        Srec_out = zeros(N);
        for ii = 1:2
            %Srec = [rec_cell{ii}(:,2:Ny),zeros(Nx,1)];
            Srec = [zeros(Nx,1),rec_cell{ii}(:,1:Ny-1)];
            rec_out = rec_out + rec_cell{ii}*out_omega(ii);
            Srec_out = Srec_out + Srec*out_omega(ii);
        end

        
        dfe_mat = zeros(N);
        for ii = 1:2
            Srec = [zeros(Nx,1),rec_cell{ii}(:,1:Ny-1)];
            dfe_mat=dfe_mat+(out_omega(ii)-gt_out(m,ii))*...
                ((rec_cell{ii}).*conj(Hx)/2-...
                (rec_cell{ii}).*(1i/(equ.omega*gdl)).*conj(Ez)/2+...
                (Srec).*(1i/(equ.omega*gdl)).*conj(SEz)/2);
        end
        
        %calculating the gradient w.r.t permittivity
        dfe_zero = zeros(prod(N),1);
		tot_dfe = [dfe_zero;dfe_zero;dfe_mat(:)];
		tot_dfe = tot_dfe(order);
		lam_mat = -(A.')\tot_dfe;
		grad_tempor = (lam_mat.*e_omega);
        grad_tempor = -((osc_cell{m}.in_omega0())^2)*real(grad_tempor)*gdl*gdl;
	
        grad_tempor = reshape(grad_tempor, Axis.count, prod(N));
        grad_tempor = grad_tempor(int(Axis.z), :);
        grad_tempor = reshape(grad_tempor,N);
        grad_tempor(mask==0) = mask(mask==0);
        
        %% CONTINUE TRAINING PROCESS
        
        %finding the edges for spline gradient 
        edges = zeros(size(my_surf));

        [~,vetrices,~] = isocontour(my_surf,0);
        for i = 1:size(vetrices,1)
            edges(round(vetrices(i,1)),round(vetrices(i,2)))=1;
        end
        
        %calculating teh gradient w.r.t spline control parameters
        grad_tempor = grad_tempor(floor(Nx/2)+(-100:100),floor(Ny/2)+(-50:50));
	%{
        for ix = 1:nx+1
            for jy = 1:ny+1
                spline_mat = (Nikx(ix,:).'*Njky(jy,:));
                grad_spline(ix,jy) = sum(sum(grad_tempor.*edges.*spline_mat));
            end
        end
	%}
	grad_spline = Njky*(grad_tempor.*edges)*Nikx.';
        
        Pij = Pij - lr{m}*grad_spline;
        Pij = Pij/(max(abs(Pij(:))));   %for stability of the process
   
        my_surf = (Nikx.')*Pij*Njky;    %creating the level-set surface
    
        %updating the medium
        temp = (eps_si-eps_sio2)*double(my_surf>=0);

        se = strel('disk',1);
        temp2 = temp;
        eroded = imerode(temp2,se);
        dilated = imdilate(eroded,se);
        dilated2 = imdilate(dilated,se);
        eroded2 = imerode(dilated2,se);
        eroded2([1,end],:)= max(temp(:));
        eroded2(:,[1,end])= max(temp(:));
        temp = eroded2;

        eps_cell{Axis.z}(floor(Nx/2)+(-100:100),floor(Ny/2)+(-50:50)) = ...
                temp + eps_sio2*ones(size(temp));
    
        fprintf('iteration: %04d, wavelength:%dnm, cost: %f \n',iter,in_lam(m),cost);
        fprintf(fileID,'%.4f\n',cost);
    end
end
