include("./hemppcat_code/mppca/MPPCA.jl")
using .MPPCA
include("./hemppcat_code/hemppcat/HeMPPCAT.jl")
using .HeMPPCAT

function MPPCA_Wrapper(X::Matrix, K::Int, q::Int, d::Int)
    # TWO CLUSTERS ONLY
    D,N = size(X)
    
    c = tscWrapper(X, q; K=K)
    T1 = svd(X[:,c .== 1])
    T2 = svd(X[:,c .== 2])
    F1 = T1.U[:,1:d]*Diagonal(sqrt.(T1.S[1:d]))
    F2 = T2.U[:,1:d]*Diagonal(sqrt.(T2.S[1:d]))

    F_init = [F1 , F2] # Factor matrices
    μ_init = zeros(D, K) # Mixture means
    π_init = [0.5, 0.5] # Mixing proportions
    v_init = [1.0; 1.0]

    model = create_MPPCAModel(F_init, μ_init, π_init, v_init, N) 
    results, _ = mppca(model, X; eps=1e-6); 

    U1 = svd(results.F[1]).U
    U2 = svd(results.F[2]).U

    labelsMPPCA = clusterAssignment(X, [U1,U2], K)
    
    return labelsMPPCA
end

function hemppcatWrapperTSC(X::Matrix,K::Int,d::Int)
    D,N = size(X)
    c = tscWrapper(X, 10; K=K)
    
    T1 = svd(X[:,c .== 1])
    T2 = svd(X[:,c .== 2])
    F1 = T1.U[:,1:d]*Diagonal(sqrt.(T1.S[1:d]))
    F2 = T2.U[:,1:d]*Diagonal(sqrt.(T2.S[1:d]))
    
    F_init = [F1 , F2] # Factor matrices
    μ_init = zeros(D, K) # Mixture means
    π_init = [0.5, 0.5] # Mixing proportions
    
    v_init = ones(N)
    n = ones(N) # number of points in each variance group
    
    # Mixture HEPPCAT
    model = create_HeMPPCATModel(F_init, μ_init, π_init, v_init, n) # Initialize a HeMPPCATModel object
    Y = [c[:] for c in eachcol(X)]
    results, _ = hemppcat(model, Y; eps=1e-6); # Run HeMPPCAT
    
    U1 = svd(results.F[1]).U
    U2 = svd(results.F[2]).U
    labelsHEMPPCAT = clusterAssignment(X, [U1,U2], K)
    
    return labelsHEMPPCAT
end

function hemppcatWrapper(X::Matrix,X1::Matrix,X2::Matrix,K::Int,d::Int,p::Vector,variances::Vector)
    D = size(X)[1]
    c = ALPCAHUS(X, K, d; T=100, B=1, subspaceMethod=:pca)
    T1 = svd(X[:,c .== 1])
    T2 = svd(X[:,c .== 2])
    
    if length(T1.S) <= 3 || length(T2.S) <= 3
        T1 = svd(randn(D,d))
        T2 = svd(randn(D,d))
    end
    
    F1 = T1.U[:,1:d]*Diagonal(sqrt.(T1.S[1:d]))
    F2 = T2.U[:,1:d]*Diagonal(sqrt.(T2.S[1:d]))
    F_init = [F1 , F2] # Factor matrices
    μ_init = zeros(D, K) # Mixture means
    π_init = [0.5, 0.5] # Mixing proportions
    weighted_var = (p[1]/(p[2]+p[1]))*variances[1] + (p[2]/(p[2]+p[1]))*variances[2] # smooth variance based on points for MPPCA
    v_init = [weighted_var, weighted_var] # Noise variances
    n = [2*p[1], 2*p[2]] # number of points in each variance group
    # Run MPPCA
    model = create_MPPCAModel(F_init, μ_init, π_init, v_init, sum(n)) 
    results, _ = mppca(model, X; eps=1e-6); 
    
    U1 = 0
    U2 = 0
    resultF1 = 0
    resultF2 = 0
    
    try
        U1 = svd(results.F[1]).U
        U2 = svd(results.F[2]).U
        resultF1 = results.F[1]
        resultF2 = results.F[2]
    catch
        U1 = svd(randn(D,d)).U
        U2 = svd(randn(D,d)).U
        resultF1 = U1
        resultF2 = U2
    end
    
    labelsMPPCA = clusterAssignment(X, [U1,U2], K)
    # Mixture HEPPCAT
    model = create_HeMPPCATModel([resultF1, resultF2], μ_init, π_init, variances, n) # Initialize a HeMPPCATModel object
    Y = [hcat(X1[:,1:p[1]], X2[:,1:p[1]]), hcat(X1[:,(p[1]+1):end], X2[:,(p[1]+1):end])]
    results, _ = hemppcat(model, Y; eps=1e-6); # Run HeMPPCAT
    
    try
        U1 = svd(results.F[1]).U
        U2 = svd(results.F[2]).U
    catch
        U1 = svd(randn(D,d)).U
        U2 = svd(randn(D,d)).U
    end
    
    labelsHEMPPCAT = clusterAssignment(X, [U1,U2], K)
    return labelsMPPCA, labelsHEMPPCAT
end

function hemppcatWrapperMPPCA(X::Matrix,X1::Matrix,X2::Matrix,K::Int,d::Int,p::Vector,variances::Vector)
    D = size(X)[1]
    F1 = svd(randn(D,d)).U
    F2 = svd(randn(D,d)).U
    F_init = [F1 , F2] # Factor matrices
    μ_init = zeros(D, K) # Mixture means
    π_init = [0.5, 0.5] # Mixing proportions
    weighted_var = (p[1]/(p[2]+p[1]))*variances[1] + (p[2]/(p[2]+p[1]))*variances[2] # smooth variance based on points for MPPCA
    v_init = [weighted_var, weighted_var] # Noise variances
    n = [2*p[1], 2*p[2]] # number of points in each variance group
    # Run MPPCA
    model = create_MPPCAModel(F_init, μ_init, π_init, v_init, sum(n)) 
    results, _ = mppca(model, X; eps=1e-6); 
    
    U1 = 0
    U2 = 0
    resultF1 = 0
    resultF2 = 0
    
    try
        U1 = svd(results.F[1]).U
        U2 = svd(results.F[2]).U
        resultF1 = results.F[1]
        resultF2 = results.F[2]
    catch
        U1 = svd(randn(D,d)).U
        U2 = svd(randn(D,d)).U
        resultF1 = U1
        resultF2 = U2
    end
    
    labelsMPPCA = clusterAssignment(X, [U1,U2], K)
    # Mixture HEPPCAT
    model = create_HeMPPCATModel([resultF1, resultF2], μ_init, π_init, variances, n) # Initialize a HeMPPCATModel object
    Y = [hcat(X1[:,1:p[1]], X2[:,1:p[1]]), hcat(X1[:,(p[1]+1):end], X2[:,(p[1]+1):end])]
    results, _ = hemppcat(model, Y; eps=1e-6); # Run HeMPPCAT
    
    try
        U1 = svd(results.F[1]).U
        U2 = svd(results.F[2]).U
    catch
        U1 = svd(randn(D,d)).U
        U2 = svd(randn(D,d)).U
    end
    
    labelsHEMPPCAT = clusterAssignment(X, [U1,U2], K)
    return labelsMPPCA, labelsHEMPPCAT
end