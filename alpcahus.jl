using LinearAlgebra
using TSVD
using FlipPA

function ALPCAHUS(X::Matrix, K::Int, d::Vector; T::Int=3, B::Int=128, q::Int=24,
    subspaceMethod::Symbol=:alpcah, fastCompute::Bool=false, alpcahIter::Int=10,
    finalStep::Bool=true, varfloor::Real=1e-9, spectralStart::Bool=false,
    adaptiveRank::Bool=false, quantileAmount::Real=0.9, flipTrials::Int=100,
    flipSkip::Int=10, adaptiveMethod::Symbol=:flippa, fixedSeed::Bool=false)
    """
    Heteroscedastic subspace clustering using K-Subspaces method by applying
    ALPCAH in the subspace update step instead of PCA. Generalized to the ensemble
    method to measure similarity of clusterings in trials. Various extra things
    included such as spectral initialization for KSS, adaptive rank method, etc...

    Input:
    X is DxN data matrix where N is number of points and D is ambient dimension
    K is integer describing number of subspaces of the data e.g. K=2
    d is vector of length K that contains subspace dimensions e.g. [3;3]
    T is integer describing how many times to update subspace basis and clusters
        for each base clustering b in 1:B e.g. T=3 is very reasonable when B>>1
    B is integer for number of trials to do in the ensemble method, usually
        as many trials as time allows is best
    q is integer describing how many columns/rows to keep in affinity matrix
        constructed. Cross validation might be required but usually something
        greater than d and much less than D is sufficient

    Optional parameters:
    subspaceMethod could be :alpcah or :pca depending if KSS/EKSS is wanted
        over ALPCAHUS
    alpcahIter (integer) tells how many updates of L,R,Pi are done to get an
        estimate of the basis. Low number works well e.g. 5/10/20 no need to
        solve problem exactly at each KSS step
    finalStep (bool) true/false determines whether to take final clustering of
        spectral clustering of the affinity matrix to calculate subspace basis and
        reclassify the points one final time. Suggested to keep on all times.
    varfloor (real) prevents noise variances from going below this threshold
        during ALPCAH subspace calculation. Suggested to keep at default value
        unless one actually knows something about the quality of the data
    fastCompute (bool) determines whether to use compact SVD to determine subspace
        basis or to use partial svd methods (Krylov) for fast computation.
    spectralStart (bool) is used only when B=1 (essentially KSS meaning one trial)
        and uses TIPS to get initial clusters and performs KSS with ideally a
        large T since B=1. Highly suggested over random init for KSS.
    adaptiveRank (bool) is used when it's difficult to predict or know the
        subspace dimensions. If on, start overparameterized with a "high d" to
        start shrinking the dimension.
    adaptiveMethod (symbol) can be :flippa or :eigengap depending on what
        method one is interested in. SignFlipPA outperforms Eigengap heuristic
        experimentally so use this when possible.

    Output:
    C is vector of length N that has labels for each point
    """
    D,N = size(X)
    Cb = zeros(Int8,N,B)
    Threads.@threads for b = 1:B
        U = []
        c = zeros(Int8,N)
        for k = 1:K
            if fixedSeed == false
                push!(U,svd(randn(D,d[k])).U)
            else
                push!(U,svd(randn(Xoshiro(1234+k), D,d[k])).U)
            end
        end

        if spectralStart==false
            c = clusterAssignment(X, U, K)
        else
            c = TIPS(X, K; q=q)
        end

        for t = 1:T
            for k = 1:K
                if subspaceMethod === :pca
                    U[k] = fastPCA(X[:,c .== k],d[k]; fastCompute=fastCompute)
                end
                if subspaceMethod === :alpcah
                    U[k] = fastALPCAH(X[:,c .== k], d[k]; alpcahIter=alpcahIter, varfloor=varfloor, fastCompute=fastCompute)
                end
            end
            c = clusterAssignment(X, U, K)
            if (adaptiveRank == true) && (t%flipSkip==0)
                for k = 1:K
                    d[k] = estimateRank(X[:,c .== k]; rankMethod=adaptiveMethod, quantileAmount=quantileAmount, flipTrials=flipTrials)
                end
            end
        end
        Cb[:,b] = c
    end
    if B == 1
        return vec(Cb)
    end
    A = affinityMatrix(Cb,B)
    A = affinityTresh(A,q)
    C = spectralClustering(A,K);
    S = []
    if finalStep==true
        for k=1:K
            push!(S,fastALPCAH(X[:,C .== k], d[k]; alpcahIter=100, varfloor=varfloor, fastCompute=fastCompute))
        end
        C = clusterAssignment(X, S, K)
    end

    return C
end

function estimateRank(A::Matrix; rankMethod::Symbol=:flippa, quantileAmount::Real=0.95, flipTrials::Int=10)
    """
    Estimates rank of a matrix by using methods listed

    Input:
    A is DxN matrix where D is ambient dimension and N is number of points

    Optional:
    rankMethod (symbol) can be :flippa or :eigengap. FlipPA method works
        best especially when the data is heteroscedastic. Much harder to differentiate
        signal components from noise in this setting.
    quantileAmount (real) between 0.0 and 1.0 is a quantile metric for the
        trials done in flippa. Best to leave 0.95 or 1.0 for confidence reasons.
    flipTrials (integer) describes how many trials of permutations to do
        for flippa method. Higher is better, do as many as computation time allows.

    Output:
    Integer estimate of the subspace associated with low rank matrix A
    """
    if rankMethod === :flippa
        return flippa(A; quantile=quantileAmount, trials=flipTrials)
    elseif rankMethod === :eigengap
        return argmax( -1*diff( reverse( eigen(A*A').values ) ) )
    end
end

function fastPCA(A::Matrix, d::Int; fastCompute::Bool=false)
    """
    Returns subspace basis given data matrix A and specified dimension of basis
    by using the total least squares (TLS) method to extract U via SVD

    Input:
    A is data matrix size DxN where N is number of points and D is
        ambient dimension
    d is integer signifying subspace dim (must be known or predicted)

    Optional:
    fastCompute (bool) determines whether to use partial svd method (Krylov) or
    regular compact SVD. Multithreading safe using TSVD instead of Arpack.

    Output:
    U is Dxd subspace basis of orthonormal vectors
    """
    D,N = size(A)
    # account for edge case and restart otherwise return left singular vectors
    if N >= d
        if fastCompute == false
            return svd(A).U[:,1:d]
        else
            return tsvd(A, d)[1]
        end
    else
        return svd(randn(D,d)).U
    end
end

function fastALPCAH(Y::Matrix,d::Int; varfloor::Real=1e-9, alpcahIter::Int = 10, fastCompute::Bool=false)
    """
    Returns subspace basis given data matrix Y and specified dimension of basis
    by using a matrix factorized version of ALPCAH that treats each point as
    having its own noise variance for heteroscedastic data

    Input:
    Y is DxN data matrix of N data points and ambient dimension D
    d is integer of subspace (must be known or predicted before hand)
    varfloor is opt. parameter to keep noise variances from pushing to 0
    alpcahIter is opt. integer specifing how many iterations to run the algorithm

    Optional:
    fastCompute (bool) determines whether to use partial svd method (Krylov) or
    regular compact SVD. Multithreading safe using TSVD instead of Arpack.

    Output:
    U is Dxd subspace basis of d orthonormal vectors given ambient dimension D
    """
    rank = 0; Π = 0; v = 0; L=0; R=0;
    D,N = size(Y)
    # check edge case of not enough points and restart
    if (N >= d)
        rank = d
    else
        return svd(randn(D,d)).U
    end

    if fastCompute==false
        T = svd(Y)
        L = T.U[:,1:rank]*Diagonal(sqrt.(T.S[1:rank]))
        R = T.V[:,1:rank]*Diagonal(sqrt.(T.S[1:rank]))
    else
        T = tsvd(Y, rank)
        L = T[1]*Diagonal(sqrt.(T[2]))
        R = T[3]*Diagonal(sqrt.(T[2]))
    end

    # variance method initialization
    v = grouplessVarianceUpdate(Y, L*R'; varfloor=varfloor)
    Π = Diagonal(v.^-1)

    for i=1:alpcahIter
        # left right updates
        L = Y*Π*R*inv(R'*Π*R)
        R = Y'*L*inv(L'*L)
        # variance updates
        v = grouplessVarianceUpdate(Y, L*R'; varfloor=varfloor)
        Π = Diagonal(v.^-1)
    end
    # extract left vectors from L
    U = svd(L).U
    #U = tsvd(L,rank)[1]
    return U
end

function grouplessVarianceUpdate(Y::Matrix, L::Matrix; varfloor::Real=1e-9)
    """
    Given a data matrix Y and low rank approximation L, computes noise variances
    given residual for each point, treating each point as having its own distrubtion

    Input:
    Y,L is DxN matrix consisting of N data points of ambient dimension D

    Output:
    vector of noise variances that are prevented from going below a noise floor
    """
    v = (1/size(Y)[1])*norm.(eachcol(Y - L)).^2
    return max.(v, varfloor)
end

function cost(Y, C, K, d)
    value = 0
    D,N = size(Y)
    for i=1:K
        U = fastALPCAH(Y[:,C .== i], d[i]; alpcahIter=10)
        v = grouplessVarianceUpdate(Y[:, C .== i], U*U'*Y[:, C .== i])
        Π = Diagonal(v.^-0.5)
        value = value + 0.5*norm( (Y[:, C .== i] - U*U'*Y[:, C .== i])*Π,2)^2 + 0.5*D*sum(log10.(v))
    end
    return value
end
