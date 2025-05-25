using LinearAlgebra
using ParallelKMeans
using Hungarian
#using Arpack: eigs

function affinityMatrix(Cb::Matrix, B::Int)
    """
    Constructs affinity matrix based on ensemble trials of similar clusterings

    Input:
    Cb is size NxB where N is data points and B is total clusterings

    Output:
    A is NxN matrix of N data points that describes similarity
    """
    N = size(Cb)[1]
    A = zeros(N,N)
    for i = 1:N
        for j = 1:N
            if (i != j)
                A[i,j] = sum(Cb[i,:] .== Cb[j,:])
            end
        end
    end
    return (1/B)*A
end

function affinityTresh(A::Matrix, q::Int)
    """
    Thresholds any matrix A such that each row and column has roughly
    q number of nonzero values (highest values kept)

    Input:
    A is NxN affinity matrix where N is number of data points
    q is integer that describes how many values to keep, other values set to 0

    Output:
    Threshold affinity matrix to reduce noise, same size
    """
    Zrow = zeros(size(A))
    Zcol = zeros(size(A))
    N = size(A)[2]
    if q > N
        q = N
    end

    for i = 1:N
        row_index = sortperm(A[i,:], rev=true)[1:q]
        col_index = sortperm(A[:,i], rev=true)[1:q]
        Zrow[i,row_index] = A[i,row_index]
        Zcol[col_index,i] = A[col_index,i]
    end
    return 0.5*(Zrow+Zcol)
end

function spectralClustering(A::Matrix, K::Int)
    """
    Performs spectral clustering on affinity matrices using random walk
    normalized laplacian to find clusters

    Input:
    A is NxN affinity matrix where N is number of points
    K is number of subspaces integer

    Output:
    C is vector of length N that groups points using digit labels
    """
    D = Diagonal(vec(sum(A,dims=2)).^-1)
    L = I - D*A
    E = real(eigen(L).vectors[:,1:K]) #E = real(eigs(L; nev=K, which = :SM)[2])
    C = kmeans(Elkan(), E', K; k_init="k-means++").assignments
    return C
end

function TIPS(Y::Matrix, K::Int; q::Int=24)
    """
    Spectral initialization scheme using dot products to construct affinity Matrix
    Used for ALPCAHUS (B=1) instead of random init. KSS

    Input:
    Y is data matrix DxN where D is ambient dimension and N is number of points
    K is number of subspaces integer
    q is optional and decides what thresholding to do in affinity Matrix
    mostly to reduce noise in clusterings

    Output:
    C is vector of length N where each entry contains label for ith point
    """
    A = abs.(Y'*Y)
    A = A - Diagonal(diag(A))
    A = affinityTresh(A, q)
    D = Diagonal(vec(sum(A,dims=2)).^-1)
    L = I - D*A
    E = real(eigen(L).vectors[:,1:K])
    C = kmeans(Elkan(), E', K; k_init="k-means++").assignments
    return C
end

function clusterAssignment(X::Matrix, S::Vector, K::Int)
    """
    Returns clusterings based on residual distances from
    provided subspaces and data matrix

    Input:
    X is data matrix DxN where D is ambient dim. and N is # of points
    S is a vector where each entry is a subspace
    K is int describing # of subspaces

    Output:
    C is a vector of length N where each entry is a label
    """
    N = size(X)[2]
    C = zeros(N,K)
    for k=1:K
        C[:,k] = norm.(eachcol(S[k]'*X))
    end
    C = argmax(C, dims=2)[:]
    C = getindex.(C,2)
    return C
end

function clusterError(trueLabels::Vector, estimatedLabels::Vector)
    """
    Returns cluster error given true labels and estimated labels vectors
    accounts for permutations of true labels by using hungarian Algorithm

    Input:
    trueLabels/estimatedLabels are vectors of length N # of points
    each element is some int describing grouping

    Output:
    Returns percentage of points 'misclassified'
    """
    estimatedLabels = hungarianAlgorithm(trueLabels, estimatedLabels)
    error = 100*sum(trueLabels .!= estimatedLabels)/length(trueLabels)
    return error
end

function hungarianAlgorithm(trueLabels::Vector, estimatedLabels::Vector)
    """
    Hungarian algorithm to deal with permuted labels to calculate accurate
    clustering error. Big thanks to John Lipor for the Python implementation.

    Input:
    trueLabels/estimatedLabels vectors of length N # of points with groupings

    Output:
    outLabels is a permuted version of estimatedLabels that is 're-oriented'
    based on trueLabels
    """
    trueLabelVals = unique(trueLabels)
    trueCount = length(trueLabelVals)
    estimatedLabelVals = unique(estimatedLabels)
    estimatedCount = length(estimatedLabelVals)
    # cost matrix
    costMatrix = zeros(estimatedCount, trueCount)
    for i in range(1,estimatedCount)
        indices = (estimatedLabels .== estimatedLabelVals[i])
        for j in range(1,trueCount)
            costMatrix[i,j] = sum(trueLabels[indices] .== trueLabelVals[j])
        end
    end
    # hungarian algorithm
    columnIndex,_ = hungarian(-costMatrix)
    rowIndex = range(1,estimatedCount)
    # label generation
    outLabels = zeros(Int,length(estimatedLabels))
    for i in range(1,estimatedCount)
        outLabels[estimatedLabels .== estimatedLabelVals[rowIndex[i]]] .= trueLabelVals[columnIndex[i]]
    end
    outLabelVals = unique(outLabels)

    if length(outLabelVals) < maximum(outLabels)
        lVal = 1
        for i in range(1,length(outLabelVals))
            outLabels[outLabels .== outLabelVals[i]] .= lVal
            lVal = lVal + 1
        end
    end

    return outLabels
end
