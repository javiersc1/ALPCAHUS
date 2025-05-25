using LinearAlgebra
using Distributions
using Random
using Statistics
using Plots
using Hungarian

function generateSubspace(ambientSpace::Int, latentSpace::Int; seed::Int=rand(1:100000))
    return svd(rand(Xoshiro(seed),Normal(0,1),ambientSpace,latentSpace)).U
end

function generateData(U::Matrix, v::Vector = ones(2), points::Vector = 10*ones(Int,2); coordinateWindow::Real = 10, coordinateType::Symbol = :uniform, seed::Int=rand(1:100000))
    ambientSpace, dimSubspace = size(U)
    totalPoints = sum(points)
    if coordinateType === :uniform
        X = U*rand(Xoshiro(seed),Uniform(-coordinateWindow,coordinateWindow),dimSubspace,totalPoints)
    end
    if coordinateType === :gaussian
        X = U*rand(Xoshiro(seed), Normal(0, (1/sqrt(3))*coordinateWindow ), dimSubspace, totalPoints)
    end
    Y = zeros(ambientSpace,totalPoints)
    Y[:,1:points[1]] = X[:,1:points[1]] +  rand(Xoshiro(seed),Normal(0,sqrt(v[1])), ambientSpace, points[1])
    Y[:,(points[1]+1):end] = X[:,(points[1]+1):end] +  rand(Xoshiro(seed),Normal(0,sqrt(v[2])), ambientSpace, points[2])
    return Y
end

function generateDataHomo(U::Matrix, v::Real = 0.1, totalPoints::Int = 6; coordinateWindow::Real = 10, coordinateType::Symbol = :uniform, seed::Int=rand(1:100000))
    ambientSpace, dimSubspace = size(U)
    if coordinateType === :uniform
        X = U*rand(Xoshiro(seed),Uniform(-coordinateWindow,coordinateWindow),dimSubspace,totalPoints)
    end
    if coordinateType === :gaussian
        X = U*rand(Xoshiro(seed), Normal(0, (1/sqrt(3))*coordinateWindow ), dimSubspace, totalPoints)
    end
    Y = X +  rand(Xoshiro(seed),Normal(0,sqrt(v)), ambientSpace, totalPoints)
    return Y
end

function generateHeatmap(x, y, tensor; plotType::Symbol=:median, methodType="ALPCAHUS", goodPoints::Int=6, ν1::Real=0.1)
    if plotType === :median
        matrix = median(tensor,dims=3)[:,:,1]
        header = "Median"
        lim=(0,50)
    end
    if plotType === :mean
        matrix = mean(tensor,dims=3)[:,:,1]
        header = "Mean"
        lim=(0,50)
    end
    if plotType === :variance
        matrix = var(tensor,dims=3)[:,:,1]
        header = "Variance"
        lim=(0,maximum(vec(matrix)))
    end
    heatmap(x, y, matrix, title="", xlabel="Point Ratio N2/N1, N1=$goodPoints", ylabel="Variance Ratio ν2/ν1, ν1=$ν1",yflip=true, clim=lim,xticks=round.(Int,x),yticks=round.(Int,y),     legendfontpointsize=12,xtickfontsize=12,ytickfontsize=12,guidefontsize=12,titlefontsize=18)
    fontsize = 16
    nrow, ncol = size(matrix)
    ann = [(x[j],y[i], text(round(matrix[i,j], digits=1), fontsize, :white, :center)) for i in 1:nrow for j in 1:ncol]
    annotate!(ann, linecolor=:white)
end

function generateHeatmap2(x, y, tensor; plotType::Symbol=:median, methodType="ALPCAHUS", badPoints::Int=500, ν1::Real=0.1)
    if plotType === :median
        matrix = median(tensor,dims=3)[:,:,1]
        header = "Median"
        lim=(0,50)
    end
    if plotType === :mean
        matrix = mean(tensor,dims=3)[:,:,1]
        header = "Mean"
        lim=(0,50)
    end
    if plotType === :variance
        matrix = var(tensor,dims=3)[:,:,1]
        header = "Variance"
        lim=(0,maximum(vec(matrix)))
    end
    xRange=1:length(x)
    yRange=1:length(y)
    heatmap(matrix, title="", xlabel="N1, N2=$badPoints", ylabel="Variance Ratio ν2/ν1, ν1=$ν1",yflip=true, clim=lim,xticks=(xRange,round.(Int,x)),yticks=(yRange,round.(Int,y)), legendfontpointsize=12,xtickfontsize=12,ytickfontsize=12,guidefontsize=12,titlefontsize=18)
    fontsize = 16
    nrow, ncol = size(matrix)
    ann = [(xRange[j],yRange[i], text(round(matrix[i,j], digits=1), fontsize, :white, :center)) for i in 1:nrow for j in 1:ncol]
    annotate!(ann, linecolor=:white)
end

function generateHeatmap3(x, y, tensor; plotType::Symbol=:median, methodType="ALPCAHUS", startingPoints::Int=6, v::Real=0.1)
    if plotType === :median
        matrix = median(tensor,dims=3)[:,:,1]
        header = "Median"
        lim=(0,50)
    end
    if plotType === :mean
        matrix = mean(tensor,dims=3)[:,:,1]
        header = "Mean"
        lim=(0,50)
    end
    if plotType === :variance
        matrix = var(tensor,dims=3)[:,:,1]
        header = "Variance"
        lim=(0,maximum(vec(matrix)))
    end
    heatmap(x, y, matrix, title=methodType*" - "*header*" Clustering Error %", xlabel="Point Multiplier, N=$startingPoints", ylabel="Variance Multiplier, v=$v",yflip=true, clim=lim,xticks=round.(Int,x),yticks=round.(Int,y))
    fontsize = 15
    nrow, ncol = size(matrix)
    ann = [(x[j],y[i], text(round(matrix[i,j], digits=1), fontsize, :white, :center)) for i in 1:nrow for j in 1:ncol]
    annotate!(ann, linecolor=:white)
end

function subspaceError(Y, U)
    return norm(Y - U*U'*Y)/norm(Y)
end

function reconError2(Y, C, K, d)
    error = 0
    for i=1:K
        U = fastALPCAH(Y[:, C .== i], d[i]; alpcahIter= 10);
        error = error + subspaceError(Y[:, C .== i], U)
    end
    return error
end

function reconError(Y, C, K, d)
    error = 0
    for i=1:K
        U = fastPCA(Y[:, C .== i], d[i]);
        error = error + subspaceError(Y[:, C .== i], U)
    end
    return error
end

function affinityError(Ut::Matrix, U::Matrix)
    return norm(U*U' - Ut*Ut', 2)/norm(Ut*Ut', 2)
end

function reconErrorGroundTruth(Y, C, Ctrue, K, d)
    C = hungarianAlgorithm(Ctrue, C)
    error = 0
    for i=1:K
        U = fastALPCAH(Y[:, C .== i], d[i]; alpcahIter= 10);
        Utrue = fastALPCAH(Y[:, Ctrue .== i], d[i]; alpcahIter= 10);
        error = error + affinityError(Utrue, U)
    end
    return error
end

function reconErrorGroundTruthPCA(Y, C, Ctrue, K, d)
    C = hungarianAlgorithm(Ctrue, C)
    error = 0
    for i=1:K
        U = fastPCA(Y[:, C .== i], d[i]);
        Utrue = fastALPCAH(Y[:, Ctrue .== i], d[i]; alpcahIter= 10);
        error = error + affinityError(Utrue, U)
    end
    return error
end

function IOU(prediction::AbstractVector, ground_truth::AbstractVector, num_classes::Int)
    iou_sum = 0.0

    for class in 0:(num_classes-1)
        # Calculate intersection and union for the current class
        intersection = sum((prediction .== class) .& (ground_truth .== class))
        union = sum((prediction .== class) .| (ground_truth .== class))

        # Avoid division by zero
        if union != 0
            iou_sum += intersection / union
        end
    end

    # Calculate mean IOU
    mean_iou = iou_sum / num_classes
    return mean_iou
end