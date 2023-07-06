function [Network2  BestCost] = TrainUsing_PSO_Fcn(Network,Xtr,Ytr)

%% Problem Statement
IW = Network.IW{1,1}; IW_Num = numel(IW);
LW = Network.LW{2,1}; LW_Num = numel(LW);
b1 = Network.b{1,1}; b1_Num = numel(b1);
b2 = Network.b{2,1}; b2_Num = numel(b2);

TotalNum = IW_Num + LW_Num + b1_Num + b2_Num;

NPar = TotalNum;

VarMin = -1*ones(1,TotalNum);
VarMax = +1*ones(1,TotalNum);

CostFuncName = 'Cost_ANN_EA';

%% Algorithm's Parameters
SwarmSize = 200;
MaxIteration = 35;
C1 = 2; % Cognition Coefficient;
C2 = 4 - C1; % Social Coefficient;
%% Initial Population
GBest.Cost = inf;
GBest.Position = [];
GBest.CostMAT = [];
for p = 1:SwarmSize
    Particle(p).Position = rand(1,NPar) .* (VarMax - VarMin) + VarMin;
    Particle(p).Cost = feval(CostFuncName,Particle(p).Position,Xtr,Ytr,Network);
    Particle(p).Velocity = [];
    Particle(p).LBest.Position = Particle(p).Position;
    Particle(p).LBest.Cost = Particle(p).Cost;
    
    if Particle(p).LBest.Cost < GBest.Cost
        GBest.Cost = Particle(p).LBest.Cost;
        GBest.Position = Particle(p).LBest.Position;
    end
end

%% Start of Optimization
for Iter = 1:MaxIteration
    %% Velocity update
    for p = 1:SwarmSize
        Particle(p).Velocity = C1 * rand * (Particle(p).LBest.Position - Particle(p).Position) + C2 * rand * (GBest.Position - Particle(p).Position);
        Particle(p).Position = Particle(p).Position + Particle(p).Velocity;
        
        Particle(p).Position = max(Particle(p).Position , VarMin);
        Particle(p).Position = min(Particle(p).Position , VarMax);        
        
        Particle(p).Cost = feval(CostFuncName,Particle(p).Position,Xtr,Ytr,Network);
        
        if Particle(p).Cost < Particle(p).LBest.Cost
            Particle(p).LBest.Position = Particle(p).Position;
            Particle(p).LBest.Cost = Particle(p).Cost;
            
            if Particle(p).LBest.Cost < GBest.Cost
                GBest.Cost = Particle(p).LBest.Cost;
                GBest.Position = Particle(p).LBest.Position;
            end
        end
    end
    %% Display
    disp(['Itretion = ' num2str(Iter) '; Best Cost = ' num2str(GBest.Cost) ';'])
    GBest.CostMAT = [GBest.CostMAT GBest.Cost];
end

GBest.Position
plot(GBest.CostMAT)
Network2 = ConsNet_Fcn(Network,GBest.Position);
BestCost = GBest.Cost;
end