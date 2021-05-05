# This dataset contains records of simulation crashes encountered during 
# climate model uncertainty quantification (UQ) ensembles. Ensemble
# members were constructed using a Latin hypercube method in LLNL’s
# UQ Pipeline software system to sample the uncertainties of 18 model
# parameters within the Parallel Ocean Program (POP2) component of 
# the Community Climate System Model (CCSM4). Three separate Latin 
# hypercube ensembles were conducted, each containing 180 ensemble
# members. We shall assume that all these ensemble members are independent
# of each other. Forty-six out of the 540 simulations failed for numerical
# reasons at combinations of parameter values.
# The goal is to use classification to predict simulation outcomes 
# (variable outcome with values ‘fail’ or ‘succeed’) from input parameter 
# values (columns 3-20), and to use sensitivity analysis and feature
# selection to determine the causes of simulation crashes. Columns 3-20 
# contain numerical values of 18 climate model parameters scaled in the 
# interval [0, 1], the scenario where ANN and SVM models are suitable.
########################################################################
dat <- read.table(file=
                    "http://archive.ics.uci.edu/ml/machine-learning-databases/00252/pop_failures.dat",
                  header=TRUE)
dim(dat)
head(dat)
anyNA(dat)

############### EDA ###########################################
###############################################################
data <- dat[, c(-1,-2)]
boxplot(data, main = "Boxplot", horizontal = F, col="orange",  border="brown")

########  Missing values by visualization ###############
library(naniar)
vis_miss(data)
gg_miss_var(data)

# DATA PREPARATION - NEED TO DEAL WITH CATEGORICAL PREDICTORS
X.dat <- as.data.frame(model.matrix(outcome~.-1, data=data))

# Keep in mind that since your network is tasked with 
# learning how to combine the inputs through a series of linear combinations
# and nonlinear activations, the input variable must be scaled.
# Beacuse your dataset may contain features highly varying in 
# magnitudes, units and range. To supress this effect, we
# need to bring all features to the same level of magnitudes.
# This scaling can be done by the following code:

# X.dat <- scale(X.dat)

# But this time, I already have the scaled data,
# so I commented the above code.

dat1 <- data.frame(cbind(X.dat, outcome=data$outcome))

#####################################################################
#####################################################################
#####################################################################
set.seed(123)
n <- nrow(dat1)
split_data <- sample(x=1:2, size = n, replace=TRUE, prob=c(0.67, 0.33))
train1 <- dat1[split_data == 1, ]
test1 <- dat1[split_data == 2, ]
yobs <- test1$outcome
test1 <- test1[ , -19]




########## Fittine the model with 1 hidden layer ################################
library(neuralnet); 
options(digits=3)
net1 <- neuralnet(outcome ~ vconst_corr+vconst_2+vconst_3+ 
                  vconst_4+vconst_5+vconst_7+ah_corr+ah_bolus+slm_corr+efficiency_factor+
                  tidal_mix_max+vertical_decay_scale+convect_corr+bckgrnd_vdc1+
                  bckgrnd_vdc_ban+bckgrnd_vdc_eq+bckgrnd_vdc_psim+Prandtl, 
                  data=train1, 
                  hidden=3, #1 hidden layer, 3 units
                  act.fct='logistic', err.fct="ce", linear.output=F, likelihood=TRUE)


# PLOT THE MODEL
plot(net1, rep="best", show.weights=T, dimension=6.5, information=F, radius=.15,
     col.hidden="red", col.hidden.synapse="black", lwd=1, fontsize=9)


# PREDICTION
ypred <- compute(net1, covariate=test1)$net.result

MSE.c <- mean((yobs-ypred)^2)
MSE.c

ypred <- ifelse(ypred>0.5,1,0)


library(cvAUC)
rf_AUC <- ci.cvAUC(predictions = ypred, labels =yobs, folds=1:NROW(test1), confidence = 0.95)
rf_AUC
(rf_auc.ci <- round(rf_AUC$ci, digits = 3))

library(verification)
mod.nn <- verify(obs = yobs, pred = ypred)
roc.plot(mod.nn, plot.thres=NULL)
text(x=0.7, y=0.2, paste("Area under ROC = ", round(rf_AUC$cvAUC, digits = 3), "with 95% CI (",
                         rf_auc.ci[1], ",", rf_auc.ci[2], ").", sep = " "), col="blue", cex =1.2)

###############################################################
############# Another neural network with 2 layers ##########################
###############################################################
nn <- neuralnet(outcome~vconst_corr+vconst_2+vconst_3+ 
                vconst_4+vconst_5+vconst_7+ah_corr+ah_bolus+slm_corr+efficiency_factor+
                tidal_mix_max+vertical_decay_scale+convect_corr+bckgrnd_vdc1+
                bckgrnd_vdc_ban+bckgrnd_vdc_eq+bckgrnd_vdc_psim+Prandtl, data=train1,
                hidden=c(5,3),linear.output=T)
plot(nn, rep="best", show.weights=T, dimension=6.5, information=F, radius=.15,
     col.hidden="red", col.hidden.synapse="black", lwd=1, fontsize=9)

###############################################################
########################## PREDICTION #########################
###############################################################
ypred <- compute(nn, covariate=test1)$net.result

ypred <- ifelse(ypred>0.5,1,0)

MSE.c <- mean((yobs-ypred)^2)
MSE.c

(miss.rate <- mean(yobs != ypred))

library(cvAUC)
rf_AUC <- ci.cvAUC(predictions = ypred, labels =yobs, folds=1:NROW(test1), confidence = 0.95)
rf_AUC
(rf_auc.ci <- round(rf_AUC$ci, digits = 3))

library(verification)
mod.nn <- verify(obs = yobs, pred = ypred)
roc.plot(mod.nn, plot.thres=NULL)
text(x=0.7, y=0.2, paste("Area under ROC = ", round(rf_AUC$cvAUC, digits = 3), "with 95% CI (",
                         rf_auc.ci[1], ",", rf_auc.ci[2], ").", sep = " "), col="blue", cex =1.2)
